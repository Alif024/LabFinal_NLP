"""ประเมินระบบ RAG ด้วย pipeline เดียวกับหน้าเว็บ (rag.py)

1. การค้นหา: เทียบ vector / BM25 / hybrid ด้วย test_questions.csv (ระดับไฟล์)
   และ scripts/retrieval_dev.csv (คำถามภาษาพูด 30 ข้อ ตรวจระดับรายการยา)
2. การตอบคำถาม (เรียก LLM จริง): ปฏิเสธถูกต้อง, อ้างอิงเอกสารที่ถูก, มีข้อเท็จจริงสำคัญครบ
3. การสนทนาต่อเนื่อง: คำถามที่ 2 ต้องเข้าใจบริบทจากคำถามแรก

รัน: python scripts/evaluate.py [--retrieval-only] [--k 5]
ต้องมี GROQ_API_KEY ใน environment หรือ .streamlit/secrets.toml
"""
import argparse
import csv
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import rag  # noqa: E402

CONVERSATIONS = [
    (["พาราเซตามอลกินได้วันละไม่เกินเท่าไร", "ถ้าดื่มเหล้าเป็นประจำล่ะ"], "2000"),
    (["ยาแก้แพ้ลดน้ำมูกที่เป็นยาสามัญประจำบ้านคือยาอะไร", "ยานี้มีผลข้างเคียงอะไรบ้าง"], "ง่วง"),
]


def matches(chunk, targets):
    """targets เช่น "16:เฟอร์รัส|03:" = ไฟล์ขึ้นต้น 16_ และหัวข้อมีคำว่าเฟอร์รัส หรือไฟล์ใดก็ได้ที่ขึ้นต้น 03_"""
    for target in targets.split("|"):
        prefix, _, section = target.partition(":")
        if chunk.file.startswith(prefix + "_") and section in chunk.section:
            return True
    return False


def has_facts(answer, facts):
    """facts คั่นด้วย ; (ต้องมีครบ) และ | (มีอย่างใดอย่างหนึ่ง) เทียบแบบไม่สนตัวพิมพ์ ช่องว่าง และจุลภาค"""
    norm = lambda s: s.lower().replace(",", "").replace(" ", "")
    return all(any(norm(alt) in norm(answer) for alt in fact.split("|")) for fact in facts.split(";") if fact)


def evaluate_retrieval(index, test_rows, rewrite=None):
    dev = list(csv.DictReader(open(ROOT / "scripts" / "retrieval_dev.csv", encoding="utf-8")))
    test = [{"query": r["question"],
             "targets": "|".join(f.strip().split("_")[0] + ":" for f in r["source_file"].split(";"))}
            for r in test_rows if r["answerable"] == "yes"]
    print("## การค้นหา (hit@k = สัดส่วนคำถามที่เจอเอกสารที่ถูกใน k อันดับแรก)\n")
    print("| วิธีค้นหา | ชุดข้อมูล | hit@1 | hit@3 | hit@5 | MRR |")
    print("|---|---|---|---|---|---|")
    methods = [("vector (e5 + FAISS)", True, False, False),
               ("BM25 (คำ + พยางค์)", False, True, False),
               ("hybrid (RRF)", True, True, False)]
    if rewrite:
        methods.append(("hybrid + query rewriting (LLM)", True, True, True))
    for name, use_vector, use_bm25, use_rewrite in methods:
        for label, rows in ((f"test {len(test)} ข้อ", test), (f"dev {len(dev)} ข้อ", dev)):
            ranks = []
            for r in rows:
                queries = rewrite(r["query"]).queries if use_rewrite else r["query"]
                hits = index.search(queries, 10, use_vector, use_bm25)
                ranks.append(next((h.rank for h in hits if matches(h.chunk, r["targets"])), None))
            hit = lambda k: sum(1 for x in ranks if x and x <= k) / len(ranks)
            mrr = sum(1 / x for x in ranks if x) / len(ranks)
            print(f"| {name} | {label} | {hit(1):.2f} | {hit(3):.2f} | {hit(5):.2f} | {mrr:.2f} |")
    print()


def ask(index, client, model, question, history, k):
    query = rag.rewrite_query(client, model, question, history)
    hits = index.search(query.queries, k)
    text = "".join(rag.answer_stream(client, model, question, hits, history))
    answer, cited = rag.finalize_answer(text, len(hits))
    return query, hits, answer, cited


def evaluate_answers(index, client, model, test_rows, k):
    print(f"## การตอบคำถาม (LLM: {model}, top-k = {k})\n")
    passed = 0
    for r in test_rows:
        _, hits, answer, cited = ask(index, client, model, r["question"], [], k)
        refused = rag.is_not_found(answer, cited)
        cited_files = [hits[i - 1].chunk.file for i in cited]
        if r["answerable"] == "yes":
            expected = [f.strip() for f in r["source_file"].split(";")]
            checks = {"ตอบได้": not refused,
                      "อ้างอิงถูกไฟล์": any(f in cited_files for f in expected),
                      "ข้อเท็จจริงครบ": has_facts(answer, r["key_facts"])}
        else:
            checks = {"ตอบว่าไม่พบข้อมูล": refused}
        ok = all(checks.values())
        passed += ok
        status = "ผ่าน" if ok else "ไม่ผ่าน: " + ", ".join(n for n, v in checks.items() if not v)
        print(f"[{r['id']}] {r['question']} -> {status}")
        print("    " + answer.replace("\n", "\n    "))
        print(f"    อ้างอิง: {cited_files or '-'}\n")
    print(f"สรุป: ผ่าน {passed}/{len(test_rows)} ข้อ\n")
    return passed


def evaluate_conversations(index, client, model, k):
    print("## การสนทนาต่อเนื่อง\n")
    passed = 0
    for turns, fact in CONVERSATIONS:
        history = []
        for q in turns:
            query, _, answer, _ = ask(index, client, model, q, history, k)
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
        ok = has_facts(answer, fact)
        passed += ok
        print(f"{' -> '.join(turns)}\n    คำถามที่ใช้ค้นหา: {query.question}\n    คำค้นเพิ่มเติม: {query.keywords}"
              f"\n    {answer[:300]}\n    "
              f"{'ผ่าน' if ok else 'ไม่ผ่าน'} (ต้องมี: {fact})\n")
    print(f"สรุป: ผ่าน {passed}/{len(CONVERSATIONS)} บทสนทนา")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    index = rag.RagIndex()
    test_rows = list(csv.DictReader(open(ROOT / "test_questions.csv", encoding="utf-8")))
    print(f"คลังเอกสาร: {len(index.docs)} เอกสาร, {len(index.chunks)} chunks\n")
    if args.retrieval_only:
        evaluate_retrieval(index, test_rows)
        return

    from groq import Groq
    secrets_file = ROOT / ".streamlit" / "secrets.toml"
    secrets = tomllib.loads(secrets_file.read_text(encoding="utf-8")) if secrets_file.exists() else {}
    client = Groq(api_key=os.environ.get("GROQ_API_KEY") or secrets["GROQ_API_KEY"], max_retries=6)
    model = os.environ.get("GROQ_MODEL") or secrets.get("GROQ_MODEL", rag.DEFAULT_LLM)
    evaluate_retrieval(index, test_rows, rewrite=lambda q: rag.rewrite_query(client, model, q, []))
    evaluate_answers(index, client, model, test_rows, args.k)
    evaluate_conversations(index, client, model, args.k)


if __name__ == "__main__":
    main()
