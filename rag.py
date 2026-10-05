"""แกนหลักของระบบ RAG: โหลดเอกสาร -> แบ่ง chunk -> ค้นหาแบบ hybrid -> สร้าง prompt -> เรียก LLM -> จัดการเลขอ้างอิง

แยกออกจาก app.py (หน้าเว็บ) เพื่อให้ scripts/evaluate.py ทดสอบ pipeline เดียวกับที่หน้าเว็บใช้ได้
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from groq import RateLimitError
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import syllable_tokenize, word_tokenize
from pythainlp.util import normalize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_LLM = "openai/gpt-oss-120b"
FALLBACK_LLMS = ["openai/gpt-oss-20b"]  # ใช้เมื่อโควตาของโมเดลหลักเต็ม (โควตาแยกกันรายโมเดล)
CHUNK_SIZE = 900       # ตัวอักษรสูงสุดของเนื้อหาต่อ chunk (รวมหัวข้อแล้วยังไม่เกิน 512 token ของ e5)
CHUNK_OVERLAP = 150    # ตัวอักษรที่ซ้อนกับชิ้นก่อนหน้า เมื่อย่อหน้าเดียวยาวเกิน CHUNK_SIZE
CANDIDATES = 20        # จำนวนผลจากแต่ละวิธีค้นหา ก่อนนำมารวมอันดับ
RRF_K = 60             # ค่าคงที่ของ Reciprocal Rank Fusion
NOT_FOUND = "ไม่พบข้อมูลในเอกสาร"

# [1]  [1][2]  [1, 2]  [1-3]
CITATION = re.compile(r"\[(\d+(?:\s*[-–,，、]\s*\d+)*)\]")

SYSTEM_PROMPT = f"""คุณคือ "ผู้ช่วยยาสามัญประจำบ้าน" ตอบคำถามเรื่องยาสามัญประจำบ้านของไทยเป็นภาษาไทย

กฎที่ต้องทำตามอย่างเคร่งครัด:
1. ตอบโดยใช้ข้อมูลจาก <context> ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอกหรือเดาเพิ่มเติม
2. ทุกประโยคที่เป็นข้อเท็จจริงต้องอ้างอิงหมายเลข id ของ <doc> ในวงเล็บเหลี่ยม เช่น [1] หรือ [2][3]
   อ้างอิงเฉพาะเอกสารที่มีข้อมูลนั้นจริง เอกสารที่ไม่เกี่ยวกับคำถามให้ข้ามไป ไม่ต้องกล่าวถึง
3. ถ้า <context> ไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบเพียงว่า "{NOT_FOUND}" แล้วไม่ต้องอธิบายเพิ่ม
   ข้อยกเว้น: ถ้าเอกสารเขียนไว้ชัดว่า "ไม่มี" ยาหรือรายการที่ถูกถาม ถือว่าเอกสารมีคำตอบ ให้ตอบว่าไม่มีพร้อมอ้างอิง
   เช่น เอกสารเขียนว่า "ปัจจุบันยาในกลุ่มนี้ไม่มีรายการ..." ให้ตอบว่า "ปัจจุบันยังไม่มียาในกลุ่มนี้ที่เป็นยาสามัญประจำบ้าน [n]"
4. ถ้ามีข้อมูลเพียงบางส่วน ให้ตอบเฉพาะส่วนที่มี และบอกว่าส่วนใดไม่พบในเอกสาร
5. ถ้าคำถามถามว่า "มีอะไรบ้าง" ให้ระบุทุกรายการใน <context> ที่เกี่ยวข้องให้ครบ รวมถึงรายการที่ตรงกับคำถามเพียงบางส่วน
   (เช่น ถามถึงโรค 2 โรค ให้รวมยาที่รักษาได้เพียงโรคใดโรคหนึ่งด้วย และบอกว่ารักษาโรคใด)
6. ตอบกระชับ อ่านง่าย ใช้หัวข้อย่อยได้เมื่อมีหลายรายการ
7. เฉพาะคำถามเรื่องขนาดยา วิธีใช้ หรือความปลอดภัยในการใช้ยา ให้ปิดท้ายสั้น ๆ ว่าควรปรึกษาเภสัชกรหรือแพทย์"""

CONDENSE_PROMPT = """หน้าที่ของคุณคือเขียนคำถามล่าสุดใหม่ให้เป็นคำถามที่สมบูรณ์ในตัวเอง เพื่อใช้ค้นหาเอกสาร ห้ามตอบคำถามเด็ดขาด
แทนคำสรรพนามหรือคำที่ละไว้ (เช่น "ยานี้" "แล้ว...ล่ะ") ด้วยสิ่งที่อ้างถึงในประวัติการสนทนา
ถ้าคำถามสมบูรณ์อยู่แล้วให้ใช้คำถามเดิม ห้ามใส่ตัวเลขหรือข้อมูลที่เป็นคำตอบลงในคำถาม

ประวัติการสนทนา:
{history}

คำถามล่าสุด: {question}

ตอบบรรทัดเดียวในรูปแบบ
คำถาม: ..."""


@dataclass
class Chunk:
    title: str     # ชื่อเอกสาร
    section: str   # หัวข้อในเอกสาร เช่น ชื่อตำรับยา (ว่างได้)
    body: str
    source: str    # URL ต้นฉบับ
    file: str

    @property
    def text(self) -> str:
        """ข้อความที่ใช้ทำ embedding/BM25 และส่งให้ LLM: ใส่ชื่อเอกสารและหัวข้อไว้หน้าเนื้อหาเพื่อให้รู้บริบท"""
        return "\n".join(p for p in (self.title, self.section, self.body) if p)


@dataclass
class Hit:
    chunk: Chunk
    rank: int          # อันดับหลังรวมผลค้นหา (1 = เกี่ยวข้องที่สุด)
    similarity: float  # cosine similarity ระหว่างคำถามกับ chunk


# ---------- 1. Document Loading, Cleaning & Chunking ----------

def clean_text(text: str) -> str:
    text = normalize(text)                      # จัดการสระ/วรรณยุกต์ซ้ำซ้อนของภาษาไทย
    text = text.replace("​", "")           # zero-width space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: Path = DATA_DIR):
    docs = []
    for path in sorted(data_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        title = re.search(r"^# (.+)$", raw, flags=re.M).group(1).strip()
        source = re.search(r"^แหล่งที่มา: (\S+)", raw, flags=re.M).group(1)
        # ตัดส่วนหัว (ชื่อเรื่อง/แหล่งที่มา/สัญญาอนุญาต) ออก เหลือเฉพาะเนื้อหา
        body = raw.split("\n\n", 1)[1] if "\n\n" in raw else raw
        docs.append({"title": title, "source": source, "file": path.name, "body": clean_text(body)})
    return docs


def split_sections(body: str):
    """แยกเนื้อหาตามหัวข้อ "## " -> [(หัวข้อ, เนื้อหา)] ส่วนก่อนหัวข้อแรกมีหัวข้อว่าง"""
    parts = re.split(r"^## ", body, flags=re.M)
    sections = [("", parts[0].strip())] if parts[0].strip() else []
    for part in parts[1:]:
        heading, _, content = part.partition("\n")
        sections.append((heading.strip(), content.strip()))
    return sections


def _cut_point(text: str, start: int, end: int) -> int:
    """จุดตัดที่ไม่อยู่กลางคำ: ขึ้นบรรทัดใหม่ > ช่องว่าง > ขอบเขตคำจาก pythainlp"""
    lower = start + (end - start) // 2
    for sep in ("\n", " "):
        cut = text.rfind(sep, lower, end)
        if cut != -1:
            return cut
    words = word_tokenize(text[start:end], engine="newmm")
    return end - len(words[-1]) if len(words) > 1 else end


def _overlap_start(text: str, cut: int, overlap: int) -> int:
    """จุดเริ่มชิ้นถัดไป: ย้อนจากจุดตัด overlap ตัวอักษร แล้วเลื่อนไปที่ขอบเขตคำถัดไป"""
    pos = max(cut - overlap, 0)
    space = re.compile(r"\s").search(text, pos, cut)
    if space:
        return space.end()
    words = word_tokenize(text[pos:cut], engine="newmm")
    return pos + len(words[0]) if len(words) > 1 else cut


def split_long(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """ตัดย่อหน้าที่ยาวเกิน size เป็นหลายชิ้นที่ซ้อนกัน overlap ตัวอักษร โดยไม่ตัดกลางคำ"""
    pieces, start = [], 0
    while len(text) - start > size:
        cut = _cut_point(text, start, start + size)
        pieces.append(text[start:cut].strip())
        start = _overlap_start(text, cut, overlap)
    pieces.append(text[start:].strip())
    return [p for p in pieces if p]


def chunk_document(doc):
    """แบ่งตามหัวข้อก่อน (ยา 1 ตำรับ หรือ 1 หัวข้อของบทความ ไม่รวมข้ามหัวข้อ)
    แล้วรวมย่อหน้าในหัวข้อเดียวกันให้ได้ไม่เกิน CHUNK_SIZE ทุกชิ้นจึงมีชื่อเอกสารและหัวข้อกำกับเสมอ"""
    chunks = []
    for section, content in split_sections(doc["body"]):
        pieces, buf = [], ""
        for para in (p.strip() for p in content.split("\n\n")):
            if not para:
                continue
            if len(para) > CHUNK_SIZE:
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.extend(split_long(para))
            elif buf and len(buf) + 2 + len(para) <= CHUNK_SIZE:
                buf = f"{buf}\n\n{para}"
            else:
                if buf:
                    pieces.append(buf)
                buf = para
        if buf:
            pieces.append(buf)
        chunks += [Chunk(doc["title"], section, p, doc["source"], doc["file"]) for p in pieces]
    return chunks


# ---------- 2. Embedding & Vector Search (+ BM25) ----------

STOPWORDS = frozenset(thai_stopwords())


def tokenize(text: str):
    """ตัดคำสำหรับ BM25 ด้วย pythainlp ทั้งระดับคำ (newmm) และระดับพยางค์ โดยตัดช่องว่าง เครื่องหมาย และ stopword ออก
    พยางค์ช่วยจับคู่คำประสมที่ตัดคำต่างกัน เช่น "ยาบำรุง|เลือดจาง" กับ "ยาเม็ด|บำรุง|โลหิต|จาง" (บำ, รุง, จาง)"""
    keep = lambda t: t not in STOPWORDS and re.search(r"\w", t)
    words = [w.lower() for w in word_tokenize(text, engine="newmm", keep_whitespace=False) if keep(w)]
    syllables = [f"§{s.lower()}" for s in syllable_tokenize(text, engine="dict", keep_whitespace=False) if keep(s)]
    return words + syllables


class RagIndex:
    """ดัชนีค้นหา 2 แบบ: ความหมาย (multilingual-e5 + FAISS) และคำสำคัญ (BM25) แล้วรวมอันดับด้วย RRF"""

    def __init__(self, data_dir: Path = DATA_DIR, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.docs = load_documents(data_dir)
        self.chunks = [c for d in self.docs for c in chunk_document(d)]
        # โมเดล e5 ต้องขึ้นต้นด้วย "passage: " / "query: " และ normalize เพื่อให้ inner product = cosine
        self.vectors = self.model.encode([f"passage: {c.text}" for c in self.chunks], batch_size=32,
                                         normalize_embeddings=True, show_progress_bar=False).astype("float32")
        self.faiss = faiss.IndexFlatIP(self.vectors.shape[1])
        self.faiss.add(self.vectors)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def search(self, queries, k: int, use_vector: bool = True, use_bm25: bool = True):
        """ค้นหาด้วยคำค้นหนึ่งหรือหลายคำค้น (เช่น คำถาม + คำค้นเพิ่มเติม) แล้วรวมทุกอันดับด้วย RRF"""
        queries = [clean_text(q) for q in ([queries] if isinstance(queries, str) else queries) if q.strip()]
        q = self.model.encode([f"query: {x}" for x in queries], normalize_embeddings=True).astype("float32")
        n = min(CANDIDATES, len(self.chunks))
        rankings = []
        if use_vector:
            _, ids = self.faiss.search(q, n)
            rankings += [[int(i) for i in row if i != -1] for row in ids]
        if use_bm25:
            for x in queries:
                scores = self.bm25.get_scores(tokenize(x))
                rankings.append([int(i) for i in np.argsort(-scores)[:n] if scores[i] > 0])
        # Reciprocal Rank Fusion: คะแนน = ผลรวมของ 1/(RRF_K + อันดับ) จากทุกวิธีและทุกคำค้น
        fused = defaultdict(float)
        for ranking in rankings:
            for rank, i in enumerate(ranking, start=1):
                fused[i] += 1 / (RRF_K + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        # ความใกล้เคียงที่แสดงผล = cosine similarity กับคำค้นแรก (คำถาม)
        return [Hit(self.chunks[i], rank, float(self.vectors[i] @ q[0])) for rank, i in enumerate(top, start=1)]


# ---------- 3–4. Prompt Engineering & LLM ----------

def strip_citations(text: str) -> str:
    """ลบเลขอ้างอิงออกจากคำตอบเก่าในประวัติแชต เพราะเลข [n] ผูกกับเอกสารของคำถามนั้น ๆ เท่านั้น"""
    return re.sub(r"[ \t]*" + CITATION.pattern, "", text)


def chat(client, model: str, **kwargs):
    """เรียก LLM ถ้าโควตาของโมเดลหลักเต็ม (HTTP 429) จะลองโมเดลสำรองตามลำดับ"""
    models = [model] + [m for m in FALLBACK_LLMS if m != model]
    for i, m in enumerate(models):
        try:
            return client.chat.completions.create(model=m, **kwargs)
        except RateLimitError:
            if i == len(models) - 1:
                raise


def condense_question(client, model: str, question: str, history) -> str:
    """ทำคำถามต่อเนื่องให้สมบูรณ์ในตัวเองก่อนค้นหา เช่น "แล้วเด็กกินได้ไหม" -> "เด็กกินพาราเซตามอลได้ไหม"
    คำถามแรกของบทสนทนาสมบูรณ์อยู่แล้ว จึงใช้ตามที่ผู้ใช้พิมพ์โดยไม่เรียก LLM"""
    if not history:
        return question
    recent = "\n".join(f"{'ผู้ใช้' if m['role'] == 'user' else 'ผู้ช่วย'}: {strip_citations(m['content'])[:500]}"
                       for m in history[-4:])
    resp = chat(client, model, temperature=0, max_tokens=800, reasoning_effort="low",
                messages=[{"role": "user", "content": CONDENSE_PROMPT.format(history=recent, question=question)}])
    out = resp.choices[0].message.content or ""
    rewritten = re.search(r"คำถาม\W*:\s*(.+)", out)
    return rewritten.group(1).strip() if rewritten else (out.strip().splitlines() or [question])[0]


def build_context(hits) -> str:
    return "\n\n".join(f'<doc id="{n}">\n{h.chunk.text}\n</doc>' for n, h in enumerate(hits, start=1))


def answer_stream(client, model: str, question: str, hits, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": strip_citations(m["content"])} for m in history[-6:]]
    messages.append({"role": "user", "content":
                     f"<context>\n{build_context(hits)}\n</context>\n\nคำถาม: {question}"})
    stream = chat(client, model, messages=messages, temperature=0.1, max_tokens=2048,
                  reasoning_effort="low", stream=True)
    for part in stream:
        if part.choices:
            yield part.choices[0].delta.content or ""


def _citation_numbers(inner: str):
    """"1, 3" -> [1, 3] และ "2-4" -> [2, 3, 4]"""
    nums = []
    for part in re.split(r"[,，、]", inner):
        lo, _, hi = re.sub(r"\s", "", part).replace("–", "-").partition("-")
        hi = hi if hi and 0 <= int(hi) - int(lo) <= 20 else lo
        nums += range(int(lo), int(hi) + 1)
    return nums


def finalize_answer(text: str, n_docs: int):
    """จัดเลขอ้างอิงใหม่ตามลำดับที่ปรากฏในคำตอบ (เอกสารที่ถูกอ้างถึงก่อนได้ [1]) และตัดเลขที่ไม่มีเอกสารจริง
    คืนค่า (คำตอบ, ลำดับ id เดิมของเอกสารที่ถูกอ้างอิง)"""
    # บางโมเดลอ้างอิงแบบ 【1】 หรือ 【1†source】 ให้เป็นรูปแบบ [1] เหมือนกันทุกคำตอบ
    text = re.sub(r"【(\d+)[^】]*】", r"[\1]", text)
    cited = []
    for m in CITATION.finditer(text):
        for n in _citation_numbers(m.group(1)):
            if 1 <= n <= n_docs and n not in cited:
                cited.append(n)
    new_id = {old: new for new, old in enumerate(cited, start=1)}

    def renumber(m):
        nums = dict.fromkeys(new_id[n] for n in _citation_numbers(m.group(1)) if n in new_id)
        return "".join(f"[{n}]" for n in nums)

    return CITATION.sub(renumber, text).strip(), cited


def is_not_found(answer: str, cited) -> bool:
    return not cited and NOT_FOUND in answer
