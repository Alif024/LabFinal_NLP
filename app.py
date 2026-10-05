"""ผู้ช่วยยาสามัญประจำบ้าน: แชตบอต RAG ตอบคำถามจากคลังเอกสารใน data/"""
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pythainlp.util import normalize
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_LLM = "openai/gpt-oss-120b"
CHUNK_SIZE = 700       # ตัวอักษรต่อ chunk (โดยประมาณ)
CHUNK_OVERLAP = 120    # ตัวอักษรที่ซ้อนกับ chunk ก่อนหน้า
NOT_FOUND = "ไม่พบข้อมูลในเอกสาร"

SYSTEM_PROMPT = f"""คุณคือ "ผู้ช่วยยาสามัญประจำบ้าน" ตอบคำถามเรื่องยาสามัญประจำบ้านของไทยเป็นภาษาไทย

กฎที่ต้องทำตามอย่างเคร่งครัด:
1. ตอบโดยใช้ข้อมูลจาก <context> ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอกหรือเดาเพิ่มเติม
2. ทุกประโยคที่เป็นข้อเท็จจริงต้องอ้างอิงหมายเลขเอกสารในวงเล็บเหลี่ยม เช่น [1] หรือ [2][3]
3. ถ้า <context> ไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบเพียงว่า "{NOT_FOUND}" แล้วไม่ต้องอธิบายเพิ่ม
   ข้อยกเว้น: ถ้าเอกสารเขียนไว้ชัดว่า "ไม่มี" ยาหรือรายการที่ถูกถาม ถือว่าเอกสารมีคำตอบ ให้ตอบว่าไม่มีพร้อมอ้างอิง
   เช่น เอกสารเขียนว่า "ปัจจุบันยาในกลุ่มนี้ไม่มีรายการ..." ให้ตอบว่า "ปัจจุบันยังไม่มียาในกลุ่มนี้ที่เป็นยาสามัญประจำบ้าน [n]"
4. ถ้ามีข้อมูลเพียงบางส่วน ให้ตอบเฉพาะส่วนที่มี และบอกว่าส่วนใดไม่พบในเอกสาร
5. ตอบกระชับ อ่านง่าย ใช้หัวข้อย่อยได้เมื่อมีหลายรายการ
6. ถ้าคำถามเกี่ยวกับขนาดยาหรือความปลอดภัย ให้ปิดท้ายสั้น ๆ ว่าควรปรึกษาเภสัชกรหรือแพทย์"""

CONDENSE_PROMPT = """จากประวัติการสนทนาและคำถามล่าสุด ให้เขียนคำถามล่าสุดใหม่เป็นคำถามภาษาไทยที่สมบูรณ์ในตัวเอง
(แทนคำสรรพนามอย่าง "มัน" "ยานี้" ด้วยชื่อที่อ้างถึง) เพื่อใช้ค้นหาเอกสาร ตอบเฉพาะคำถามที่เขียนใหม่เท่านั้น

ประวัติการสนทนา:
{history}

คำถามล่าสุด: {question}
คำถามที่เขียนใหม่:"""

EXAMPLE_QUESTIONS = [
    "ท้องเสียควรใช้ยาสามัญประจำบ้านตัวไหน?",
    "พาราเซตามอลกินได้วันละไม่เกินเท่าไร?",
    "ยาหม่องใช้บรรเทาอาการอะไรได้บ้าง?",
    "ทำไมเด็กไม่ควรกินแอสไพริน?",
]


@dataclass
class Chunk:
    text: str
    title: str
    source: str
    file: str


# ---------- 1. Document Loading & Chunking ----------

def clean_text(text: str) -> str:
    text = normalize(text)                      # จัดการสระ/วรรณยุกต์ซ้ำซ้อนของภาษาไทย
    text = text.replace("​", "")           # zero-width space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents():
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        title = re.search(r"^# (.+)$", raw, flags=re.M).group(1).strip()
        source = re.search(r"^แหล่งที่มา: (\S+)", raw, flags=re.M).group(1)
        # ตัดส่วนหัว (ชื่อเรื่อง/แหล่งที่มา/สัญญาอนุญาต) ออก เหลือเฉพาะเนื้อหา
        body = raw.split("\n\n", 1)[1] if "\n\n" in raw else raw
        docs.append({"title": title, "source": source, "file": path.name, "body": clean_text(body)})
    return docs


def split_long(text: str, size: int, overlap: int):
    """ตัดข้อความยาวที่ขอบเขตช่องว่าง/บรรทัด (ภาษาไทยเว้นวรรคระหว่างประโยค) พร้อม overlap"""
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # เลื่อนจุดเริ่มไปที่ช่องว่างถัดไป จะได้ไม่ตัดกลางคำ
        space = text.find(" ", start, end)
        if space != -1:
            start = space + 1
    return [p for p in pieces if p]


def chunk_document(doc):
    """แบ่งตามโครงสร้างก่อน (หัวข้อ ## = ยา 1 ตำรับ / ย่อหน้า) แล้วรวมให้ได้ขนาดใกล้ CHUNK_SIZE"""
    sections = re.split(r"\n(?=## )", doc["body"])
    units = []
    for sec in sections:
        if sec.startswith("## "):
            units.append(sec)                      # ยา 1 ตำรับเก็บไว้ด้วยกันเสมอ
        else:
            units.extend(p for p in sec.split("\n\n") if p.strip())

    chunks, buf = [], ""
    for unit in units:
        if len(unit) > CHUNK_SIZE:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(split_long(unit, CHUNK_SIZE, CHUNK_OVERLAP))
        elif len(buf) + len(unit) + 2 <= CHUNK_SIZE:
            buf = f"{buf}\n\n{unit}" if buf else unit
        else:
            chunks.append(buf)
            buf = unit
    if buf:
        chunks.append(buf)

    # ใส่ชื่อเอกสารไว้หน้าทุก chunk เพื่อให้ embedding รู้บริบท (เช่น ชื่อกลุ่มยา)
    return [Chunk(f"{doc['title']}\n{c}", doc["title"], doc["source"], doc["file"]) for c in chunks]


# ---------- 2. Embedding & Vector Search ----------

@st.cache_resource(show_spinner="กำลังโหลดโมเดลและสร้างดัชนีค้นหา...")
def build_index():
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    docs = load_documents()
    chunks = [c for d in docs for c in chunk_document(d)]
    # โมเดล e5 ต้องขึ้นต้นด้วย "passage: " / "query: "
    vectors = model.encode([f"passage: {c.text}" for c in chunks], batch_size=32,
                           normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(vectors.shape[1])   # inner product ของเวกเตอร์ normalize = cosine
    index.add(np.asarray(vectors, dtype="float32"))
    return model, index, chunks, docs


def search(query: str, k: int):
    model, index, chunks, _ = build_index()
    q = model.encode([f"query: {query}"], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), k)
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


# ---------- 3–4. Prompt Engineering & LLM ----------

@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def llm_model():
    return st.secrets.get("GROQ_MODEL", DEFAULT_LLM)


def condense_question(question: str, history):
    """เขียนคำถามต่อเนื่องให้สมบูรณ์ในตัวเอง เช่น "แล้วเด็กกินได้ไหม" -> "เด็กกินพาราเซตามอลได้ไหม" """
    if not history:
        return question
    recent = "\n".join(f"{'ผู้ใช้' if m['role'] == 'user' else 'ผู้ช่วย'}: {m['content'][:500]}"
                       for m in history[-4:])
    resp = get_client().chat.completions.create(
        model=llm_model(), temperature=0, max_tokens=500, reasoning_effort="low",
        messages=[{"role": "user", "content": CONDENSE_PROMPT.format(history=recent, question=question)}],
    )
    return resp.choices[0].message.content.strip() or question


def build_context(results):
    return "\n\n".join(f'<doc id="{n}" title="{c.title}">\n{c.text}\n</doc>'
                       for n, (c, _) in enumerate(results, start=1))


def answer_stream(question: str, results, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content":
                     f"<context>\n{build_context(results)}\n</context>\n\nคำถาม: {question}"})
    stream = get_client().chat.completions.create(
        model=llm_model(), messages=messages, temperature=0.1, max_tokens=2048,
        reasoning_effort="low", stream=True)
    for part in stream:
        # บางโมเดลอ้างอิงด้วย 【1】 แปลงให้เป็นรูปแบบ [1] เหมือนกันทุกคำตอบ
        yield (part.choices[0].delta.content or "").replace("【", "[").replace("】", "]")


# ---------- 5. Chatbot Interface ----------

def render_sources(sources, used_query=None):
    with st.expander(f"📚 เอกสารอ้างอิง ({len(sources)} รายการ)"):
        if used_query:
            st.caption(f"คำค้นที่ใช้: {used_query}")
        for n, s in enumerate(sources, start=1):
            st.markdown(f"**[{n}] [{s['title']}]({s['source']})** · `{s['file']}` · ความใกล้เคียง {s['score']:.2f}")
            st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))


def main():
    st.set_page_config(page_title="ผู้ช่วยยาสามัญประจำบ้าน", page_icon="💊")
    st.title("💊 ผู้ช่วยยาสามัญประจำบ้าน")
    st.caption("ถาม-ตอบเรื่องยาสามัญประจำบ้านของไทย จากเอกสารอ้างอิง Wikipedia ภาษาไทย "
               "(ตามประกาศกระทรวงสาธารณสุข พ.ศ. 2568)")

    if "GROQ_API_KEY" not in st.secrets:
        st.error("ยังไม่ได้ตั้งค่า GROQ_API_KEY ใน Secrets ของ Streamlit")
        st.stop()

    _, _, chunks, docs = build_index()

    with st.sidebar:
        st.header("ตั้งค่า")
        top_k = st.slider("จำนวนเอกสารที่ค้นมาใช้ตอบ (top-k)", 2, 10, 5)
        if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.header("ลองถามดู")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True):
                st.session_state.pending = q
        st.header("คลังเอกสาร")
        st.caption(f"{len(docs)} เอกสาร · {len(chunks)} chunks")
        with st.expander("รายชื่อเอกสาร"):
            for d in docs:
                st.markdown(f"- [{d['title']}]({d['source']})")
        st.warning("ข้อมูลนี้ใช้เพื่อการศึกษา ไม่ใช่คำแนะนำทางการแพทย์ "
                   "ควรปรึกษาเภสัชกรหรือแพทย์ก่อนใช้ยา", icon="⚠️")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                render_sources(m["sources"], m.get("query"))

    question = st.chat_input("พิมพ์คำถามเกี่ยวกับยาสามัญประจำบ้าน...")
    question = question or st.session_state.pop("pending", None)
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            with st.spinner("กำลังค้นหาเอกสาร..."):
                query = condense_question(question, history)
                results = search(query, top_k)
            answer = st.write_stream(answer_stream(question, results, history))
        except Exception as e:  # แสดงข้อผิดพลาดของ API แทนการทำให้แอปล่ม
            st.error(f"เกิดข้อผิดพลาด: {e}")
            st.session_state.messages.pop()
            return
        # ตอบว่าไม่พบข้อมูล (และไม่ได้อ้างอิงเอกสารใดเลย) -> ไม่ต้องแสดงเอกสารอ้างอิง
        not_found = NOT_FOUND in answer and not re.search(r"\[\d+\]", answer)
        sources = [] if not_found else [
            {"title": c.title, "source": c.source, "file": c.file, "text": c.text, "score": s}
            for c, s in results]
        if sources:
            render_sources(sources, query if query != question else None)

    st.session_state.messages.append({"role": "assistant", "content": answer,
                                      "sources": sources, "query": query if query != question else None})


if __name__ == "__main__":
    main()
