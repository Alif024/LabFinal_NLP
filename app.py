"""ผู้ช่วยยาสามัญประจำบ้าน: หน้าเว็บแชตบอต RAG (ส่วน pipeline อยู่ใน rag.py)"""
import streamlit as st
from groq import Groq

import rag

EXAMPLE_QUESTIONS = [
    "ท้องเสียควรใช้ยาสามัญประจำบ้านตัวไหน?",
    "พาราเซตามอลกินได้วันละไม่เกินเท่าไร?",
    "ยาบำรุงเลือดสำหรับคนที่โลหิตจางมีอะไรบ้าง?",
    "ทำไมเด็กไม่ควรกินแอสไพริน?",
]


@st.cache_resource(show_spinner="กำลังโหลดโมเดลและสร้างดัชนีค้นหา...")
def get_index():
    return rag.RagIndex()  # โหลดโมเดลและสร้างดัชนีครั้งเดียว ใช้ร่วมกันทุกผู้ใช้


@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def has_api_key() -> bool:
    try:
        return "GROQ_API_KEY" in st.secrets
    except Exception:  # ไม่มีไฟล์ secrets เลย
        return False


def source_info(hit):
    c = hit.chunk
    return {"title": c.title, "section": c.section, "body": c.body, "source": c.source,
            "file": c.file, "rank": hit.rank, "similarity": hit.similarity}


def render_sources(m):
    label = "📚 เอกสารอ้างอิงที่ใช้ตอบ" if m["cited"] else "📚 เอกสารที่ค้นพบ (คำตอบไม่ได้ระบุเลขอ้างอิง)"
    with st.expander(f"{label} ({len(m['sources'])} รายการ)"):
        q = m["query"]
        lines = [f"คำถามที่ใช้ค้นหา: {q.question}"] if m["rewritten"] else []
        lines += [f"คำค้นเพิ่มเติม: {q.keywords}"] if q.keywords else []
        if lines:
            st.caption("  \n".join(lines))
        for n, s in enumerate(m["sources"], start=1):
            st.markdown(f"**[{n}] [{s['title']}]({s['source']})**  \n"
                        f"`{s['file']}` · ผลค้นหาอันดับที่ {s['rank']} · ความใกล้เคียง {s['similarity']:.2f}")
            if s["section"]:
                st.markdown(f"**{s['section']}**")
            st.text(s["body"])
        if m.get("others"):
            st.caption("ผลค้นหาอื่นที่ไม่ได้ใช้ตอบ: " + " · ".join(
                f"{o['section'] or o['title']} (อันดับที่ {o['rank']})" for o in m["others"]))


def main():
    st.set_page_config(page_title="ผู้ช่วยยาสามัญประจำบ้าน", page_icon="💊")
    st.title("💊 ผู้ช่วยยาสามัญประจำบ้าน")
    st.caption("ถาม-ตอบเรื่องยาสามัญประจำบ้านของไทย จากเอกสารอ้างอิง Wikipedia ภาษาไทย "
               "(ตามประกาศกระทรวงสาธารณสุข พ.ศ. 2568)")

    if not has_api_key():
        st.error("ยังไม่ได้ตั้งค่า GROQ_API_KEY ใน Secrets ของ Streamlit")
        st.stop()

    index = get_index()
    llm = st.secrets.get("GROQ_MODEL", rag.DEFAULT_LLM)

    with st.sidebar:
        st.header("ตั้งค่า")
        top_k = st.slider("จำนวนผลค้นหาที่ส่งให้ LLM (top-k)", 3, 10, 5,
                          help="ยิ่งมากยิ่งลดโอกาสพลาดข้อมูล แต่คำตอบจะช้าลงเล็กน้อย "
                               "เอกสารอ้างอิงจะแสดงเฉพาะชิ้นที่คำตอบอ้างถึงจริง")
        if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.header("ลองถามดู")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True):
                st.session_state.pending = q
        st.header("คลังเอกสาร")
        st.caption(f"{len(index.docs)} เอกสาร · {len(index.chunks)} chunks · "
                   "ค้นหาแบบ hybrid (embedding + FAISS และ BM25)")
        with st.expander("รายชื่อเอกสาร"):
            for d in index.docs:
                st.markdown(f"- [{d['title']}]({d['source']})")
        st.warning("ข้อมูลนี้ใช้เพื่อการศึกษา ไม่ใช่คำแนะนำทางการแพทย์ "
                   "ควรปรึกษาเภสัชกรหรือแพทย์ก่อนใช้ยา", icon="⚠️")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                render_sources(m)

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
                search_query = rag.rewrite_query(get_client(), llm, question, history)
                hits = index.search(search_query.queries, top_k)
            placeholder, text = st.empty(), ""
            for token in rag.answer_stream(get_client(), llm, question, hits, history):
                text += token
                # เลขอ้างอิงเรียงตามลำดับที่ปรากฏ จึงจัดเลขใหม่ระหว่าง stream ได้โดยเลขเดิมไม่เปลี่ยน
                placeholder.markdown(rag.finalize_answer(text, len(hits))[0] + "▌")
        except Exception as e:  # แสดงข้อผิดพลาดของ API แทนการทำให้แอปล่ม
            st.error(f"เกิดข้อผิดพลาด: {e}")
            st.session_state.messages.pop()
            return

        # เลขอ้างอิงเรียงตามลำดับที่คำตอบอ้างถึง และแสดงเฉพาะเอกสารที่ถูกอ้างอิงจริง
        answer, cited = rag.finalize_answer(text, len(hits))
        if not answer:
            placeholder.error("ไม่ได้รับคำตอบจาก LLM กรุณาลองถามใหม่อีกครั้ง")
            st.session_state.messages.pop()
            return
        placeholder.markdown(answer)
        if cited:
            sources = [source_info(hits[i - 1]) for i in cited]
            others = [source_info(h) for n, h in enumerate(hits, start=1) if n not in cited]
        elif rag.is_not_found(answer, cited):
            sources, others = [], []          # ตอบว่าไม่พบข้อมูล -> ไม่ต้องแสดงเอกสารอ้างอิง
        else:
            sources, others = [source_info(h) for h in hits], []
        message = {"role": "assistant", "content": answer, "sources": sources, "others": others,
                   "cited": bool(cited), "query": search_query,
                   "rewritten": search_query.question != question}
        if sources:
            render_sources(message)

    st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
