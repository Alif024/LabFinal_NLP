# Todo: แบบทดสอบเก็บคะแนน ครั้งที่ 2 (10 คะแนน)

สร้าง Web Application แชตบอต RAG ที่ตอบคำถามจากคลังเอกสาร (ไทย/อังกฤษ) แล้ว Deploy บน Streamlit Community Cloud

## 1. เลือกหัวข้อ: ผู้ช่วยยาสามัญประจำบ้าน
- [x] คิดหัวข้อ/Domain ที่สนใจ (เช่น ผู้ช่วยกฎหมาย PDPA, ไกด์เที่ยวจังหวัด, คู่มือสินค้า)

## 2. เตรียมข้อมูล
- [x] สร้างโฟลเดอร์ `data/` เก็บเอกสารความรู้ อย่างน้อย 10 ไฟล์ หรือรวมไม่น้อยกว่า 15,000 ตัวอักษร
- [x] สร้าง `test_questions.csv` คำถามทดสอบ ≥ 10 ข้อ พร้อมคำตอบที่ถูกต้อง
  - [x] มีคำถามที่ **ไม่มีคำตอบในเอกสาร** อย่างน้อย 2 ข้อ

## 3. พัฒนาระบบ RAG (`app.py`)
- [x] **Document Loading & Chunking**: โหลดเอกสาร ทำความสะอาดข้อความ แล้วแบ่งเป็น chunk
- [x] **Embedding & Vector Search**: แปลง chunk เป็นเวกเตอร์ด้วย Sentence Embedding และค้นหาด้วย FAISS
- [x] **Prompt Engineering**:
  - [x] ให้ LLM ตอบจาก context ที่ค้นได้เท่านั้น
  - [x] อ้างอิงแหล่งที่มา
  - [x] ตอบว่า "ไม่พบข้อมูล" เมื่อเอกสารไม่มีคำตอบ
- [x] **LLM**: เรียกใช้ผ่าน API (เช่น Groq) ด้วย `st.secrets["GROQ_API_KEY"]`
- [x] **Chatbot Interface**: หน้าแชตที่คุยต่อเนื่องได้ และแสดงเอกสารอ้างอิงที่ใช้ตอบทุกครั้ง
- [x] ใช้ Embedding Model ขนาดเล็ก และโหลดโมเดลกับ index ครั้งเดียว (`@st.cache_resource`)

## 4. ไฟล์ประกอบใน Repository
- [x] `requirements.txt` (streamlit, sentence-transformers, faiss-cpu, groq, pythainlp)
- [x] `README.md` อธิบาย:
  - [x] วิธีใช้งาน
  - [x] แนวคิดของ Domain
  - [x] แหล่งที่มาของเอกสาร
  - [x] ตัวอย่าง Prompt ที่ใช้สั่ง AI
- [x] `.gitignore` ใส่ `.streamlit/secrets.toml`

## 5. GitHub & Deploy
- [x] ตรวจให้แน่ใจว่า **ไม่มี API Key ใน Repository** (ถ้ามีจะถูกหักคะแนน)
- [x] Push โค้ดขึ้น GitHub Repository ส่วนตัว (https://github.com/Alif024/LabFinal_NLP)
- [x] Deploy บน Streamlit Community Cloud (repo `Alif024/LabFinal_NLP`, branch `main`, ไฟล์ `app.py`, Python 3.12): https://labfinalnlp-czysquxhqbgcpbsgvfewbr.streamlit.app/
- [x] ใส่ `GROQ_API_KEY` และ `GROQ_MODEL` ใน Secrets ของ Streamlit Cloud
- [x] ทดสอบเปิด URL จากเครื่องอื่นหรือโหมด Incognito ว่าใช้งานได้จริง (5 ต.ค. 2569: headless Chromium แบบไม่มีคุกกี้ ตอบได้ ถามต่อเนื่องได้ ตอบ "ไม่พบข้อมูล" ได้)
- [x] ใส่ URL ของ Streamlit ใน `README.md`
- [x] ทดสอบด้วยคำถามใน `test_questions.csv` (รวมถึงคำถามที่ต้องตอบว่า "ไม่พบข้อมูล")

## 6. สิ่งที่ต้องส่ง
- [x] กรอก `NLP-SubTest2.ipynb`:
  - [x] รหัสนักศึกษา
  - [x] ชื่อ-สกุล
  - [x] หัวข้อที่ใช้
  - [x] URL หน้าเว็บ Streamlit
  - [x] GitHub Repository URL
- [ ] ไฟล์ PDF รวมภาพหน้าจอการทำงานของเว็บ พร้อมคำอธิบาย (มีภาพตั้งต้นใน `screenshots/`)

## เกณฑ์การให้คะแนน
| เกณฑ์ | คะแนน |
|---|---|
| ไฟล์เอกสารความรู้และไฟล์คำถามทดสอบ | 1 |
| โค้ดขึ้น GitHub และเว็บ Streamlit ใช้งานได้จริง | 2 |
| ความคิดสร้างสรรค์ของหัวข้อ และความเหมาะสมของเอกสาร | 2 |
| ใช้เทคนิค RAG ถูกต้อง (Chunking, Retrieval, Prompt) | 3 |
| ตอบอิงเอกสาร แสดงแหล่งอ้างอิง ปฏิเสธได้เมื่อไม่มีข้อมูล | 1 |
| หน้าเว็บใช้งานง่าย | 1 |
