"""ดึงบทความจาก Wikipedia ภาษาไทยมาสร้างคลังเอกสารใน data/

- บทความหลัก "ยาสามัญประจำบ้าน" -> แยกไฟล์ตามกลุ่มยา (แปลงตาราง wikitext เป็นข้อความ)
  ยา 1 ตำรับ = 1 หัวข้อ (## ชื่อตำรับยา) และมีหัวข้อสรุปรายชื่อยาของแต่ละกลุ่ม
- บทความตัวยา/ผลิตภัณฑ์ -> ไฟล์ละ 1 บทความ (plain text แบ่งตามหัวข้อของบทความ)

รัน: python scripts/build_data.py
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://th.wikipedia.org/w/api.php"
WIKI = "https://th.wikipedia.org/wiki/"
UA = {"User-Agent": "LabFinalNLP-RAG/0.1 (student project)"}
LICENSE = "สัญญาอนุญาต: CC BY-SA 4.0 (Wikipedia ภาษาไทย)"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MAIN_ARTICLE = "ยาสามัญประจำบ้าน"
DRUG_ARTICLES = ["พาราเซตามอล", "แอสไพริน", "คลอเฟนะมีน", "มะขามแขก", "ยาหม่อง", "ยาดม"]
TABLE_COLUMNS = 5  # ลำดับ | ชื่อตำรับยา | ตัวยาสำคัญและความแรง | สรรพคุณ | ขนาดบรรจุ
SKIP_SECTIONS = r"อ้างอิง|แหล่งข้อมูลอื่น|ดูเพิ่ม|เชิงอรรถ|บรรณานุกรม"


def api_get(params, retries=6):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            wait = 10 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed after {retries} retries: {url}")


def clean_wikitext(text):
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)  # [[a|b]] -> b
    text = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", text)  # [url label] -> label
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text).strip()               # ตัดช่องว่างหน้า/ท้ายบรรทัด
    text = re.sub(r"^[*#]{2}\s*", "    - ", text, flags=re.M)  # รายการย่อย (** หรือ ##)
    text = re.sub(r"^[*#]\s*", "  - ", text, flags=re.M)       # รายการ (* หรือ # แบบมีลำดับ)
    return text


def parse_table(table_text):
    """แปลง wikitable เป็น list ของแถว (แต่ละแถวคือ list ของ cell)"""
    rows, row, cell = [], None, None
    for line in table_text.splitlines():
        if line.startswith("{|") or line.startswith("!"):
            continue
        if line.startswith("|-") or line.startswith("|}"):
            if row:
                if cell is not None:
                    row.append(cell)
                rows.append(row)
            row, cell = [], None
            continue
        if line.startswith("|"):
            if cell is not None:
                row.append(cell)
            cell = line[1:]
        elif cell is not None:
            cell += "\n" + line
    rows = [[clean_wikitext(c) for c in r] for r in rows if r]
    for r in rows:
        if len(r) != TABLE_COLUMNS:
            raise ValueError(f"ตารางมี {len(r)} คอลัมน์ (ต้องเป็น {TABLE_COLUMNS}): {r[:2]}")
    return rows


def entry_text(cells, group_title):
    """ยา 1 ตำรับ: ใส่ชื่อกลุ่มทุกรายการ และวางสรรพคุณไว้ก่อนส่วนประกอบที่อาจยาวมาก"""
    num, name, ingredients, use, pack = cells

    def field(label, value):  # ค่าหลายบรรทัด/เป็นรายการ ให้ขึ้นบรรทัดใหม่
        return f"{label}:\n{value}" if "\n" in value or value.startswith(" ") else f"{label}: {value}"

    return "\n".join([f"## {name}",
                      f"กลุ่มยา: {group_title} (รายการที่ {num} ตามประกาศฯ)",
                      field("สรรพคุณ", use),
                      field("ขนาดบรรจุ", pack),
                      field("ตัวยาสำคัญและความแรง", ingredients)])


def sections_to_markdown(text):
    """แปลงหัวข้อ == ก == / === ข === ของ Wikipedia เป็น "## ก" / "## ก > ข" และตัดหัวข้อที่ไม่มีเนื้อหา"""
    sections, path, heading, lines = [], {}, None, []
    for line in text.split("\n"):
        m = re.match(r"^(=+)\s*(.+?)\s*=+\s*$", line)
        if not m:
            lines.append(line)
            continue
        sections.append((heading, lines))
        level = len(m.group(1))
        path = {lv: t for lv, t in path.items() if lv < level}
        path[level] = m.group(2)
        heading, lines = " > ".join(path[lv] for lv in sorted(path)), []
    sections.append((heading, lines))

    parts = []
    for heading, lines in sections:
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(l.strip() for l in lines)).strip()
        if body:  # เช่น "คลังภาพ" หรือหัวข้อแม่ที่มีแต่หัวข้อย่อย จะไม่มีเนื้อหา
            parts.append(f"## {heading}\n{body}" if heading else body)
    return "\n\n".join(parts)


def header(title, url, extra=""):
    h = f"แหล่งที่มา: {url}\n{LICENSE}\n"
    if extra:
        h += extra + "\n"
    return f"# {title}\n{h}\n"


def write(name, text):
    path = DATA_DIR / name
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"{name}: {len(text)} chars")


def build_main_article():
    data = api_get({"action": "parse", "page": MAIN_ARTICLE, "prop": "wikitext|revid"})["parse"]
    wikitext, revid = data["wikitext"], data["revid"]
    url = f"{WIKI}{MAIN_ARTICLE}?oldid={revid}"
    note = ("อ้างอิงตามประกาศกระทรวงสาธารณสุข เรื่อง ยาสามัญประจำบ้านแผนปัจจุบัน พ.ศ. 2568 "
            "(ราชกิจจานุเบกษา 14 สิงหาคม 2568)")

    intro = wikitext.split("\n==", 1)[0]
    groups = re.findall(r"^=== (กลุ่มที่ (\d+) .+?) ===\n(.*?)(?=^==)", wikitext, flags=re.S | re.M)
    group_names = "\n".join(f"- {title}" for title, _, _ in groups)
    write("00_ภาพรวมยาสามัญประจำบ้าน.txt",
          header("ภาพรวมยาสามัญประจำบ้าน", url, note)
          + clean_wikitext(intro)
          + f"\n\n## กลุ่มของยาสามัญประจำบ้านแผนปัจจุบัน\n"
            f"ยาสามัญประจำบ้านแผนปัจจุบันแบ่งออกเป็น {len(groups)} กลุ่ม ได้แก่\n{group_names}")

    for title, num, body in groups:
        parts = []
        table = re.search(r"\{\|.*?\n\|\}", body, flags=re.S)
        prose = clean_wikitext(body[:table.start()] if table else body)
        if prose:
            parts.append(prose)
        if table:
            rows = parse_table(table.group(0))
            names = "\n".join(f"- {cells[1]}" for cells in rows)
            parts.append(f"## รายชื่อยาใน{title}\n"
                         f"{title} มียาสามัญประจำบ้านแผนปัจจุบันทั้งหมด {len(rows)} รายการ ได้แก่\n{names}")
            parts.extend(entry_text(cells, title) for cells in rows)
        write(f"{int(num):02d}_{title.split(' ', 2)[2].split()[0]}.txt",
              header(f"ยาสามัญประจำบ้านแผนปัจจุบัน {title}", url, note) + "\n\n".join(parts))


def build_drug_article(index, title):
    page = api_get({"action": "query", "prop": "extracts|revisions", "rvprop": "ids",
                    "explaintext": 1, "redirects": 1, "titles": title})["query"]["pages"][0]
    if page.get("missing"):
        print(f"skip {title}: missing", file=sys.stderr)
        return
    text = re.split(rf"\n==+ (?:{SKIP_SECTIONS}) ==+", page["extract"])[0]
    url = f"{WIKI}{page['title']}?oldid={page['revisions'][0]['revid']}"
    write(f"{index}_{page['title']}.txt", header(page["title"], url) + sections_to_markdown(text))


def main():
    DATA_DIR.mkdir(exist_ok=True)
    build_main_article()
    for i, title in enumerate(DRUG_ARTICLES, start=20):
        time.sleep(3)
        build_drug_article(i, title)
    total = sum(len(p.read_text(encoding="utf-8")) for p in DATA_DIR.glob("*.txt"))
    print(f"\nรวม {len(list(DATA_DIR.glob('*.txt')))} ไฟล์, {total:,} ตัวอักษร")


if __name__ == "__main__":
    main()
