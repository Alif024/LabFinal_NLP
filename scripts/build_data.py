"""ดึงบทความจาก Wikipedia ภาษาไทยมาสร้างคลังเอกสารใน data/

- บทความหลัก "ยาสามัญประจำบ้าน" -> แยกไฟล์ตามกลุ่มยา (แปลงตาราง wikitext เป็นข้อความ)
- บทความตัวยา/ผลิตภัณฑ์ -> ไฟล์ละ 1 บทความ (plain text)

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
TABLE_FIELDS = ["ลำดับ", "ชื่อตำรับยา", "ตัวยาสำคัญและความแรง", "สรรพคุณ", "ขนาดบรรจุ"]


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
    text = re.sub(r"^\*\*\s*", "    - ", text, flags=re.M)
    text = re.sub(r"^\*\s*", "  - ", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


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
    return [[clean_wikitext(c) for c in r] for r in rows if r]


def row_to_text(cells):
    lines = []
    for name, value in zip(TABLE_FIELDS, cells):
        if "\n" in value:
            lines.append(f"{name}:\n{value}")
        else:
            lines.append(f"{name}: {value}")
    return "\n".join(lines)


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
          + "\n\nยาสามัญประจำบ้านแผนปัจจุบันแบ่งออกเป็น 16 กลุ่ม ได้แก่\n" + group_names)

    for title, num, body in groups:
        parts = []
        table = re.search(r"\{\|.*?\n\|\}", body, flags=re.S)
        prose = clean_wikitext(body[:table.start()] if table else body)
        if prose:
            parts.append(prose)
        if table:
            for cells in parse_table(table.group(0)):
                parts.append(f"## {cells[1]}\n" + row_to_text(cells))
        write(f"{int(num):02d}_{title.split(' ', 2)[2].split()[0]}.txt",
              header(f"ยาสามัญประจำบ้านแผนปัจจุบัน {title}", url, note) + "\n\n".join(parts))


def build_drug_article(index, title):
    page = api_get({"action": "query", "prop": "extracts|revisions", "rvprop": "ids",
                    "explaintext": 1, "redirects": 1, "titles": title})["query"]["pages"][0]
    if page.get("missing"):
        print(f"skip {title}: missing", file=sys.stderr)
        return
    text = page["extract"]
    text = re.split(r"\n== (?:อ้างอิง|แหล่งข้อมูลอื่น|ดูเพิ่ม) ==", text)[0]
    text = re.sub(r"\n{3,}", "\n\n", text)
    url = f"{WIKI}{page['title']}?oldid={page['revisions'][0]['revid']}"
    write(f"{index}_{page['title']}.txt", header(page["title"], url) + text)


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
