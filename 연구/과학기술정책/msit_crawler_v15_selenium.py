# ============================================================
# 과학기술정보통신부 크롤러 v15
# - v14 기능 전체 유지
# - 첨부파일 다운로드 로직 변경:
#   1차: AI 키워드 검색 (제목)
#   2차: 본문 읽고 의료 키워드 필터링
#   3차: 통과한 경우에만 첨부파일 다운로드
# ============================================================
from __future__ import annotations
import os, re, time
from datetime import datetime
from typing import List, Dict
from urllib.parse import quote_plus
from collections import Counter

import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ============================================================
EDGE_DRIVER_PATH  = r"C:\Users\korea\OneDrive\Desktop\msedgedriver.exe" #본인 워킹 스페이스에 맞춰 수정
SAVE_DIR          = r"C:\Users\korea\Desktop\msit_data" #본인 워킹 스페이스에 맞춰 수정
ATTACH_DIR        = os.path.join(SAVE_DIR, "attachments")
DOWNLOAD_ATTACH   = True
MAX_PAGES         = 30
SLEEP_LIST        = 4.0
SLEEP_DETAIL      = 3.5
SLEEP_DOWNLOAD    = 3.0
COLLECT_DETAIL    = True
WAIT_TIMEOUT      = 20
MAX_RETRY         = 3
RETRY_SLEEP       = 5.0

BASE = "https://www.msit.go.kr"

BOARDS = [
    {"name": "보도자료",         "mPid": "208", "mId": "307", "bbsSeqNo": "94"},
    {"name": "공지사항",         "mPid": "121", "mId": "310", "bbsSeqNo": "96"},
    {"name": "연구개발,미래인재", "mPid": "243", "mId": "244", "bbsSeqNo": "65"},
    {"name": "정보통신",         "mPid": "243", "mId": "245", "bbsSeqNo": "67"},
    {"name": "네트워크,전파",     "mPid": "243", "mId": "328", "bbsSeqNo": "127"},
    {"name": "과학기술혁신",      "mPid": "243", "mId": "246", "bbsSeqNo": "122"},
    {"name": "기타",             "mPid": "243", "mId": "98",  "bbsSeqNo": "78"},
    {"name": "간행물",           "mPid": "100", "mId": "102", "bbsSeqNo": "81"},
    {"name": "통계정보",         "mPid": "74",  "mId": "99",  "bbsSeqNo": "79"},
]

AI_KEYWORDS = [
    "인공지능", "머신러닝", "딥러닝", "컴퓨터비전",
    "거대언어모델", "자연어처리", "AI", "신경망", "LLM",
]

MEDICAL_KEYWORDS = [
    "의료", "헬스케어", "건강", "병원", "진단",
    "치료", "임상", "환자", "바이오", "디지털헬스", "의료기기",
    "의약", "의학", "보건", "재활", "원격의료", "헬스",
]

# ============================================================
def init_driver(download_dir: str = None) -> webdriver.Edge:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-extensions")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    if download_dir:
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        options.add_experimental_option("prefs", prefs)
    service = Service(executable_path=EDGE_DRIVER_PATH)
    return webdriver.Edge(service=service, options=options)

def make_search_url(board: dict, kw: str, page: int) -> str:
    return (
        f"{BASE}/bbs/list.do?sCode=user"
        f"&mPid={board['mPid']}&mId={board['mId']}&bbsSeqNo={board['bbsSeqNo']}"
        f"&pageIndex={page}&searchOpt=ALL&searchTxt={quote_plus(kw)}"
    )

def safe_get(driver, url: str, sleep: float = SLEEP_LIST) -> bool:
    for attempt in range(1, MAX_RETRY + 1):
        try:
            driver.get(url)
            WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(sleep)
            return True
        except Exception as e:
            if attempt < MAX_RETRY:
                print(f"    [로딩 실패 {attempt}/{MAX_RETRY}회] {url[:50]}... {RETRY_SLEEP}초 후 재시도")
                time.sleep(RETRY_SLEEP)
            else:
                print(f"    [로딩 최종 실패] {url[:60]}: {e}")
                return False
    return False

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()

def download_file_via_selenium(driver, atch_file_no: str, file_ord: str,
                                file_extsn: str, title: str) -> str:
    safe_title = sanitize_filename(title[:40])
    fname      = f"{safe_title}__{atch_file_no}_{file_ord}.{file_extsn}"
    save_path  = os.path.join(ATTACH_DIR, fname)

    if os.path.exists(save_path):
        print(f"      [스킵] {fname}")
        return fname

    try:
        js = f"""
            document.getElementById('atchFileNo').value = '{atch_file_no}';
            document.getElementById('fileOrd').value    = '{file_ord}';
            document.getElementById('fileBtn').value    = 'A';
            document.getElementById('fileForm').action  = '/ssm/file/fileDown.do';
            document.getElementById('fileForm').submit();
        """
        driver.execute_script(js)
        time.sleep(SLEEP_DOWNLOAD)

        files = sorted(
            [os.path.join(ATTACH_DIR, f) for f in os.listdir(ATTACH_DIR)
             if not f.endswith(".crdownload")],
            key=os.path.getmtime, reverse=True
        )
        if files:
            latest = files[0]
            target = os.path.join(ATTACH_DIR, fname)
            if latest != target:
                os.rename(latest, target)
            print(f"      📥 {fname}")
            return fname
        else:
            print(f"      [다운로드 파일 없음] {atch_file_no}_{file_ord}.{file_extsn}")
            return ""
    except Exception as e:
        print(f"      [다운로드 실패] {atch_file_no}/{file_ord}: {e}")
        return ""

def extract_date_from_text(text: str) -> str:
    m = re.search(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", text)
    if m:
        y, mon, d = m.groups()
        return f"{y}-{mon.zfill(2)}-{d.zfill(2)}"
    return ""

def parse_list_page(html: str, board_name: str, kw: str) -> List[Dict]:
    soup  = BeautifulSoup(html, "html.parser")
    items = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "view.do" not in href and "nttSeqNo" not in href:
            continue
        title = a.get_text(strip=True)
        if not title or len(title) < 2:
            continue
        url = BASE + href if href.startswith("/") else href

        date_text = extract_date_from_text(title)
        if not date_text:
            tr = a.find_parent("tr")
            if tr:
                for td in tr.find_all("td"):
                    txt = td.get_text(strip=True)
                    date_text = extract_date_from_text(txt)
                    if date_text:
                        break

        items.append({
            "게시판":     board_name,
            "검색키워드": kw,
            "제목":       title,
            "URL":        url,
            "날짜":       date_text,
            "연도":       date_text[:4] if len(date_text) >= 4 else "",
            "본문":       "",
            "첨부파일":   "",
            "첨부파일경로": "",
            "매칭의료어": "",
            "매칭AI어":   "",
        })
    return items

def extract_body(html: str) -> str:
    """본문만 추출 (첨부파일 다운로드 없이)"""
    soup = BeautifulSoup(html, "html.parser")
    body = ""
    for sel in [".view_cont", ".bbs_view", ".cont_area", "#content .view", ".board_view"]:
        tag = soup.select_one(sel)
        if tag:
            body = tag.get_text(separator=" ", strip=True)
            if len(body) > 50:
                break
    if not body:
        body = soup.get_text(separator=" ", strip=True)[:3000]
    return body

def download_attachments(driver, html: str, title: str) -> tuple:
    """
    의료 키워드 통과 후 호출 — 첨부파일 다운로드만 수행.
    반환: (첨부파일명, 첨부파일경로)
    """
    soup = BeautifulSoup(html, "html.parser")
    attach_names, attach_paths = [], []

    if DOWNLOAD_ATTACH:
        pattern = re.compile(r"fn_download\(\'(\d+)\',\s*\'(\d+)\',\s*\'(\w+)\'\)")
        for a in soup.find_all("a", onclick=pattern):
            m = pattern.search(a.get("onclick", ""))
            if not m:
                continue
            atch_no, file_ord, extsn = m.group(1), m.group(2), m.group(3)
            if extsn.lower() == "odt":
                continue
            fname = download_file_via_selenium(driver, atch_no, file_ord, extsn, title)
            if fname:
                attach_names.append(fname)
                attach_paths.append(os.path.join(ATTACH_DIR, fname))
    else:
        # 이름만 수집
        pattern = re.compile(r"fn_download\(\'(\d+)\',\s*\'(\d+)\',\s*\'(\w+)\'\)")
        for a in soup.find_all("a", onclick=pattern):
            m = pattern.search(a.get("onclick", ""))
            if m:
                attach_names.append(f"{m.group(1)}_{m.group(2)}.{m.group(3)}")

    return " | ".join(attach_names), " | ".join(attach_paths)

def get_matched_keywords(text: str, keyword_list: list) -> str:
    return ", ".join([kw for kw in keyword_list if kw in text])

def has_medical_keyword(text: str) -> bool:
    return any(kw in text for kw in MEDICAL_KEYWORDS)

def has_next_page(html: str, current_page: int) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select("div.paging a, div.pagination a, .pager a"):
        txt = a.get_text(strip=True)
        if txt.isdigit() and int(txt) > current_page:
            return True
    if soup.select_one("a.next, a[title=\'다음\'], a[title=\'다음 페이지\']"):
        return True
    return False

def collect_board(driver, board: dict, kw: str) -> List[Dict]:
    results, seen = [], set()
    for page in range(1, MAX_PAGES + 1):
        url = make_search_url(board, kw, page)
        if not safe_get(driver, url, SLEEP_LIST):
            break
        html  = driver.page_source
        items = parse_list_page(html, board["name"], kw)
        if not items:
            print(f"    → p{page}: 게시글 없음, 중단")
            break
        new_count = 0
        for item in items:
            if item["URL"] in seen:
                continue
            seen.add(item["URL"])
            if COLLECT_DETAIL:
                if safe_get(driver, item["URL"], SLEEP_DETAIL):
                    page_html = driver.page_source

                    # 1단계: 본문만 먼저 추출
                    body = extract_body(page_html)
                    item["본문"] = body

                    # 2단계: 의료 키워드 필터링
                    combined = item["제목"] + " " + body
                    if has_medical_keyword(combined):
                        # 3단계: 통과한 경우에만 첨부파일 다운로드
                        attach_names, attach_paths = download_attachments(
                            driver, page_html, item["제목"]
                        )
                        item["첨부파일"]     = attach_names
                        item["첨부파일경로"] = attach_paths
                        item["매칭의료어"]   = get_matched_keywords(combined, MEDICAL_KEYWORDS)
                        item["매칭AI어"]     = get_matched_keywords(combined, AI_KEYWORDS)
                        results.append(item)
                        new_count += 1
        print(f"    → p{page}: 파싱 {len(items)}건 / 의료필터 {new_count}건")
        if not has_next_page(html, page):
            break
    return results

def save_checkpoint(all_results: list, ts: str):
    if not all_results:
        return
    os.makedirs(SAVE_DIR, exist_ok=True)
    df   = pd.DataFrame(all_results)
    path = os.path.join(SAVE_DIR, f"msit_ai_medical_{ts}_checkpoint.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  💾 중간 저장 → {path} ({len(df)}건)")

def print_summary(df, csv_path, xlsx_path, elapsed_sec, total_combos):
    SEP = "=" * 62
    sep = "-" * 62
    print("\n" + SEP)
    print("  📊 수집 완료 요약")
    print(SEP)
    print(f"  총 수집 건수     : {len(df):,}건")
    print(f"  고유 URL 수      : {df['URL'].nunique():,}개")
    print(f"  처리 조합 수     : {total_combos}개")
    mins, secs = divmod(int(elapsed_sec), 60)
    print(f"  소요 시간        : {mins}분 {secs}초")
    print(f"  CSV  저장        : {csv_path}")
    print(f"  XLSX 저장        : {xlsx_path}")
    attach_cnt = (df["첨부파일"] != "").sum()
    print(f"  첨부파일 다운로드: {attach_cnt:,}건 → {ATTACH_DIR}")

    for label, col in [("게시판", "게시판"), ("키워드", "검색키워드")]:
        print(f"\n  [ {label}별 건수 ]")
        print(sep)
        cnt = df[col].value_counts()
        for k, v in cnt.items():
            bar = "█" * min(int(v / max(cnt) * 20), 20)
            print(f"  {k:<16} {v:>4}건  {bar}")

    print(f"\n  [ 매칭 의료 키워드 TOP10 ]")
    print(sep)
    counter = Counter()
    for cell in df["매칭의료어"].dropna():
        for kw in [k.strip() for k in cell.split(",") if k.strip()]:
            counter[kw] += 1
    if counter:
        top = counter.most_common(10)
        for kw, v in top:
            bar = "█" * min(int(v / top[0][1] * 20), 20)
            print(f"  {kw:<10} {v:>4}건  {bar}")

    if "연도" in df.columns and df["연도"].notna().any():
        print(f"\n  [ 연도별 건수 ]")
        print(sep)
        yr = df[df["연도"] != ""]["연도"].value_counts().sort_index()
        if len(yr):
            for y, v in yr.items():
                bar = "█" * min(int(v / yr.max() * 20), 20)
                print(f"  {y}년  {v:>4}건  {bar}")
    print(SEP + "\n")

def main():
    os.makedirs(SAVE_DIR,   exist_ok=True)
    os.makedirs(ATTACH_DIR, exist_ok=True)
    all_results, seen_urls = [], set()
    total   = len(BOARDS) * len(AI_KEYWORDS)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start = time.time()

    print("=" * 62)
    print("  과학기술정보통신부 크롤러 v15")
    print("  (의료 키워드 필터링 → 첨부파일 다운로드)")
    print(f"  {len(BOARDS)}개 게시판 × {len(AI_KEYWORDS)}개 키워드 = {total}개 조합")
    print(f"  첨부파일 저장: {ATTACH_DIR}")
    print("=" * 62)

    driver = init_driver(download_dir=ATTACH_DIR)
    try:
        for idx, board in enumerate(BOARDS):
            for jdx, kw in enumerate(AI_KEYWORDS):
                combo = idx * len(AI_KEYWORDS) + jdx + 1
                print(f"[{combo}/{total}] {board['name']} / \'{kw}\' ...")
                items = collect_board(driver, board, kw)
                new   = [i for i in items if i["URL"] not in seen_urls]
                seen_urls.update(i["URL"] for i in new)
                all_results.extend(new)
                print(f"  → 필터 통과: {len(new)}건 / 누적: {len(all_results)}건")
                save_checkpoint(all_results, ts)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 중단 — 데이터 저장 중...")
    finally:
        driver.quit()
        print("[드라이버 종료]")

    elapsed = time.time() - t_start
    if all_results:
        df   = pd.DataFrame(all_results)
        csv  = os.path.join(SAVE_DIR, f"msit_ai_medical_{ts}_final.csv")
        xlsx = os.path.join(SAVE_DIR, f"msit_ai_medical_{ts}_final.xlsx")
        df.to_csv(csv,  index=False, encoding="utf-8-sig")
        df.to_excel(xlsx, index=False, engine="openpyxl")
        print_summary(df, csv, xlsx, elapsed, total)
    else:
        print("\n수집된 데이터 없음")

if __name__ == "__main__":
    main()
