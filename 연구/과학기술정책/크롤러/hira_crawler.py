import os
import time
from urllib.parse import urljoin
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# --- 설정(Configuration) ---
BASE_URL = "https://www.hira.or.kr/bbsDummy.do?pgmid=HIRAA020041000100&WT.gnb=%EB%B3%B4%EB%8F%84%EC%9E%90%EB%A3%8C"
DOWNLOAD_DIR = os.path.abspath("./hira_downloads")
START_DATE = datetime.strptime("2026-05-15", "%Y-%m-%d")
END_DATE = datetime.strptime("2016-01-01", "%Y-%m-%d")

# 다운로드 폴더 생성
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

def setup_driver():
    """Selenium Webdriver 및 자동 다운로드 설정"""
    chrome_options = Options()
    
    # 크롬 자동 다운로드 환경 설정 (팝업 없이 지정된 폴더로 바로 다운로드)
    prefs = {
        "download.default_directory": DOWNLOAD_DIR,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True 
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(5)
    return driver

def wait_for_downloads(timeout=30):
    """파일이 다운로드 완료될 때까지 대기 (.crdownload 확장자가 없어질 때까지)"""
    seconds = 0
    while seconds < timeout:
        time.sleep(1)
        is_downloading = any(filename.endswith('.crdownload') or filename.endswith('.tmp') 
                             for filename in os.listdir(DOWNLOAD_DIR))
        if not is_downloading:
            return True
        seconds += 1
    return False

def main():
    driver = setup_driver()
    page_index = 1
    keep_crawling = True
    
    try:
        while keep_crawling:
            print(f"\n=== {page_index} 페이지 크롤링 중 ===")
            list_url = f"{BASE_URL}&pageIndex={page_index}"
            driver.get(list_url)
            time.sleep(2)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # 1. 게시글 목록 추출
            board_list = soup.select("table tbody tr")
            
            if not board_list:
                print("게시글 목록을 찾을 수 없거나 마지막 페이지입니다.")
                break
                
            # 현재 페이지의 게시글 링크와 날짜를 먼저 수집 (화면 이동 시 DOM 초기화 방지)
            posts_to_visit = []
            
            for row in board_list:
                # 공지사항 스킵: 번호 열의 텍스트가 숫자인지 판별
                num_td = row.select_one("td.col-num")
                if not num_td or not num_td.text.strip().isdigit():
                    continue

                # 2. 날짜 파싱 ("2026-05-15" 형태)
                date_td = row.select_one("td.col-date")
                if not date_td:
                    continue
                
                date_str = date_td.text.strip()
                try:
                    post_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue

                # 날짜 조건 필터링
                if post_date > START_DATE:
                    continue
                
                if post_date < END_DATE:
                    print(f"\n목표 날짜({END_DATE.strftime('%Y-%m-%d')}) 도달. 크롤링을 종료합니다.")
                    keep_crawling = False
                    break
                
                # 3. 상세 페이지 URL 추출
                title_link = row.select_one("td.col-tit a")
                if title_link:
                    href = title_link.get('href')
                    # href가 "?pgmid=..." 형태이므로 urljoin으로 합침
                    detail_url = urljoin(BASE_URL, href)
                    posts_to_visit.append((detail_url, post_date))

            # 4. 수집한 링크를 바탕으로 상세페이지 진입 및 다운로드
            for detail_url, p_date in posts_to_visit:
                driver.get(detail_url)
                time.sleep(1.5)
                
                detail_soup = BeautifulSoup(driver.page_source, 'html.parser')
                file_items = detail_soup.select("div.fileBox ul li")
                
                print(f"[{p_date.strftime('%Y-%m-%d')}] 게시글 첨부파일 확인 중...")
                
                for f_item in file_items:
                    # '자료가 다운되지 않을 경우...' 같은 안내 문구 li는 a 태그가 없으므로 스킵
                    down_btn = f_item.select_one("a.btn_file")
                    if not down_btn:
                        continue
                        
                    # --- 1차 필터링: 아이콘(img) 기반 제외 ---
                    icon = f_item.select_one("i")
                    if icon:
                        icon_title = icon.get('title', '').lower()
                        icon_class = " ".join(icon.get('class', [])).lower()
                        
                        if 'img' in icon_title or 'img' in icon_class:
                            print("  [스킵] 이미지 파일 발견 (아이콘 기준)")
                            continue
                    
                    # 다운로드 스크립트 추출 (예: onclick="downLoadBbs('1','11797','1','158');")
                    js_download = down_btn.get('onclick', '')
                    if js_download:
                        # 다운로드 전 폴더 내 파일 목록 저장 (새로 추가된 파일을 찾기 위함)
                        before_files = set(os.listdir(DOWNLOAD_DIR))
                        
                        # 파일 다운로드 실행
                        driver.execute_script(js_download)
                        time.sleep(1) # 다운로드 시작 대기
                        wait_for_downloads() # 완료 대기
                        
                        # --- 2차 필터링: 실제 다운로드된 파일 확장자 확인 ---
                        after_files = set(os.listdir(DOWNLOAD_DIR))
                        new_files = after_files - before_files
                        
                        for new_file in new_files:
                            file_ext = os.path.splitext(new_file)[1].lower()
                            file_path = os.path.join(DOWNLOAD_DIR, new_file)
                            
                            # 조건: .hwp 이거나 이미지 확장자인 경우 즉시 삭제 (.hwpx는 유지)
                            if file_ext == '.hwp' or file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                                os.remove(file_path)
                                print(f"  [삭제] 조건에 맞지 않는 파일 (확장자 {file_ext}): {new_file}")
                            else:
                                print(f"  [완료] 다운로드 성공: {new_file}")
            
            if not keep_crawling:
                break
                
            page_index += 1

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        driver.quit()
        print("\n모든 크롤링 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()