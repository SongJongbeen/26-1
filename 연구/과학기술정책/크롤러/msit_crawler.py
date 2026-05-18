import os
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# --- 설정(Configuration) ---
BASE_URL = "https://www.msit.go.kr/bbs/list.do?sCode=user&mPid=208&mId=307"
# 절대 경로로 지정해야 Selenium 자동 다운로드가 정상 작동합니다.
DOWNLOAD_DIR = os.path.abspath("./msit_downloads")
START_DATE = datetime.strptime("2026-05-15", "%Y-%m-%d")
END_DATE = datetime.strptime("2016-01-01", "%Y-%m-%d")

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
        is_downloading = any(filename.endswith('.crdownload') for filename in os.listdir(DOWNLOAD_DIR))
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
            print(f"\n=== {page_index} 페이지 탐색 중 ===")
            list_url = f"{BASE_URL}&pageIndex={page_index}"
            driver.get(list_url)
            time.sleep(2) # 페이지 로딩 대기
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # 1. 전달받은 HTML에 맞춘 게시글 목록 추출
            board_list = soup.select("div.toggle")
            
            if not board_list:
                print("게시글 목록을 찾을 수 없거나 마지막 페이지입니다.")
                break
            
            # 현재 페이지에서 접근해야 할 게시글 ID 목록을 먼저 수집합니다.
            posts_to_visit = []
            
            for item in board_list:
                # 공지사항(상단 고정글) 스킵: 번호가 숫자가 아니면 공지글로 판단
                num_div = item.select_one("div.num")
                if not num_div or not num_div.text.strip().isdigit():
                    continue

                # 2. 날짜 파싱 ("2026. 5. 15" 형식 대응)
                date_div = item.select_one("div.date")
                if not date_div:
                    continue
                
                date_str = date_div.text.strip().replace(" ", "") # "2026.5.15" 로 변환
                try:
                    post_date = datetime.strptime(date_str, "%Y.%m.%d")
                except ValueError:
                    continue

                # 날짜 조건 필터링
                if post_date > START_DATE:
                    continue # 시작일보다 미래의 글이면 건너뜀
                
                if post_date < END_DATE:
                    print(f"\n목표 날짜({END_DATE.strftime('%Y-%m-%d')}) 도달. 크롤링을 완전히 종료합니다.")
                    keep_crawling = False
                    break # 과거 글로 넘어가면 반복 종료
                
                # 3. 상세 페이지 이동용 자바스크립트 함수 추출 (fn_detail(3187323))
                a_tag = item.select_one("a")
                onclick_attr = a_tag.get('onclick', '') if a_tag else ''
                
                # 정규식으로 숫자(ID)만 추출
                match = re.search(r"fn_detail\((\d+)\)", onclick_attr)
                if match:
                    post_id = match.group(1)
                    posts_to_visit.append((post_id, post_date))
            
            # 4. 수집한 ID를 바탕으로 상세페이지 진입 및 다운로드
            for post_id, p_date in posts_to_visit:
                # 안전한 실행을 위해 매번 목록 페이지 상태로 초기화
                driver.get(list_url)
                time.sleep(1)
                
                # 자바스크립트 실행으로 상세페이지 진입
                driver.execute_script(f"fn_detail({post_id});")
                time.sleep(2) # 상세페이지 로딩 대기
                
                # 상세페이지 파싱
                detail_soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                # 5. 전달받은 HTML에 맞춘 첨부파일 다운로드 영역 찾기
                down_links = detail_soup.select("ul.down_file li a.down")
                
                for down in down_links:
                    title_attr = down.get('title', '')
                    
                    # 'ODT 다운로드' 버튼 제외
                    if "ODT" in title_attr:
                        continue 
                    
                    # 다운로드 자바스크립트 추출 (fn_download('53770', '1', 'hwpx');)
                    js_download = down.get('onclick', '')
                    if js_download:
                        print(f"[{p_date.strftime('%Y-%m-%d')}] 파일 다운로드 실행...")
                        driver.execute_script(js_download)
                        time.sleep(1)

                        wait_for_downloads()
            
            if not keep_crawling:
                break
                
            page_index += 1

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        driver.quit()
        print("\n모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
