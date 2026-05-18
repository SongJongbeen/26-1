import os
import time
import requests
from urllib.parse import urljoin
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# --- 설정(Configuration) ---
BASE_URL = "https://www.mfds.go.kr/brd/m_99/list.do"
DOWNLOAD_DIR = "./mfds_downloads"
START_DATE = datetime.strptime("2026-05-15", "%Y-%m-%d")
END_DATE = datetime.strptime("2016-01-01", "%Y-%m-%d")

# 다운로드 폴더 생성
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

def setup_driver():
    """Selenium Webdriver 설정"""
    chrome_options = Options()
    # chrome_options.add_argument('--headless') # 디버깅 완료 후 주석 해제 추천
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(5)
    return driver

def get_requests_session(driver):
    """Selenium의 쿠키와 User-Agent를 Requests 세션으로 복사"""
    session = requests.Session()
    agent = driver.execute_script("return navigator.userAgent")
    session.headers.update({"User-Agent": agent})
    for cookie in driver.get_cookies():
        session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])
    return session

def download_file(session, file_url, file_name):
    """Requests를 이용해 파일 다운로드"""
    try:
        # 파일명에 포함될 수 있는 사용할 수 없는 특수문자 제거
        valid_file_name = "".join(i for i in file_name if i not in r'\/:*?"<>|')
        file_path = os.path.join(DOWNLOAD_DIR, valid_file_name)
        
        # 이미 존재하는 파일이면 스킵
        if os.path.exists(file_path):
            print(f"  [스킵] 이미 다운로드된 파일: {valid_file_name}")
            return

        # 스트림 방식으로 파일 다운로드
        response = session.get(file_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  [완료] {valid_file_name}")
        
    except Exception as e:
        print(f"  [실패] {file_name}: {e}")

def main():
    driver = setup_driver()
    page_index = 1
    keep_crawling = True
    
    try:
        # 최초 1회 접속하여 쿠키 생성
        driver.get(BASE_URL)
        time.sleep(2)
        session = get_requests_session(driver)
        
        while keep_crawling:
            print(f"\n=== {page_index} 페이지 크롤링 중 ===")
            list_url = f"{BASE_URL}?page={page_index}"
            driver.get(list_url)
            time.sleep(1.5)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # 1. 목록의 li 태그들 추출 (페이지네이션 등 다른 li와 구분하기 위해 center_column 포함 여부 확인)
            board_list = [li for li in soup.find_all("li") if li.select_one("div.center_column")]
            
            if not board_list:
                print("게시글 목록이 없습니다. 크롤링을 종료합니다.")
                break
                
            for row in board_list:
                # 2. 날짜 파싱 ("2026-05-15" 형태)
                date_div = row.select_one("div.right_column")
                if not date_div:
                    continue
                
                date_str = date_div.text.strip()
                try:
                    post_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue

                # 날짜 조건 필터링
                if post_date > START_DATE:
                    continue
                
                if post_date < END_DATE:
                    print(f"\n목표 날짜({END_DATE.strftime('%Y-%m-%d')})에 도달하여 전체 작업을 종료합니다.")
                    keep_crawling = False
                    break
                
                # 3. 상세 페이지 URL 추출
                title_link = row.select_one("a.title")
                if not title_link:
                    continue
                
                href = title_link.get('href')
                # urljoin을 사용해 상대경로(./view.do)를 절대경로로 안전하게 변환
                detail_url = urljoin(BASE_URL, href)
                
                # 4. 상세 페이지 파싱 (Requests로 빠르게 처리)
                detail_response = session.get(detail_url)
                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                
                # 5. 첨부파일 다운로드 링크 찾기
                file_items = detail_soup.select("div.bv_file_box ul.bbs_file_view_list > li")
                
                if file_items:
                    print(f"[{post_date.strftime('%Y-%m-%d')}] 게시글 탐색 중...")
                    for f_item in file_items:
                        name_tag = f_item.select_one("strong")
                        link_tag = f_item.select_one("a.bbs_icon_filedown")
                        
                        if name_tag and link_tag:
                            file_name = name_tag.text.strip()
                            file_href = link_tag.get('href')
                            down_url = urljoin(BASE_URL, file_href)
                            
                            # 반복문을 통해 hwpx, pdf 등 모두 순차적으로 다운로드
                            download_file(session, down_url, file_name)

            if not keep_crawling:
                break
                
            page_index += 1

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        driver.quit()
        print("\n크롤링이 완료되었습니다.")

if __name__ == "__main__":
    main()