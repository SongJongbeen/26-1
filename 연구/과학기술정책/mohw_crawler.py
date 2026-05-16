import os
import time
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# --- 설정(Configuration) ---
BASE_URL = "https://www.mohw.go.kr/board.es?mid=a10503010100&bid=0027"
DOWNLOAD_DIR = "./mohw_downloads"
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
            # 보건복지부 페이지 파라미터는 보통 nPage를 사용합니다.
            list_url = f"{BASE_URL}&nPage={page_index}"
            driver.get(list_url)
            time.sleep(1.5)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # 1. 목록의 tr 태그들 추출 (tbody 내의 tr)
            board_list = soup.select("table tbody tr")
            
            if not board_list:
                print("게시글 목록이 없습니다. 크롤링을 종료합니다.")
                break
                
            for row in board_list:
                # 공지사항 스킵: 번호 열의 텍스트가 숫자인지 판별
                num_td = row.select_one("td[data-label='번호']")
                if not num_td or not num_td.text.strip().isdigit():
                    continue

                # 2. 날짜 파싱 ("2026-05-15" 형태)
                date_td = row.select_one("td[data-label='등록일']")
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
                    print(f"\n목표 날짜({END_DATE.strftime('%Y-%m-%d')})에 도달하여 전체 작업을 종료합니다.")
                    keep_crawling = False
                    break
                
                # 3. 상세 페이지 URL 추출
                title_link = row.select_one("td.txt_left a")
                if not title_link:
                    continue
                
                href = title_link.get('href')
                detail_url = "https://www.mohw.go.kr" + href if href.startswith("/") else href
                
                # 4. 상세 페이지 파싱 (Selenium 없이 Requests로 빠르게 처리)
                detail_response = session.get(detail_url)
                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                
                # 5. 첨부파일 다운로드 링크 찾기
                # a 태그 내 텍스트에 "다운로드"가 포함된 요소 추출
                down_links = detail_soup.select("div.file ul.list li span.link a")
                
                print(f"[{post_date.strftime('%Y-%m-%d')}] 게시글 탐색 중...")
                for down_btn in down_links:
                    if "다운로드" in down_btn.text:
                        file_href = down_btn.get('href')
                        
                        # 파일명은 title 속성에 담겨있음 ("파일명.hwpx" 형식)
                        raw_file_name = down_btn.get('title', '알수없는파일')
                        
                        down_url = "https://www.mohw.go.kr" + file_href if file_href.startswith("/") else file_href
                        
                        # pdf, hwpx 등 여러 개가 있어도 반복문을 통해 전부 다운로드됨
                        download_file(session, down_url, raw_file_name)

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