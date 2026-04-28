from playwright.sync_api import sync_playwright
import os

os.makedirs("oii_faculty_pages", exist_ok=True)

with open("oii_faculty_links.txt", "r", encoding="utf-8") as f:
    links = [link.strip() for link in f.readlines() if link.strip()]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for link in links:
        try:
            # 네트워크 통신이 멈출 때까지(로딩이 거의 끝날 때까지) 대기합니다.
            page.goto(link, wait_until="networkidle")
            
            # 요소가 화면에 '보일 때(visible)'까지 기다리는 코드는 에러를 유발하므로 삭제하고,
            # 단순히 body 태그가 DOM에 '존재(attached)'하는지만 확인합니다.
            page.wait_for_selector("body", state="attached")
            
            # 특정 div나 main을 찾는 대신, body 전체의 텍스트를 가져옵니다.
            # inner_text()는 눈에 보이는 텍스트를 가져옵니다.
            main_text = page.locator("body").inner_text()
            
            # 파일 이름 추출
            filename = link.strip('/').split('/')[-1]

            # .txt 파일로 저장
            with open(f"oii_faculty_pages/{filename}.txt", "w", encoding="utf-8") as f:
                # 혹시 앞뒤 공백이나 불필요한 줄바꿈이 많을 수 있으므로 strip() 처리
                f.write(main_text.strip())
            print(f"Saved {filename}.txt")
        except Exception as e:
            print(f"Error on {link}: {e}")
            

    browser.close()
