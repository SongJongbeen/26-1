import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

def web_to_knowledge_file(url, output_filename):
    # 1. 웹페이지 가져오기 (가짜 User-Agent로 차단 방지)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # 2. BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # 3. 불필요한 요소 제거 (광고, 네비게이션, 푸터, 스크립트 등)
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()

    # Tip: 대부분의 정보성 글은 <article> 이나 class="content" 등에 들어있습니다.
    # 만약 특정 사이트의 본문만 정확히 타겟팅하려면 아래처럼 수정 가능합니다.
    # main_content = str(soup.find('article')) 
    main_content = str(soup.body) # 일단 body 전체를 가져옵니다.

    # 4. HTML을 마크다운(Markdown)으로 변환
    # heading_style='ATX'는 #, ## 같은 직관적인 마크다운 헤더를 사용하게 합니다.
    markdown_text = md(main_content, heading_style="ATX")

    # 5. 빈 줄이 너무 많을 수 있으므로 약간의 정제 (선택 사항)
    cleaned_text = '\n'.join([line for line in markdown_text.splitlines() if line.strip() != ''])

    # 6. 파일로 저장 (.md 혹은 .txt)
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"✅ 성공적으로 저장되었습니다: {output_filename}")

# === 실행 예시 ===
# FM Scout나 가이드 웹사이트의 URL을 넣으세요.
target_url = "https://www.fmscout.com/a-football-manager-2026-top-bargains.html"
web_to_knowledge_file(target_url, "FM26_Top_Bargains_Guide.md")
