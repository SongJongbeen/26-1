import markdown
from xhtml2pdf import pisa
from pathlib import Path

CSS_STYLE = """
@page {
    size: A4;
    margin: 2cm;
}
body {
    font-family: HYGothic-Medium;
    font-size: 11pt;
    line-height: 1.6;
}

/* 중첩 리스트 들여쓰기 */
ul, ol {
    margin: 0;
    padding-left: 1.5em;
}
ul ul, ol ol, ul ol, ol ul {
    padding-left: 1.5em;
    margin-top: 0.2em;
    margin-bottom: 0.2em;
}
li {
    margin-bottom: 0.3em;
}

/* 헤딩 계층 강조 */
h1 { font-size: 18pt; margin-top: 1em; }
h2 { font-size: 15pt; margin-top: 0.8em; padding-left: 0.5em; border-left: 3px solid #555; }
h3 { font-size: 13pt; margin-top: 0.6em; padding-left: 1em; }
h4 { font-size: 11pt; margin-top: 0.4em; padding-left: 1.5em; }

/* 코드 블록 */
pre {
    background-color: #f4f4f4;
    padding: 0.5em;
    font-size: 9pt;
}
code {
    font-family: monospace;
    font-size: 9pt;
}
"""

def md_to_pdf(md_file: str, pdf_file: str):
    with open(md_file, 'r', encoding='utf-8') as file:
        text = file.read()

    html_body = markdown.markdown(
        text,
        extensions=['tables', 'fenced_code', 'sane_lists'],
        tab_length=2  # 탭 1개 = 4스페이스로 중첩 레벨 인식
    )

    html_full = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <style>{CSS_STYLE}</style>
</head>
<body>{html_body}</body>
</html>"""

    with open(pdf_file, 'wb') as output_file:
        result = pisa.CreatePDF(html_full, dest=output_file)

    if result.err:
        print(f"PDF 변환 오류 발생: {result.err}")
    else:
        print(f"PDF 저장 완료: {pdf_file}")

if __name__ == "__main__":
    md_file = "finals/sos797/final_note.md"
    pdf_file = "finals/sos797/final_note.pdf"
    md_to_pdf(md_file, pdf_file)
