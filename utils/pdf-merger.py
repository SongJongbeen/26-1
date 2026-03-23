# python code that merges multiple pdf files into one from the given folder
from pypdf import PdfReader, PdfWriter
from pathlib import Path

def merge_pdfs(input_folder: str, output_file: str):
    writer = PdfWriter()
    pdf_files = sorted(Path(input_folder).glob("*.pdf"))  # 폴더 내 PDF 파일 목록

    for pdf_path in pdf_files:
        reader = PdfReader(pdf_path)  # 파일별로 PdfReader 생성
        for page in reader.pages:
            writer.add_page(page)

    with open(output_file, "wb") as f:  # 바이너리 쓰기 모드로 출력
        writer.write(f)

if __name__ == "__main__":
    input_folder = "finals/sos797/presentation"
    output_file = "finals/sos797/presentation.pdf"
    merge_pdfs(input_folder, output_file)
