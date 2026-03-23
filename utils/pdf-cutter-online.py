import gradio as gr
import pymupdf
import tempfile
import os
from PIL import Image

# ─── 핵심 함수들 ───────────────────────────────────────────────

def load_pdf(pdf_file):
    """PDF 업로드 시 모든 페이지를 이미지로 변환하고 체크박스 목록 생성"""
    if pdf_file is None:
        return [], gr.update(choices=[], value=[]), [], "PDF를 업로드해주세요."
    
    try:
        doc = pymupdf.open(pdf_file)
        images, choices = [], []
        
        for i in range(len(doc)):
            page = doc[i]
            mat = pymupdf.Matrix(1.5, 1.5)  # 1.5x 해상도
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((img, f"페이지 {i + 1}"))
            choices.append(f"페이지 {i + 1}")
        
        doc.close()
        return (
            images,
            gr.update(choices=choices, value=choices),  # 기본값: 전체 선택
            choices,
            f"✅ 총 {len(choices)}페이지 로드 완료"
        )
    except Exception as e:
        return [], gr.update(choices=[], value=[]), [], f"❌ 오류: {str(e)}"


def select_all(choices_state):
    return gr.update(value=choices_state)


def deselect_all():
    return gr.update(value=[])


def save_pdf(pdf_file, selected, output_name, choices_state):
    """선택된 페이지만 추출하여 새 PDF로 저장"""
    if pdf_file is None:
        return None, "❌ PDF를 먼저 업로드해주세요."
    if not selected:
        return None, "❌ 저장할 페이지를 최소 1개 선택하세요."
    
    fname = (output_name or "output").strip()
    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"
    
    # "페이지 N" → 0-based index 변환 후 정렬
    page_nums = sorted(int(p.replace("페이지 ", "")) - 1 for p in selected)
    
    try:
        src = pymupdf.open(pdf_file)
        out = pymupdf.open()
        
        for pn in page_nums:
            out.insert_pdf(src, from_page=pn, to_page=pn)
        
        save_path = os.path.join(tempfile.gettempdir(), fname)
        out.save(save_path)
        src.close()
        out.close()
        
        nums_str = ", ".join(str(p + 1) for p in page_nums)
        return save_path, f"✅ 저장 완료: {fname} | {len(page_nums)}페이지 ({nums_str})"
    except Exception as e:
        return None, f"❌ 저장 오류: {str(e)}"


# ─── Gradio UI ──────────────────────────────────────────────────

with gr.Blocks(title="PDF 페이지 편집기", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📄 PDF 페이지 편집기\nPDF를 업로드하고 원하는 페이지를 선택하여 새 PDF로 저장하세요.")

    choices_state = gr.State([])  # 현재 로드된 페이지 목록 저장

    with gr.Row():

        # ── 왼쪽: 컨트롤 패널 ──
        with gr.Column(scale=1, min_width=280):

            pdf_input = gr.File(
                label="📤 PDF 파일 업로드",
                file_types=[".pdf"],
                type="filepath"
            )
            status = gr.Textbox(
                label="📊 상태",
                value="PDF를 업로드해주세요.",
                interactive=False,
                max_lines=2
            )

            gr.Markdown("---")
            gr.Markdown("### 🗂️ 페이지 선택")

            with gr.Row():
                btn_select_all   = gr.Button("✅ 모두 선택", variant="secondary", size="sm")
                btn_deselect_all = gr.Button("☐ 모두 해제",  variant="secondary", size="sm")

            page_selector = gr.CheckboxGroup(
                label="저장할 페이지",
                choices=[],
                value=[],
                interactive=True
            )

            gr.Markdown("---")
            gr.Markdown("### 💾 저장")

            output_filename = gr.Textbox(
                label="저장 파일명",
                value="output.pdf",
                placeholder="예: my_document.pdf"
            )
            btn_save    = gr.Button("📥 선택 페이지 PDF 저장", variant="primary")
            output_file = gr.File(label="다운로드", interactive=False)

        # ── 오른쪽: 페이지 미리보기 ──
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="📖 페이지 미리보기",
                columns=3,
                height=780,
                object_fit="contain",
                show_label=True
            )

    # ─── 이벤트 연결 ───────────────────────────────────────────

    pdf_input.change(
        fn=load_pdf,
        inputs=[pdf_input],
        outputs=[gallery, page_selector, choices_state, status]
    )

    btn_select_all.click(
        fn=select_all,
        inputs=[choices_state],
        outputs=[page_selector]
    )

    btn_deselect_all.click(
        fn=deselect_all,
        inputs=[],
        outputs=[page_selector]
    )

    btn_save.click(
        fn=save_pdf,
        inputs=[pdf_input, page_selector, output_filename, choices_state],
        outputs=[output_file, status]
    )

if __name__ == "__main__":
    demo.launch()
