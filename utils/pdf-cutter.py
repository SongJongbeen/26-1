from pypdf import PdfReader, PdfWriter
from pathlib import Path

def extract_pdf_pages(input_pdf: str, start_page: int, end_page: int, output_pdf: str = None) -> bool:
    """
    Extract a range of pages from a PDF and save to a new file.
    
    Args:
        input_pdf (str): Path to input PDF file
        start_page (int): Starting page number (1-indexed, inclusive)
        end_page (int): Ending page number (1-indexed, inclusive)
        output_pdf (str): Path to output PDF file. If None, creates auto-named file
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        extract_pdf_pages("document.pdf", 5, 10, "pages_5-10.pdf")
    """
    try:
        # Validate input file exists
        if not Path(input_pdf).exists():
            raise FileNotFoundError(f"Input file not found: {input_pdf}")
        # Open the PDF
        reader = PdfReader(input_pdf)
        total_pages = len(reader.pages)
        # Validate page range
        if start_page < 1 or end_page < 1:
            raise ValueError("Page numbers must be 1-indexed (starting from 1)")
        if start_page > end_page:
            raise ValueError("start_page must be <= end_page")
        if start_page > total_pages:
            raise ValueError(f"start_page ({start_page}) exceeds total pages ({total_pages})")

        # Adjust end_page if it exceeds total pages
        end_page = min(end_page, total_pages)

        # Create writer and extract pages
        writer = PdfWriter()
        for page_num in range(start_page - 1, end_page):  # Convert to 0-indexed
            writer.add_page(reader.pages[page_num])

        # Generate output filename if not provided
        if output_pdf is None:
            input_path = Path(input_pdf)
            output_pdf = input_path.stem + f"_pages_{start_page}-{end_page}.pdf"

        # Write to file
        with open(output_pdf, 'wb') as f:
            writer.write(f)

        pages_extracted = end_page - start_page + 1
        print(f"✓ Successfully extracted {pages_extracted} page(s) to: {output_pdf}")
        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


# Interactive mode with user input
if __name__ == "__main__":
    print("=== PDF Page Range Extractor ===\n")
    input_pdf = input("Enter input PDF path: ").strip()

    try:
        # Show total pages
        reader = PdfReader(input_pdf)
        total = len(reader.pages)
        print(f"Total pages: {total}\n")

        start = int(input(f"Enter start page (1-{total}): "))
        end = int(input(f"Enter end page ({start}-{total}): "))

        output_pdf = input("Enter output PDF path (press Enter for auto-name): ").strip() or None

        extract_pdf_pages(input_pdf, start, end, output_pdf)
    except FileNotFoundError:
        print(f"✗ File not found: {input_pdf}")
    except ValueError as e:
        print(f"✗ Invalid input: {e}")
