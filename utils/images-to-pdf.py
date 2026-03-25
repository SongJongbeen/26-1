from PIL import Image
import os

def images_to_pdf(input_folder: str, output_pdf: str):
    # Get all images in the input folder
    images = [f for f in os.listdir(input_folder) if f.endswith(('.JPG'))]
    # Sort images by name
    images.sort()

    # Convert all images to PIL Image objects in RGB mode
    image_list = []
    for image in images:
        img_path = os.path.join(input_folder, image)
        with Image.open(img_path) as img:
            # Convert to RGB (PDF format requires RGB)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image_list.append(img.copy())

    # Save as PDF
    if image_list:
        image_list[0].save(
            output_pdf,
            save_all=True,
            append_images=image_list[1:],
            optimize=False
        )
        print(f"PDF created successfully: {output_pdf}")
    else:
        print(f"No images found in {input_folder}")


if __name__ == "__main__":
    week = input("Enter week number (e.g. 01, 02, 03, etc.): ")
    input_folder = f"{week}W"
    output_pdf = f"{week}W.pdf"
    images_to_pdf(input_folder, output_pdf)
