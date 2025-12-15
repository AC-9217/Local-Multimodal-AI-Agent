import fitz
from PIL import Image
import os

def create_pdf():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is a test paper about Computer Vision and Deep Learning. Transformers are great.")
    doc.save("test_paper.pdf")
    print("Created test_paper.pdf")

def create_image():
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save("test_image.jpg")
    print("Created test_image.jpg")

if __name__ == "__main__":
    create_pdf()
    create_image()
