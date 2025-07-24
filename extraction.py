import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='ben')
    return text

raw_text = extract_text_with_ocr("HSC26.pdf")


with open("text.txt", "w", encoding="utf-8") as f:
    f.write(raw_text)