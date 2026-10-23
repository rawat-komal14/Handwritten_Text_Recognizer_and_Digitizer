import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# IMPORTANT: Point pytesseract to your installed Tesseract executable (Update path if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clean_notebook_lines_and_segment(image_path):
    """
    Removes horizontal ruling lines from notebook paper and isolates text lines.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Binarize image (Otsu Thresholding)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Detect and remove horizontal ruling lines
    # Create horizontal kernel to detect long thin lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Subtract horizontal rule lines from the binary text image
    clean_binary = cv2.subtract(thresh, detected_lines)

    # Invert back to black text on white paper for OCR
    clean_ocr_input = cv2.bitwise_not(clean_binary)
    cv2.imwrite("temp_ocr_clean.png", clean_ocr_input)

    return clean_ocr_input, thresh

st.title("Handwritten Document Text Recognizer")
st.write("Upload a handwritten notebook page to extract digitized text.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with open("temp_input.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Handwritten Page", use_container_width=True)

    if st.button("Digitize Document"):
        with st.spinner("Removing notebook ruling lines and recognizing text..."):
            try:
                # 1. Clean notebook ruling lines
                clean_img, thresh = clean_notebook_lines_and_segment("temp_input.png")

                # 2. Configure Tesseract PSM 6 (Assume a single uniform block of text)
                # --psm 6 forces Tesseract to read line-by-line from top to bottom
                custom_config = r'--oem 3 --psm 6'
                extracted_text = pytesseract.image_to_string(clean_img, config=custom_config)

                # 3. Get line/word bounding boxes for visualization
                data = pytesseract.image_to_data(clean_img, config=custom_config, output_type=pytesseract.Output.DICT)
                
                vis_img = cv2.imread("temp_input.png")
                n_boxes = len(data['text'])
                
                for i in range(n_boxes):
                    if int(data['conf'][i]) > 15 and data['text'][i].strip() != '':
                        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 4. Display Results
                st.subheader("Digitized Text Output:")
                if extracted_text.strip():
                    st.code(extracted_text, language="text")
                else:
                    st.warning("No text could be extracted. Please check if Tesseract path is set correctly.")

                st.subheader("Detected Text Regions:")
                st.image(vis_img, caption="Line-by-Line Bounding Box Extraction", use_container_width=True)

            except Exception as e:
                st.error(f"Error executing OCR: {e}")
                st.info("Ensure Tesseract-OCR is installed on your PC and the path in `pytesseract.tesseract_cmd` is correct.")