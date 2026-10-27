import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
from datetime import datetime

# Streamlit Page Config
st.set_page_config(page_title="Handwritten Recognizer", layout="wide")

@st.cache_resource
def load_trocr():
    """
    Loads TrOCR processor and model ONCE into system RAM.
    Deferred import prevents Streamlit dashboard startup freezing.
    """
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    torch.set_num_threads(4)
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    model.eval()
    return processor, model

@st.cache_resource
def load_metrics():
    """
    Caches CER and WER metrics in memory.
    """
    import evaluate
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    return cer_metric, wer_metric

def clean_recognized_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith(('"', "'", "#", "`", ",", ".")):
        cleaned = cleaned[1:].strip()
    return cleaned

def save_benchmark_to_file(model_name: str, cer: float, wer: float, char_acc: float, word_acc: float, pred_text: str, ref_text: str):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "mode": "Single Line",
        "cer": round(float(cer), 4),
        "wer": round(float(wer), 4),
        "char_accuracy_pct": round(float(char_acc), 2),
        "word_accuracy_pct": round(float(word_acc), 2),
        "predicted_text": pred_text,
        "ground_truth_text": ref_text
    }

    # 1. Update JSON Benchmark File
    json_file = "model_benchmark_results.json"
    data = []
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
            
    data.append(log_entry)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # 2. Update CSV Benchmark File
    csv_file = "model_benchmark_results.csv"
    df_new = pd.DataFrame([log_entry])
    if not os.path.exists(csv_file):
        df_new.to_csv(csv_file, index=False, encoding="utf-8")
    else:
        df_new.to_csv(csv_file, mode='a', header=False, index=False, encoding="utf-8")

# --- Streamlit UI Header ---
st.title("Handwritten Text Recognizer & Evaluator")
st.write("Crop a single line of handwritten text to evaluate TrOCR model accuracy line-by-line.")

uploaded_file = st.file_uploader("Upload Notebook Line Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    from streamlit_cropper import st_cropper
    
    img = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("1. Select & Crop Single Text Line:")
    st.info("Drag the green cropper box directly around the single line of handwritten text you want to analyze.")
    
    cropped_img = st_cropper(
        img, 
        realtime_update=True, 
        box_color='#00FF00', 
        aspect_ratio=None
    )
    
    crop_bgr = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)

    # CLAHE contrast enhancement
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    crop_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    line_crop_pil = Image.fromarray(crop_rgb)

    st.subheader("2. Line Alignment Preview:")
    st.image(
        line_crop_pil, 
        caption="Selected Single Line Crop Strip (CLAHE Enhanced)", 
        use_container_width=True
    )

    st.subheader("3. Ground Truth Setup & Logging:")
    current_model_tag = st.text_input("Model Identifier Tag:", value="TrOCR-Base-SingleLine")
    
    ground_truth_input = st.text_input(
        "Ground Truth Reference Text (Exact text in image):", 
        value="Hello, My name is Komal Rawat.",
        placeholder="Type the exact text written in the cropped line image..."
    )

    if st.button("Digitize & Log Benchmark"):
        with st.spinner("Processing single line through cached TrOCR model..."):
            import torch
            
            processor, model = load_trocr()

            with torch.inference_mode():
                pixel_values = processor(images=line_crop_pil, return_tensors="pt").pixel_values
                
                generated_ids = model.generate(
                    pixel_values, 
                    max_new_tokens=35, 
                    num_beams=1
                )
                
                raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                reconstructed_text = clean_recognized_text(raw_text)

            st.subheader("Digitized Output Text:")
            st.code(reconstructed_text, language="text")

            # Save Metrics Routine
            if ground_truth_input.strip():
                try:
                    cer_metric, wer_metric = load_metrics()

                    ref_str = ground_truth_input.strip()
                    pred_str = reconstructed_text.strip()

                    cer_score = cer_metric.compute(predictions=[pred_str], references=[ref_str])
                    wer_score = wer_metric.compute(predictions=[pred_str], references=[ref_str])
                    
                    char_accuracy = max(0.0, (1.0 - cer_score) * 100)
                    word_accuracy = max(0.0, (1.0 - wer_score) * 100)

                    save_benchmark_to_file(
                        model_name=current_model_tag,
                        cer=cer_score,
                        wer=wer_score,
                        char_acc=char_accuracy,
                        word_acc=word_accuracy,
                        pred_text=pred_str,
                        ref_text=ref_str
                    )
                    st.success("Line digitized and metrics logged to `model_benchmark_results.json` and `.csv`!")
                except Exception as e:
                    st.error(f"Failed to log metrics: {e}")
            else:
                st.warning("Please enter the Ground Truth Reference Text above to log CER/WER metrics.")