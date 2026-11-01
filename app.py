import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Handwritten Text Recognizer & Digitizer", layout="wide")

@st.cache_resource
def load_trocr():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    model.eval()
    return processor, model

@st.cache_resource
def load_simple_cnn():
    import torch
    from models.cnn_models import SimpleCNN
    model = SimpleCNN(num_classes=62)
    weights_path = "simple_cnn_emnist.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_residual_cnn():
    import torch
    from models.cnn_models import ResidualCNN
    model = ResidualCNN(num_classes=62)
    weights_path = "resnet_emnist.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_metrics():
    import evaluate
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    return cer_metric, wer_metric

def save_benchmark_log(log_entries):
    json_file = "model_benchmark_results.json"
    data = []
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
            
    data.extend(log_entries)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    csv_file = "model_benchmark_results.csv"
    df_new = pd.DataFrame(log_entries)
    if not os.path.exists(csv_file):
        df_new.to_csv(csv_file, index=False, encoding="utf-8")
    else:
        df_new.to_csv(csv_file, mode='a', header=False, index=False, encoding="utf-8")

def recognize_characters_emnist(line_crop_bgr, model):
    import torch
    from torchvision import transforms
    from emnist_mapping import map_label

    gray = cv2.cvtColor(line_crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_boxes = []
    h_img, w_img = thresh.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < w_img * 0.4 and h < h_img * 0.8 and w >= 5 and h >= 12:
            char_boxes.append((x, y, w, h))

    if not char_boxes:
        return ""

    char_boxes = sorted(char_boxes, key=lambda b: b[0])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    recognized_chars = []
    with torch.inference_mode():
        for x, y, w, h in char_boxes:
            pad = 2
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            char_crop = thresh[y1:y2, x1:x2]

            char_resized = cv2.resize(char_crop, (28, 28), interpolation=cv2.INTER_AREA)
            char_transposed = np.transpose(char_resized, (1, 0))

            tensor_img = transform(char_transposed).unsqueeze(0)
            output = model(tensor_img)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_char = map_label(pred_idx)
            recognized_chars.append(pred_char)

    return "".join(recognized_chars)


# --- UI Setup ---
st.title("Handwritten Text Recognizer and Digitizer")
st.write("Fast real-time handwritten line digitization engine.")

uploaded_file = st.file_uploader("Upload Sample Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    from streamlit_cropper import st_cropper
    
    img = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("1. Crop Line Region:")
    cropped_img = st_cropper(img, realtime_update=True, box_color='#00FF00', aspect_ratio=None)
    crop_bgr = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)

    st.subheader("2. Settings:")
    run_full_benchmark = st.checkbox("Run full multi-model background benchmark (Simple CNN & ResNet)", value=False)
    ground_truth_input = st.text_input("Ground Truth Reference Text (Optional for accuracy logging):", value="Hello World")

    if st.button("Digitize Text"):
        with st.spinner("Digitizing..."):
            import torch

            # Fast TrOCR Inference
            line_crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            processor, trocr_model = load_trocr()

            with torch.inference_mode():
                pixel_values = processor(images=line_crop_pil, return_tensors="pt").pixel_values
                generated_ids = trocr_model.generate(pixel_values, max_new_tokens=30, num_beams=1)
                raw_trocr = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                digitized_text = raw_trocr.strip()

            st.subheader("Digitized Output Text:")
            st.code(digitized_text, language="text")

            # Background Logging only if enabled or ground truth present
            if run_full_benchmark and ground_truth_input.strip():
                try:
                    cer_metric, wer_metric = load_metrics()
                    ref_str = ground_truth_input.strip()
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    cer_trocr = cer_metric.compute(predictions=[digitized_text], references=[ref_str])
                    wer_trocr = wer_metric.compute(predictions=[digitized_text], references=[ref_str])

                    simple_model = load_simple_cnn()
                    text_simple = recognize_characters_emnist(crop_bgr, simple_model)
                    cer_simple = cer_metric.compute(predictions=[text_simple if text_simple else " "], references=[ref_str])
                    wer_simple = wer_metric.compute(predictions=[text_simple if text_simple else " "], references=[ref_str])

                    resnet_model = load_residual_cnn()
                    text_resnet = recognize_characters_emnist(crop_bgr, resnet_model)
                    cer_resnet = cer_metric.compute(predictions=[text_resnet if text_resnet else " "], references=[ref_str])
                    wer_resnet = wer_metric.compute(predictions=[text_resnet if text_resnet else " "], references=[ref_str])

                    log_entries = [
                        {
                            "timestamp": timestamp_str,
                            "model_name": "Simple CNN (EMNIST Baseline)",
                            "cer": round(float(cer_simple), 4),
                            "wer": round(float(wer_simple), 4),
                            "char_accuracy_pct": round(max(0.0, (1.0 - cer_simple) * 100), 2),
                            "predicted_text": text_simple if text_simple else "[No Chars Detected]",
                            "ground_truth_text": ref_str
                        },
                        {
                            "timestamp": timestamp_str,
                            "model_name": "Residual CNN (EMNIST Architecture)",
                            "cer": round(float(cer_resnet), 4),
                            "wer": round(float(wer_resnet), 4),
                            "char_accuracy_pct": round(max(0.0, (1.0 - cer_resnet) * 100), 2),
                            "predicted_text": text_resnet if text_resnet else "[No Chars Detected]",
                            "ground_truth_text": ref_str
                        },
                        {
                            "timestamp": timestamp_str,
                            "model_name": "TrOCR (Vision Transformer)",
                            "cer": round(float(cer_trocr), 4),
                            "wer": round(float(wer_trocr), 4),
                            "char_accuracy_pct": round(max(0.0, (1.0 - cer_trocr) * 100), 2),
                            "predicted_text": digitized_text,
                            "ground_truth_text": ref_str
                        }
                    ]

                    save_benchmark_log(log_entries)
                    st.success("Benchmark metrics logged to `model_benchmark_results.csv`!")
                except Exception as e:
                    st.warning(f"Failed logging metrics: {e}")