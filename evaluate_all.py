import cv2
import numpy as np
import torch
import pandas as pd
import os
import json
from datetime import datetime
from PIL import Image

from models.cnn_models import SimpleCNN, ResidualCNN
from emnist_mapping import map_label
import evaluate

def evaluate_models_on_image(image_path, ground_truth_text):
    print(f"Evaluating models on: {image_path}")
    
    # Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Image file not found.")
        return

    # Load Metrics
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    ref_str = ground_truth_text.strip()

    # --- 1. Simple CNN ---
    simple_model = SimpleCNN(num_classes=62)
    if os.path.exists("simple_cnn_emnist.pth"):
        simple_model.load_state_dict(torch.load("simple_cnn_emnist.pth", map_location='cpu'))
    simple_model.eval()

    # --- 2. Residual CNN ---
    resnet_model = ResidualCNN(num_classes=62)
    if os.path.exists("resnet_emnist.pth"):
        resnet_model.load_state_dict(torch.load("resnet_emnist.pth", map_location='cpu'))
    resnet_model.eval()

    # Segment and Predict for CNNs
    def run_cnn(model):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        chars = []
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
                chars.append(map_label(pred_idx))
        return "".join(chars)

    text_simple = run_cnn(simple_model)
    text_resnet = run_cnn(resnet_model)

    # --- 3. TrOCR ---
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    trocr_model.eval()

    pil_img = Image.open(image_path).convert("RGB")
    with torch.inference_mode():
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=30, num_beams=1)
        text_trocr = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Compute CER / WER
    cer_s = cer_metric.compute(predictions=[text_simple if text_simple else " "], references=[ref_str])
    cer_r = cer_metric.compute(predictions=[text_resnet if text_resnet else " "], references=[ref_str])
    cer_t = cer_metric.compute(predictions=[text_trocr], references=[ref_str])

    results = [
        {"model_name": "Simple CNN (EMNIST Baseline)", "cer": round(float(cer_s), 4), "char_accuracy_pct": round(max(0.0, (1.0 - cer_s) * 100), 2), "predicted_text": text_simple, "ground_truth_text": ref_str},
        {"model_name": "Residual CNN (EMNIST Architecture)", "cer": round(float(cer_r), 4), "char_accuracy_pct": round(max(0.0, (1.0 - cer_r) * 100), 2), "predicted_text": text_resnet, "ground_truth_text": ref_str},
        {"model_name": "TrOCR (Vision Transformer)", "cer": round(float(cer_t), 4), "char_accuracy_pct": round(max(0.0, (1.0 - cer_t) * 100), 2), "predicted_text": text_trocr, "ground_truth_text": ref_str}
    ]

    df = pd.DataFrame(results)
    df.to_csv("model_benchmark_results.csv", index=False)
    print("\nBenchmark results saved successfully to 'model_benchmark_results.csv'!")
    print(df[["model_name", "char_accuracy_pct", "cer", "predicted_text"]])

if __name__ == '__main__':
    # Replace with a path to one of your test line images and its exact text
    evaluate_models_on_image("img.png", "ABCD")