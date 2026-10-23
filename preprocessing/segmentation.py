import cv2
import numpy as np

def preprocess_and_segment(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Binarize: ensure text is pure white (255) and background is black (0)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Disconnect attached letters using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    segmented_images = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Strictly restrict bounding box width & height to single character sizes
        if 8 < w < 40 and 12 < h < 50:
            bounding_boxes.append((x, y, w, h))

    # Sort boxes top-to-bottom, left-to-right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1] // 25, b[0]))

    for x, y, w, h in bounding_boxes:
        roi = thresh[y:y+h, x:x+w]

        # Make ROI square
        pad = max(w, h) + 4
        square = np.zeros((pad, pad), dtype=np.uint8)
        dx = (pad - w) // 2
        dy = (pad - h) // 2
        square[dy:dy+h, dx:dx+w] = roi

        # Resize to 28x28
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

        # Fix EMNIST orientation (rotate 90 deg counter-clockwise and flip)
        aligned = cv2.flip(cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)

        # Normalize values to [0, 1]
        normalized = aligned.astype('float32') / 255.0
        segmented_images.append(normalized)

    return thresh, bounding_boxes, np.array(segmented_images)