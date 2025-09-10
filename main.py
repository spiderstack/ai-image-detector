from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import sys
import cv2
import numpy as np

# Load model and processor
processor = AutoImageProcessor.from_pretrained("haywoodsloan/ai-image-detector-deploy")
model = AutoModelForImageClassification.from_pretrained("haywoodsloan/ai-image-detector-deploy")

# Function to classify an image
def detect_image_info(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)[0]
    
    labels = model.config.id2label if hasattr(model.config, "id2label") else {0: "real", 1: "artificial"}
    scores = [(labels[i], float(probs[i])) for i in range(len(probs))]
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "prediction": scores[0][0],
        "confidence": f"{scores[0][1]*100:.2f}%",
        "scores": [f"{lbl}: {sc*100:.2f}%" for lbl, sc in scores]
    }
    
def detect_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)[0]
    return float(probs[0])
    
def detect_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)[0]
    return float(probs[0])
    
def detect_video(video_path, sample_rate=10):
    cap = cv2.VideoCapture(video_path)
    frame_idx, scores = 0, []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            scores.append(detect_frame(frame))
        frame_idx += 1
    cap.release()
    return float(np.mean(scores)) if scores else 0.0

# Example usage
if __name__ == "__main__":
    input_path = sys.argv[1]
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        result = detect_image(input_path)
    else:
        result = detect_video(input_path)
    print(result)
