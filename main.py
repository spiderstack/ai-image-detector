from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load model and processor
processor = AutoImageProcessor.from_pretrained("haywoodsloan/ai-image-detector-deploy")
model = AutoModelForImageClassification.from_pretrained("haywoodsloan/ai-image-detector-deploy")

# Function to classify an image
def detect_image(image_path):
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

# Example usage
if __name__ == "__main__":
    result = detect_image("path/to/your/file.jpg")
    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("Detailed scores:", result["scores"])
