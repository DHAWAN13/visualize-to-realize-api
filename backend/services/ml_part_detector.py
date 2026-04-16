import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os

# 🔥 Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "parts_model.pth")

# 🔥 Classes (must match training CSV)
PART_CLASSES = [
    "head", "ears", "eyes", "tail", "legs",
    "trunk", "branches", "leaves", "roots",
    "wheels", "windows", "doors", "body"
]

# 🔥 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(PART_CLASSES))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 🔥 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def detect_parts_ml(image_path: str, concept: str):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).squeeze().cpu()

    raw_parts = []

    # 🔥 Lower threshold for better sensitivity
    for i, p in enumerate(probs):
        if p.item() > 0.3:
            raw_parts.append(PART_CLASSES[i])

    # 🔥 Concept-aware filtering
    concept_map = {
        "cat": ["head", "ears", "eyes", "tail", "legs"],
        "dog": ["head", "ears", "eyes", "tail", "legs"],
        "tree": ["trunk", "branches", "leaves", "roots"],
        "car": ["wheels", "windows", "doors", "body"]
    }

    if concept in concept_map:
        filtered_parts = [p for p in raw_parts if p in concept_map[concept]]

        # 🔥 Important: avoid wrong predictions
        if len(filtered_parts) == 0:
            return []

        return list(set(filtered_parts))

    return list(set(raw_parts))