from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# =========================
# 🔥 DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("BLIP using device:", device)

# =========================
# 🔥 LOAD MODEL
# =========================
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)


def get_image_description(image_path):
    """
    Generate caption using BLIP
    """
    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)

    # 🔥 IMPROVED GENERATION (no garbage repetition)
    out = model.generate(
        **inputs,
        max_length=30,
        num_beams=5,
        repetition_penalty=1.2
    )

    description = processor.decode(out[0], skip_special_tokens=True)

    return description