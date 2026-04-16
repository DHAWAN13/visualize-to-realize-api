from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# =========================
# 🔥 DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Kosmos-2 using device:", device)

# =========================
# 🔥 LOAD MODEL
# =========================
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

model = AutoModelForVision2Seq.from_pretrained(
    "microsoft/kosmos-2-patch14-224"
).to(device)

model.eval()


# =========================
# 🔥 GET RAW OUTPUT
# =========================
def get_kosmos2_output(image_path):
    image = Image.open(image_path).convert("RGB")

    prompt = "<grounding> Describe this image."

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50
        )

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return output