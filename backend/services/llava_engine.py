import requests
import base64

OLLAMA_URL = "http://localhost:11434/api/generate"

# 🔥 Concept space
CONCEPTS = [
    "cat", "dog", "bird", "airplane", "bus",
    "car", "bicycle", "tree", "chair", "table",
    "person", "flower", "boat", "building"
]


def encode_image(image_path: str) -> str:
    """
    Convert image to base64 (required by Ollama)
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_output(text: str) -> str:
    """
    Extract valid concept from model output
    """
    if not text:
        return "unknown"

    text = text.lower().strip()

    # ✅ Exact match
    if text in CONCEPTS:
        return text

    # ✅ Find concept inside sentence
    for concept in CONCEPTS:
        if concept in text:
            return concept

    return "unknown"


def ask_llava(prompt: str, image_base64: str) -> str:
    """
    Send request to LLaVA
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llava",
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        },
        timeout=60
    )

    result = response.json()
    return result.get("response", "").strip()


def get_llava_concept(image_path: str) -> str:
    """
    FINAL: MCQ + Self-Correction
    """
    try:
        image_base64 = encode_image(image_path)

        options = "\n".join([f"- {c}" for c in CONCEPTS])

        # =========================
        # 🔥 PASS 1 (Strict MCQ)
        # =========================
        prompt1 = f"""
You are a strict image classifier.

Choose ONE object from the list.

Options:
{options}

Rules:
- Output ONLY ONE word
- Must be from the list
- No explanation

Answer:
"""

        output1 = ask_llava(prompt1, image_base64)
        concept1 = clean_output(output1)

        if concept1 != "unknown":
            return concept1

        # =========================
        # 🔥 PASS 2 (Guided Re-ask)
        # =========================
        prompt2 = f"""
Look carefully at the image again.

It contains ONE main object.

Choose the closest match from this list:

{options}

Hints:
- Has wings / flies → airplane
- Has ears / tail → cat or dog
- Has wheels → bus, car, bicycle
- Has leaves → tree
- Used for sitting → chair

Rules:
- MUST choose one option
- No explanation
- One word only

Answer:
"""

        output2 = ask_llava(prompt2, image_base64)
        concept2 = clean_output(output2)

        if concept2 != "unknown":
            return concept2

        # =========================
        # 🔥 PASS 3 (Forced Decision)
        # =========================
        prompt3 = f"""
You MUST choose ONE object.

Even if unsure, pick the closest.

Options:
{options}

Answer ONLY one word.
"""

        output3 = ask_llava(prompt3, image_base64)
        concept3 = clean_output(output3)

        return concept3 if concept3 else "unknown"

    except Exception as e:
        print("LLaVA Error:", str(e))
        return "unknown"