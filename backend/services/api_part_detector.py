import requests
import time

# 🔥 Replace with your token
HEADERS = {
    "Authorization": "Bearer hf_your_token_here"
}

API_URL = "https://api-inference.huggingface.co/models/IDEA-Research/grounding-dino-base"


def query(image_path, text, retries=3):
    for attempt in range(retries):
        try:
            with open(image_path, "rb") as f:
                data = f.read()

            response = requests.post(
                API_URL,
                headers=HEADERS,
                files={"file": data},
                data={"inputs": text},
                timeout=30
            )

            return response.json()

        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)
            time.sleep(2)

    return []


def detect_parts_api(image_path, concept):
    concept_parts = {
        "cat": ["head", "ears", "eyes", "tail", "legs"],
        "dog": ["head", "ears", "eyes", "tail", "legs"],
        "tree": ["trunk", "branches", "leaves", "roots"],
        "car": ["wheels", "windows", "doors", "body"]
    }

    parts = concept_parts.get(concept, [])
    detected = []

    for part in parts:
        prompt = f"{concept} {part}"
        result = query(image_path, prompt)

        # ✅ Check properly
        if isinstance(result, list) and len(result) > 0:
            detected.append(part)

    return list(set(detected))