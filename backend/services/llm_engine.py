import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"


# =========================
# 🔥 CONCEPT EXTRACTION (NEW)
# =========================
def extract_concept_llm(description):
    """
    Extract main object from image description using LLM
    """

    prompt = f"""
    Identify the main object in this image description.

    Description: {description}

    Rules:
    - Return ONLY the object name
    - One or two words only
    - No explanation

    Answer:
    """

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )

        result = response.json()
        content = result.get("response", "").strip().lower()

        # Clean output
        concept = content.replace(".", "").replace(",", "").strip()

        if concept == "":
            return "unknown"

        return concept

    except Exception as e:
        print("Concept extraction error:", e)
        return "unknown"


# =========================
# 🔥 EVALUATION (FIXED)
# =========================
def evaluate_with_llm(concept, description, visual_features):
    """
    LLM reasoning using Ollama (Mistral)
    """

    # 🔥 SAFE ACCESS (fixes your crash)
    contours = visual_features.get("num_contours", 0) if isinstance(visual_features, dict) else 0
    edges = visual_features.get("has_edges", False) if isinstance(visual_features, dict) else False

    prompt = f"""
    A student drew something.

    Detected concept: {concept}
    Description (may be noisy): {description}

    Visual features:
    - contours: {contours}
    - edges: {edges}

    IMPORTANT:
    - Do NOT fully trust description (it can be wrong)
    - Focus on concept + structure

    Evaluate:
    1. Shape correctness
    2. Missing parts
    3. Realism

    Return ONLY JSON:
    {{
        "score": (0-100),
        "feedback": [
            "issue",
            "improvement"
        ]
    }}
    """

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        result = response.json()
        content = result.get("response", "").strip()

        # 🔥 SAFE JSON EXTRACTION
        start = content.find("{")
        end = content.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("Invalid JSON from LLM")

        json_str = content[start:end]

        return json.loads(json_str)

    except Exception as e:
        print("LLM error:", e)

        return {
            "score": 50,
            "feedback": [f"LLM error: {str(e)}"]
        }