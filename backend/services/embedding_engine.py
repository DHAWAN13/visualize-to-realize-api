from sentence_transformers import SentenceTransformer, util
import torch

# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# =========================
# CONCEPT SPACE
# =========================
CONCEPTS = [
    "cat",
    "dog",
    "chair",
    "table",
    "tree",
    "car",
    "bus",
    "airplane",
    "bicycle",
    "building",
    "person",
    "flower"
]

# Precompute embeddings
concept_embeddings = model.encode(CONCEPTS, convert_to_tensor=True)


# =========================
# MAIN FUNCTION
# =========================
def get_concept_from_text(text):
    if not text or len(text.strip()) == 0:
        return "unknown"

    text_embedding = model.encode(text, convert_to_tensor=True)

    scores = util.cos_sim(text_embedding, concept_embeddings)[0]

    best_idx = torch.argmax(scores).item()

    return CONCEPTS[best_idx]