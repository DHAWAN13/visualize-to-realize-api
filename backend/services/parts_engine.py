import json
import os

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to JSON
DATA_PATH = os.path.join(BASE_DIR, "data", "concept_parts.json")

# Load data
with open(DATA_PATH, "r") as f:
    concept_parts = json.load(f)


def get_parts_for_concept(concept: str):
    return concept_parts.get(concept, [])