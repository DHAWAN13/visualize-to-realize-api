from fastapi import FastAPI, File, UploadFile
import shutil
import os

from services.llava_engine import get_llava_concept
from services.visual_engine import extract_visual_features
from services.knowledge_engine import get_expected_parts
from services.llm_engine import evaluate_with_llm

app = FastAPI()

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.get("/")
def home():
    return {"message": "Backend running 🚀"}


@app.post("/upload")
def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # SAVE FILE
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 STEP 1: LLaVA CONCEPT
    concept = get_llava_concept(file_path)

    # 🔥 STEP 2: VISUAL FEATURES
    visual_features = extract_visual_features(file_path)

    # 🔥 STEP 3: EXPECTED PARTS
    expected_parts = get_expected_parts(concept)

    # 🔥 STEP 4: LLM FEEDBACK
    llm_result = evaluate_with_llm(
        concept,
        f"This is a {concept}",
        visual_features
    )

    return {
        "filename": file.filename,
        "concept": concept,
        "expected_parts": expected_parts,
        "visual_features": visual_features,
        "score": llm_result.get("score", 50),
        "feedback": llm_result.get("feedback", ["No feedback"])
    }