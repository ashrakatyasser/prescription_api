import os

os.environ["HF_HOME"] = "models_cache"

import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer

from app.services.matcher import load_drug_list
from app.models.config import MODEL_PATH, TROCR_PATH, SENTENCE_MODEL_PATH


app_state = {}
print("TROCR_PATH =", TROCR_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    app_state["device"] = device

    # ---------------- YOLO ----------------
    app_state["yolo_model"] = YOLO(MODEL_PATH)

    # ---------------- TrOCR ----------------
    app_state["processor"] = TrOCRProcessor.from_pretrained(TROCR_PATH)

    app_state["ocr_model"] = VisionEncoderDecoderModel.from_pretrained(
        TROCR_PATH
    ).to(device)

    app_state["ocr_model"].eval()

    # ---------------- Sentence Transformer ----------------
    sentence_model = SentenceTransformer(
        SENTENCE_MODEL_PATH,
        device=device
    )

    app_state["sentence_model"] = sentence_model

    # ---------------- Drug List + Embeddings ----------------
    drug_list, drug_embeddings = load_drug_list(sentence_model, device)

    app_state["drug_list"] = drug_list
    app_state["drug_embeddings"] = drug_embeddings

    print("✅ All models loaded!")

    yield

    # cleanup
    app_state.clear()


# ---------------- FastAPI App ----------------
app = FastAPI(
    title="Prescription Scanner API",
    lifespan=lifespan
)

from app.routers import prescription
app.include_router(prescription.router)


@app.get("/health")
def health():
    return {"status": "ok"}