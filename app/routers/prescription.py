import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.models.schemas import ScanResponse, MedicineResult
from app.services.preprocessing import full_pipeline
from app.services.detection import detect_medicines
from app.services.ocr import recognize_medicine
from app.services.matcher import match_ocr_text
from app.main import app_state

router = APIRouter(prefix="/api", tags=["prescription"])


@router.post("/scan", response_model=ScanResponse)
async def scan_prescription(file: UploadFile = File(...)):

    # ---------------- Validate input ----------------
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # ---------------- Read image ----------------
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---------------- Preprocessing ----------------
    preprocessed = full_pipeline(image)
    preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)

    # ---------------- Detection ----------------
    detections, crops = detect_medicines(
        app_state["yolo_model"],
        preprocessed_rgb
    )

    medicines = []

    for det, crop in zip(detections, crops):

        # ---------------- OCR ----------------
        ocr_text, ocr_conf = recognize_medicine(
            app_state["processor"],
            app_state["ocr_model"],
            crop,
            app_state["device"]
        )

        # ---------------- Matching (NEW) ----------------
        match, score = match_ocr_text(
            ocr_text,
            drug_list=app_state["drug_list"],
            drug_embeddings=app_state["drug_embeddings"],
            model=app_state["sentence_model"],
            device=app_state["device"]
        )

        # ---------------- Append Result ----------------
        medicines.append(
            MedicineResult(
                ocr_text=ocr_text,
                matched_drug=match,
                match_score=score,
                ocr_confidence=ocr_conf
            )
        )

        # ---------------- Debug ----------------
        print(f"OCR: {ocr_text}")
        print(f"Match: {match} | Score: {score:.2f}")
        print("-" * 40)

    # ---------------- Response ----------------
    return ScanResponse(
        medicines=medicines,
        total_found=len(medicines)
    )