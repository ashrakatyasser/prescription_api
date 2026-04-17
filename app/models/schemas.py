from pydantic import BaseModel

class MedicineResult(BaseModel):
    ocr_text: str
    matched_drug: str | None
    match_score: float

class ScanResponse(BaseModel):
    medicines: list[MedicineResult]
    total_found: int