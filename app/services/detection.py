import cv2
import numpy as np
from ultralytics import YOLO
from app.models.config import CONF_THRESHOLD, IOU_THRESHOLD

def detect_medicines(model: YOLO, image: np.ndarray, padding=6):
    # احفظ الصورة مؤقتاً عشان YOLO يقراها
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    H, W = image.shape[:2]
    results = model.predict(source=tmp_path, conf=CONF_THRESHOLD,
                            iou=IOU_THRESHOLD, verbose=False)
    os.unlink(tmp_path)

    detections = []
    crops = []

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(W, x2 + padding)
        y2p = min(H, y2 + padding)

        crop = image[y1p:y2p, x1p:x2p]
        crops.append(crop)

        detections.append({
            "id": i + 1,
            "label": model.names[cls_id],
            "conf": conf,
        })

    return detections, crops