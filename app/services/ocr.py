import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.nn.functional import softmax
from app.models.config import NUM_BEAMS

def preprocess_for_recognition(image: np.ndarray, target_size=(384, 96)):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    tw, th = target_size
    h, w = image.shape[:2]
    scale = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    canvas = np.ones((th, tw, 3), dtype=np.uint8) * 255
    y_off = (th - nh) // 2
    x_off = (tw - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized

    lab = cv2.cvtColor(canvas, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    pil_img = Image.fromarray(enhanced)
    return ImageEnhance.Sharpness(pil_img).enhance(1.8)

def recognize_medicine(processor, model, image: np.ndarray, device: str):
    pil_img = preprocess_for_recognition(image)
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            num_beams=NUM_BEAMS,
            max_new_tokens=64,
            output_scores=True,
            return_dict_in_generate=True,
        )

    token_probs = []
    for step_scores in outputs.scores:
        prob = softmax(step_scores[0], dim=-1).max()
        token_probs.append(prob.item())

    import numpy as np
    confidence = float(np.exp(np.mean(np.log(np.clip(token_probs, 1e-9, 1.0)))))
    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    return text, confidence