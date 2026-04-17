import cv2
import numpy as np

def resize_image(image, target_width=1200):
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (target_width, new_h), interpolation=interp)

def to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def correct_illumination(gray):
    bg = cv2.GaussianBlur(gray, (101, 101), 0).astype(np.float32)
    corrected = (gray.astype(np.float32) / (bg + 1e-6)) * 128
    return np.clip(corrected, 0, 255).astype(np.uint8)

def remove_noise(image):
    nlm = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
    return cv2.medianBlur(nlm, 3)

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(image)

def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2)
    return cv2.addWeighted(image, 1.6, blurred, -0.6, 0)

def binarize(image):
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )

def morphological_clean(binary):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)

def deskew(image):
    inv = cv2.bitwise_not(image)
    edges = cv2.Canny(inv, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return image
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            if -15 < angle < 15:
                angles.append(angle)
    if not angles:
        return image
    median_angle = np.median(angles)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def remove_border(image, border_fraction=0.01):
    h, w = image.shape[:2]
    bh = max(1, int(h * border_fraction))
    bw = max(1, int(w * border_fraction))
    return image[bh:h-bh, bw:w-bw]

def full_pipeline(image: np.ndarray, target_width=1200) -> np.ndarray:
    steps = [
        lambda x: resize_image(x, target_width),
        to_grayscale,
        correct_illumination,
        remove_noise,
        enhance_contrast,
        sharpen_image,
        binarize,
        morphological_clean,
        deskew,
        remove_border,
    ]
    result = image
    for step in steps:
        result = step(result)
    return result