import cv2
import numpy as np

def _green_mask_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """
    Creates a mask for green-ish regions in the image.
    Tuned to work across many lighting conditions.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Two ranges to catch different greens (you can tweak if needed)
    lower1 = np.array([25, 30, 30], dtype=np.uint8)
    upper1 = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def detect_and_crop_leaf(
    img_bgr: np.ndarray,
    min_green_ratio: float = 0.08,
    min_area_ratio: float = 0.03,
    pad: float = 0.08,
):
    """
    Returns (ok, cropped_img, info_dict)
    ok=False means "not a leaf image" (reject)
    """
    h, w = img_bgr.shape[:2]
    if h < 40 or w < 40:
        return False, None, {"reason": "Image too small."}

    mask = _green_mask_hsv(img_bgr)

    green_ratio = float(mask.mean() / 255.0)
    if green_ratio < min_green_ratio:
        return False, None, {
            "reason": "Not enough green pixels (likely not a leaf).",
            "green_ratio": green_ratio,
        }

    # Find largest green contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, {"reason": "No green region found."}

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    img_area = float(h * w)
    area_ratio = area / (img_area + 1e-9)

    if area_ratio < min_area_ratio:
        return False, None, {
            "reason": "Green region too small (likely background/no leaf).",
            "area_ratio": area_ratio,
        }

    # Shape sanity checks (optional but helps):
    # solidity = contour_area / hull_area; leaves are often fairly solid blobs
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull)) + 1e-9
    solidity = area / hull_area

    if solidity < 0.35:
        return False, None, {
            "reason": "Green region shape doesn't look leaf-like (low solidity).",
            "solidity": solidity,
        }

    x, y, bw, bh = cv2.boundingRect(cnt)

    # Pad crop
    px = int(bw * pad)
    py = int(bh * pad)

    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(w, x + bw + px)
    y1 = min(h, y + bh + py)

    crop = img_bgr[y0:y1, x0:x1].copy()
    return True, crop, {
        "green_ratio": green_ratio,
        "area_ratio": area_ratio,
        "solidity": solidity,
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
    }
