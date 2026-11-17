from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass
class NormalizedImage:
    bgr: np.ndarray
    mapping: dict


def decode_image(raw, settings):
    try:
        with Image.open(BytesIO(raw)) as source:
            if source.width * source.height > settings.max_image_pixels:
                raise ValueError("decoded pixel limit exceeded")
            original_size = source.size
            orientation = source.getexif().get(274, 1)
            image = ImageOps.exif_transpose(source).convert("RGB")
            oriented_size = image.size
            if max(image.size) > settings.max_long_side:
                image.thumbnail((settings.max_long_side, settings.max_long_side), Image.Resampling.LANCZOS)
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("invalid image") from error
    return NormalizedImage(bgr, {"original_size": original_size, "orientation": orientation,
                               "oriented_size": oriented_size, "normalized_size": image.size,
                               "scale_x": image.width / oriented_size[0],
                               "scale_y": image.height / oriented_size[1]})


def jpeg(image):
    ok, result = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return result.tobytes()


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def nms(boxes, threshold=0.4):
    result = []
    for candidate in sorted(boxes, key=lambda box: box["score"], reverse=True):
        if all(iou(candidate["bbox_px"], other["bbox_px"]) < threshold for other in result):
            result.append(candidate)
    return result


def match_candidates(mp_boxes, yn_boxes, width, height, threshold=0.3, min_side=80):
    mp_boxes, yn_boxes = nms(mp_boxes), nms(yn_boxes)
    pairs = sorted(((iou(a["bbox_px"], b["bbox_px"]), i, j)
                    for i, a in enumerate(mp_boxes) for j, b in enumerate(yn_boxes)), reverse=True)
    used_mp, used_yn, result = set(), set(), []
    for overlap, i, j in pairs:
        if overlap < threshold or i in used_mp or j in used_yn:
            continue
        used_mp.add(i)
        used_yn.add(j)
        # Use union to preserve chin/forehead where detectors have different box conventions.
        a, b = mp_boxes[i]["bbox_px"], yn_boxes[j]["bbox_px"]
        box = [max(0, int(min(a[0], b[0]))), max(0, int(min(a[1], b[1]))),
               min(width, int(max(a[2], b[2]))), min(height, int(max(a[3], b[3])))]
        if min(box[2]-box[0], box[3]-box[1]) < min_side:
            continue
        result.append({"bbox_px": box, "bbox_norm": [box[k]/(width if k%2==0 else height) for k in range(4)],
                       "score": float(min(mp_boxes[i]["score"], yn_boxes[j]["score"]))})
    result.sort(key=lambda box: (box["bbox_px"][0], box["bbox_px"][1]))
    return [{"face_id": f"face_{i}", **box} for i, box in enumerate(result)]


def crop(image, box, padding=0.12):
    x1, y1, x2, y2 = box
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    height, width = image.shape[:2]
    bounds = [max(0, int(x1-dx)), max(0, int(y1-dy)), min(width, int(x2+dx)), min(height, int(y2+dy))]
    return image[bounds[1]:bounds[3], bounds[0]:bounds[2]].copy(), bounds
