import cv2
import numpy as np
from .config import GateRejected


def check_quality(matrix, image, settings):
    """Validate 3D head pose, brightness, and Laplacian sharpness on the target face crop."""
    # Remove scale from the facial transformation matrix before Euler decomposition.
    u, _, vt = np.linalg.svd(np.asarray(matrix)[:3, :3])
    rigid = u @ vt
    if np.linalg.det(rigid) < 0:
        raise GateRejected("GATE_UNAVAILABLE", "reflected transformation matrix")
    pitch, yaw, roll = cv2.RQDecomp3x3(rigid)[0]
    pose = {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if abs(yaw) > settings.max_yaw or abs(pitch) > settings.max_pitch:
        raise GateRejected("POSE_TOO_LARGE", str(pose))
    if brightness < settings.min_brightness or brightness > settings.max_brightness:
        raise GateRejected("POOR_LIGHTING", f"mean={brightness:.2f}")
    if blur < settings.min_blur:
        raise GateRejected("IMAGE_TOO_BLURRY", f"laplacian variance={blur:.2f}")
    return pose, blur, brightness
