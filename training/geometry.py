"""Pixel-plane, roll-aligned MediaPipe geometry (schema scut-geometry-v1)."""
import numpy as np

VERSION = 'scut-geometry-v1'
LANDMARKS = {
    'image_left_eye_outer_inner': [33, 133], 'image_right_eye_inner_outer': [362, 263],
    'eye_upper_lower': [[159, 145], [386, 374]], 'brow_lower': [[52, 65], [282, 295]],
    'glabella': 9, 'subnasale_proxy': 2, 'chin': 152, 'nose_wings': [98, 327],
    'upper_lip_outer_inner': [0, 13], 'lower_lip_inner_outer': [14, 17],
    'mouth_corners': [61, 291], 'axis': [168, 6, 2, 152],
    'symmetry_pairs': [[33, 263], [133, 362], [159, 386], [145, 374], [52, 282],
                       [65, 295], [98, 327], [61, 291], [234, 454], [172, 397]],
    'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                  365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
                  132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
}


def measure_geometry(normalized, width, height):
    points = np.asarray(normalized, dtype=np.float64)
    if points.shape not in ((478, 2), (478, 3)) or not np.isfinite(points).all() or min(width, height) <= 0:
        raise ValueError('Expected finite 478 normalized landmarks and positive image dimensions')
    points = points[:, :2]*[width, height]
    left = points[[33, 133]].mean(0)
    right = points[[362, 263]].mean(0)
    delta = right-left
    if delta[0] <= 0 or np.linalg.norm(delta) < 2:
        raise ValueError('Eye point order reversed or degenerate eye centers')
    angle = np.arctan2(delta[1], delta[0])
    c, s = np.cos(angle), np.sin(angle)
    points = (points-(left+right)/2) @ np.array([[c, -s], [s, c]])
    # Intersect the full face oval with the horizontal line through both eye centers.
    contour = points[LANDMARKS['face_oval']]
    intersections = []
    for a, b in zip(contour, np.roll(contour, -1, axis=0)):
        if (a[1] <= 0 < b[1]) or (b[1] <= 0 < a[1]):
            intersections.append(float(a[0] + (b[0]-a[0])*(-a[1])/(b[1]-a[1])))
    if len(intersections) < 2:
        raise ValueError('Face contour does not cross eye level twice')
    face_width = max(intersections)-min(intersections)
    left_width = points[133, 0]-points[33, 0]
    right_width = points[263, 0]-points[362, 0]
    inner = points[362, 0]-points[133, 0]
    outer = np.linalg.norm(points[263]-points[33])
    left_height = points[145, 1]-points[159, 1]
    right_height = points[374, 1]-points[386, 1]
    middle = points[2, 1]-points[9, 1]
    lower = points[152, 1]-points[2, 1]
    upper_lip = points[13, 1]-points[0, 1]
    lower_lip = points[17, 1]-points[14, 1]
    if min(face_width, left_width, right_width, inner, outer, left_height, right_height, middle, lower, upper_lip, lower_lip) <= 1e-6:
        raise ValueError('Degenerate or reversed geometry denominator')
    axis_points = points[LANDMARKS['axis']]
    center = axis_points.mean(0)
    _, _, vectors = np.linalg.svd(axis_points-center, full_matrices=False)
    direction = vectors[0]
    pairs = np.asarray(LANDMARKS['symmetry_pairs'])
    displacement = points[pairs[:, 0]]-center
    reflected = center + 2*np.outer(displacement@direction, direction)-displacement
    symmetry = np.linalg.norm(reflected-points[pairs[:, 1]], axis=1).mean()/outer
    return {
        'left_eye_width_face': float(left_width/face_width),
        'inner_canthal_face': float(inner/face_width),
        'right_eye_width_face': float(right_width/face_width),
        'left_temple_face': float((points[33, 0]-min(intersections))/face_width),
        'right_temple_face': float((max(intersections)-points[263, 0])/face_width),
        'left_brow_eye_ratio': float((points[[33, 133], 1].mean()-points[[52, 65], 1].mean())/left_height),
        'right_brow_eye_ratio': float((points[[362, 263], 1].mean()-points[[282, 295], 1].mean())/right_height),
        'middle_lower_ratio': float(middle/lower),
        'nose_inner_canthal_ratio': float(abs(points[327, 0]-points[98, 0])/inner),
        'upper_lower_lip_ratio': float(upper_lip/lower_lip),
        'mirror_asymmetry': float(symmetry),
    }
