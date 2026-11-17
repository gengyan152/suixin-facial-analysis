import cv2
import numpy as np
from .config import GateRejected

# Names follow image left/right (subject right/left in an unmirrored image).
REGIONS = {
    "eyes": [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,
             263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466],
    "brows": [70,63,105,66,107,55,65,52,53,46,336,296,334,293,300,276,283,282,295,285],
    "nose": [168,6,197,195,5,4,1,2,98,97,327,326,129,358,48,278,64,294],
    "mouth": [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,
              78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191],
    "contour": [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,
                148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
}
SYMMETRY_PAIRS = [(33,263),(133,362),(159,386),(145,374),(52,282),(65,295),
                  (98,327),(61,291),(234,454),(172,397)]
LABELS = {"eyes":"眼部区域","brows":"眉部区域","nose":"鼻部区域","mouth":"嘴唇区域","contour":"脸部轮廓与下颌"}


def measure(points, matrix, image, settings):
    if points.shape != (478,2) or not np.isfinite(points).all():
        raise GateRejected("GATE_UNAVAILABLE", "invalid landmark array")
    if abs(cv2.contourArea(points[REGIONS["contour"]].astype(np.float32))) < 100:
        raise GateRejected("GATE_UNAVAILABLE", "degenerate landmark area")
    left_center, right_center = points[[33,133]].mean(0), points[[263,362]].mean(0)
    angle = np.arctan2(*(right_center-left_center)[::-1])
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    aligned = (points - (left_center+right_center)/2) @ rotation
    if not (aligned[33,0] < aligned[133,0] < aligned[362,0] < aligned[263,0] and aligned[61,0] < aligned[291,0]):
        raise GateRejected("GATE_UNAVAILABLE", "flipped eye/mouth order")
    # Remove scale from the facial transformation matrix before Euler decomposition.
    u,_,vt = np.linalg.svd(np.asarray(matrix)[:3,:3])
    rigid = u @ vt
    if np.linalg.det(rigid) < 0:
        raise GateRejected("GATE_UNAVAILABLE", "reflected transformation matrix")
    pitch,yaw,roll = cv2.RQDecomp3x3(rigid)[0]
    pose = {"yaw":float(yaw),"pitch":float(pitch),"roll":float(roll)}
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    brightness,blur = float(gray.mean()),float(cv2.Laplacian(gray,cv2.CV_64F).var())
    if abs(yaw)>settings.max_yaw or abs(pitch)>settings.max_pitch:
        raise GateRejected("POSE_TOO_LARGE",str(pose))
    if brightness < settings.min_brightness or brightness > settings.max_brightness:
        raise GateRejected("POOR_LIGHTING",f"mean={brightness:.2f}")
    if blur < settings.min_blur:
        raise GateRejected("IMAGE_TOO_BLURRY",f"laplacian variance={blur:.2f}")
    # Intersect oval edges with the common eye-center horizontal line, rather than
    # substituting cheek width from two vertices at different heights.
    oval = aligned[REGIONS["contour"]]
    intersections = []
    for a,b in zip(oval,np.roll(oval,-1,axis=0)):
        if (a[1] <= 0 < b[1]) or (b[1] <= 0 < a[1]):
            intersections.append(a[0] + (-a[1])*(b[0]-a[0])/(b[1]-a[1]))
    if len(intersections)<2:
        raise GateRejected("GATE_UNAVAILABLE", "no face-width intersection at eye line")
    face_width = max(intersections)-min(intersections)
    le = aligned[133,0]-aligned[33,0]
    re = aligned[263,0]-aligned[362,0]
    inner = aligned[362,0]-aligned[133,0]
    eye_height = (aligned[145,1]-aligned[159,1] + aligned[374,1]-aligned[386,1])/2
    left_height = aligned[145,1]-aligned[159,1]
    right_height = aligned[374,1]-aligned[386,1]
    mid = aligned[2,1]-aligned[9,1]
    lower = aligned[152,1]-aligned[2,1]
    upper_lip,lower_lip = aligned[13,1]-aligned[0,1], aligned[17,1]-aligned[14,1]
    if min(face_width, le,re,inner,left_height,right_height,lower,lower_lip,mid) <= 1e-3 or upper_lip < 0:
        raise GateRejected("GATE_UNAVAILABLE", "degenerate measurement denominator")
    brow_ratio = ((aligned[[33,133],1].mean()-aligned[[52,65],1].mean())/left_height +
                  (aligned[[263,362],1].mean()-aligned[[282,295],1].mean())/right_height)/2
    axis_points = aligned[[168,6,2,152]]
    axis_center = axis_points.mean(0)
    _,_,vectors = np.linalg.svd(axis_points-axis_center)
    direction = vectors[0]
    left = aligned[[p[0] for p in SYMMETRY_PAIRS]]
    right = aligned[[p[1] for p in SYMMETRY_PAIRS]]
    relative = left-axis_center
    reflected = axis_center + 2*np.outer(relative@direction,direction)-relative
    outer = float(np.linalg.norm(aligned[263]-aligned[33]))
    if outer <= 1e-3:
        raise GateRejected("GATE_UNAVAILABLE", "degenerate outer canthus span")
    deviation = float(np.linalg.norm(reflected-right,axis=1).mean() / outer)
    geometry = {"measurement_note":"基于转正像素平面并经双眼连线 Roll 水平对齐计算；首版测量中下庭比例（不含发际线上庭）。",
                "mid_to_lower_face_ratio": float(mid/lower),
                "five_eyes":{"left_eye_to_face_ratio":float(le/face_width),"intercanthal_to_face_ratio":float(inner/face_width),
                             "right_eye_to_face_ratio":float(re/face_width),"inner_canthus_to_eye_width_ratio":float(inner/((le+re)/2))},
                "brow_to_eye_ratio":float(brow_ratio), "nose_wing_to_intercanthal_ratio":float(abs(aligned[327,0]-aligned[98,0])/inner),
                "upper_to_lower_lip_ratio":float(upper_lip/lower_lip),"mirror_deviation_norm":deviation,
                "mirror_symmetry_score":max(0.,1-2*deviation)}
    return geometry,pose,blur,brightness


def overlay(image,points,connections):
    result = image.copy()
    for a,b in connections:
        cv2.line(result,tuple(points[a].astype(int)),tuple(points[b].astype(int)),(70,120,70),1,cv2.LINE_AA)
    for point in points:
        cv2.circle(result,tuple(point.astype(int)),1,(100,240,220),-1,cv2.LINE_AA)
    for i,(key,indices) in enumerate(REGIONS.items()):
        for index in indices:
            cv2.circle(result,tuple(points[index].astype(int)),2,[(255,160,50),(160,90,255),(50,220,250),(80,80,255),(220,220,80)][i],-1)
    axis = points[[168,6,2,152]]
    vx,vy,x,y = cv2.fitLine(axis.astype(np.float32),cv2.DIST_L2,0,.01,.01).ravel()
    radius = np.linalg.norm(points[152]-points[168])*.65
    cv2.line(result,(int(x-vx*radius),int(y-vy*radius)),(int(x+vx*radius),int(y+vy*radius)),(255,255,255),2,cv2.LINE_AA)
    return result


def explain_regions(image,points,numerical):
    h,w = image.shape[:2]
    occupied = np.zeros((h,w),np.uint8)
    masks = {}
    for key,indices in REGIONS.items():
        mask = np.zeros((h,w),np.uint8)
        if key in ("eyes","brows"):
            half = len(indices)//2
            groups = [indices[:half],indices[half:]]
        else:
            groups = [indices]
        for group in groups:
            cv2.fillConvexPoly(mask,cv2.convexHull(points[group].astype(np.int32)),255)
        if key=="contour":
            mask -= cv2.erode(mask,np.ones((max(3,int(min(h,w)*.07))|1,)*2,np.uint8))
        else:
            mask = cv2.dilate(mask,np.ones((5,5),np.uint8))
        mask[occupied>0] = 0
        occupied |= mask
        masks[key] = mask
    blurred = cv2.GaussianBlur(image,(31,31),9)
    face_hull = np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(face_hull,cv2.convexHull(points[REGIONS["contour"]].astype(np.int32)),255)
    images = [image]
    rng = np.random.default_rng(20261114)
    references = {}
    # Each region gets 3 translations of its exact binary mask constrained to the face hull:
    # equal binary area, deterministic seed, inside-face priority, and no invented p-values.
    for key,mask in masks.items():
        feather = cv2.GaussianBlur(mask.astype(np.float32)/255,(15,15),4)[...,None]
        images.append(np.clip(image*(1-feather)+blurred*feather,0,255).astype(np.uint8))
    for key,mask in masks.items():
        ys,xs = np.where(mask>0)
        area = max(int((mask>0).sum()), 1)
        min_dx, max_dx = -int(xs.min()), max(-int(xs.min())+1, w-int(xs.max()))
        min_dy, max_dy = -int(ys.min()), max(-int(ys.min())+1, h-int(ys.max()))
        pool = []
        seen_shifts = set()
        for _ in range(96):
            dx = int(rng.integers(min_dx, max_dx))
            dy = int(rng.integers(min_dy, max_dy))
            if (dx, dy) in seen_shifts or (dx == 0 and dy == 0):
                continue
            seen_shifts.add((dx, dy))
            shifted = cv2.warpAffine(mask,np.float32([[1,0,dx],[0,1,dy]]),(w,h),flags=cv2.INTER_NEAREST)
            in_face = float(((shifted>0) & (face_hull>0)).sum()) / area
            self_overlap = float(((shifted>0) & (mask>0)).sum()) / area
            pool.append((in_face >= 0.999, round(in_face, 4), -round(self_overlap, 4), dx, dy, shifted))
        pool.sort(key=lambda item: item[:3], reverse=True)
        chosen = pool[:3] if len(pool) >= 3 else (pool + [(True,1.,0.,0,0,mask)]*3)[:3]
        indices=[]
        for *_, shifted in chosen:
            feather = cv2.GaussianBlur(shifted.astype(np.float32)/255,(15,15),4)[...,None]
            indices.append(len(images))
            images.append(np.clip(image*(1-feather)+blurred*feather,0,255).astype(np.uint8))
        references[key]=indices
    scores = numerical.beauty_batch(images)
    baseline = float(np.abs(scores[6:]-scores[0]).mean())
    regions=[]
    for i,key in enumerate(masks,1):
        delta = float(scores[0]-scores[i])
        relative = abs(delta)-float(np.abs(scores[references[key]]-scores[0]).mean())
        regions.append({"region_key":key,"region_name":LABELS[key],"perturbed_score":float(scores[i]),
                        "delta_score":delta,"abs_delta":abs(delta),"relative_to_random":relative,
                        "is_sensitive":relative>.05})
    return baseline,regions
