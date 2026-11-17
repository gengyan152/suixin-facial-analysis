from pathlib import Path
import json
import time

import cv2
import numpy as np
import torch
from .config import GateRejected
from .imaging import crop,match_candidates,jpeg,iou
from .geometry import measure,overlay,explain_regions
from .numeric import NumericalModels
from .facellm import FaceLLM,FIELD_LABELS


class Pipeline:
    def __init__(self,settings):
        self.settings=settings
        self.models={
            "gate_detectors":{"loaded":False,"version":"mediapipe_face_detector + opencv_yunet"},
            "landmarks":{"loaded":False,"version":"mediapipe_face_landmarker_478"},
            "facellm":{"loaded":False,"version":"Idiap/FaceLLM-1B","precision":"bf16"},
            "beauty_2dfap":{"loaded":False,"version":"2d_fap_mobilenetv2","weights_status":"untrained_init","weights_ready":False},
            "apparent_age":{"loaded":False,"version":"se_resnext50_appa_real","weights_status":"missing"},
            "gender_stats":{"loaded":False,"source":"not_available"},
        }
        self.device="cuda:0"
        self.warmed_up=False
        self.startup_error=None
        self.gender_stats_data=None

    def load(self):
        import mediapipe as mp
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the resident FaceLLM deployment")
        self.mp=mp
        # model_asset_buffer avoids Windows non-ASCII path problems in native APIs.
        base=mp.tasks.BaseOptions
        vision=mp.tasks.vision
        self.detector=vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=base(model_asset_buffer=(self.settings.weights/"blaze_face_short_range.tflite").read_bytes()),
            min_detection_confidence=self.settings.detector_score))
        model_bytes=np.frombuffer((self.settings.weights/"face_detection_yunet_2023mar.onnx").read_bytes(),dtype=np.uint8)
        self.yunet=cv2.FaceDetectorYN.create("onnx",model_bytes,np.array([],dtype=np.uint8),(320,320),self.settings.detector_score,.3,5000)
        self.models["gate_detectors"]["loaded"]=True
        self.landmarker=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base(model_asset_buffer=(self.settings.weights/"face_landmarker.task").read_bytes()),
            num_faces=5,output_facial_transformation_matrixes=True))
        self.models["landmarks"]["loaded"]=True
        self.numeric=NumericalModels(self.settings.weights,self.device)
        self.models["beauty_2dfap"].update(loaded=True,weights_status=self.numeric.beauty_status,weights_ready=self.numeric.beauty_status!="untrained_init")
        self.models["apparent_age"].update(loaded=True,weights_status=self.numeric.age_status)
        self.llm=FaceLLM(self.settings,self.device)
        self.models["facellm"].update(loaded=True,precision=self.llm.precision)
        self.gender_stats_data=load_gender_stats(self.settings.weights)
        if self.gender_stats_data is not None:
            self.models["gender_stats"].update(loaded=True,source=self.gender_stats_data.get("schema_version","scut-geometry-v1"))
        self.connections=[(c.start,c.end) for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]

    def detect(self,image):
        h,w=image.shape[:2]
        mp_image=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        mp_boxes=[]
        for detection in self.detector.detect(mp_image).detections:
            b=detection.bounding_box
            mp_boxes.append({"bbox_px":[b.origin_x,b.origin_y,b.origin_x+b.width,b.origin_y+b.height],"score":detection.categories[0].score})
        self.yunet.setInputSize((w,h))
        _,detections=self.yunet.detect(image)
        yn_boxes=[] if detections is None else [{"bbox_px":[float(d[0]),float(d[1]),float(d[0]+d[2]),float(d[1]+d[3])],"score":float(d[-1])} for d in detections]
        return match_candidates(mp_boxes,yn_boxes,w,h,self.settings.detector_iou,self.settings.min_face_side)

    def gate(self,image,cancelled=lambda:False):
        candidates=self.detect(image)
        humans=[]
        for candidate in candidates:
            if cancelled():
                raise GateRejected("GATE_UNAVAILABLE","task cancelled")
            face,_=crop(image,candidate["bbox_px"])
            if self.llm.classify(face,cancelled)=="HUMAN":
                humans.append(candidate)
        if not humans:
            raise GateRejected("NO_HUMAN_FACE","0 human candidates remained after per-candidate gate")
        return humans

    def target(self,image,candidate):
        face,bounds=crop(image,candidate["bbox_px"])
        h,w=face.shape[:2]
        if min(candidate["bbox_px"][2]-candidate["bbox_px"][0],candidate["bbox_px"][3]-candidate["bbox_px"][1])<self.settings.min_face_side:
            raise GateRejected("FACE_TOO_SMALL","selected candidate below minimum face side")
        detected=self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)))
        if not detected.face_landmarks:
            raise GateRejected("GATE_UNAVAILABLE","selected human landmark extraction failed")
        meshes=[np.array([[p.x*w+bounds[0],p.y*h+bounds[1]] for p in mesh],dtype=np.float64) for mesh in detected.face_landmarks]
        overlaps=[iou([p[:,0].min(),p[:,1].min(),p[:,0].max(),p[:,1].max()],candidate["bbox_px"]) for p in meshes]
        index=int(np.argmax(overlaps))
        if overlaps[index]<.2:
            raise GateRejected("GATE_UNAVAILABLE","landmark mesh does not match selected face")
        points=meshes[index]
        x1,y1,x2,y2=candidate["bbox_px"]
        geometry,pose,blur,brightness=measure(points,detected.facial_transformation_matrixes[index],image[y1:y2,x1:x2],self.settings)
        target={**candidate,"image_width":image.shape[1],"image_height":image.shape[0],"pose_deg":pose,"blur_laplacian_var":blur,"brightness_mean":brightness}
        target.pop("score",None)
        return {"face":face,"bounds":bounds,"points":points,"target_face":target,"geometry":geometry}

    def scores(self,context):
        return self.numeric.score(context["face"])

    def explanation(self,image,context):
        heat=self.numeric.gradcam(context["face"])
        x1,y1,x2,y2=context["bounds"]
        colored=cv2.applyColorMap(np.uint8(heat*255),cv2.COLORMAP_JET)
        composite=image.copy()
        alpha=(heat*.55)[...,None]
        composite[y1:y2,x1:x2]=(context["face"]*(1-alpha)+colored*alpha).astype(np.uint8)
        baseline,regions=explain_regions(context["face"],context["points"]-np.array([x1,y1]),self.numeric)
        return {"random_baseline_mean_abs_delta":baseline,"region_sensitivity":regions}, {
            "landmarks_overlay.jpg":jpeg(overlay(image,context["points"],self.connections)),"gradcam_overlay.jpg":jpeg(composite)}

    def describe(self,context,cancelled=lambda:False):
        return self.llm.describe(context["face"],cancelled)

    def warmup(self):
        raw=(self.settings.weights/"warmup.jpg").read_bytes()
        from .imaging import decode_image
        image=decode_image(raw,self.settings).bgr
        torch.cuda.reset_peak_memory_stats()
        start=time.monotonic()
        candidates=self.gate(image)
        context=self.target(image,candidates[0])
        self.scores(context)
        self.explanation(image,context)
        text_status="ok"
        try:
            self.describe(context)
        except Exception:
            # The post-text module does not control availability of the front gate.
            text_status="failed"
        self.warmup_metrics={"elapsed_ms":(time.monotonic()-start)*1000,"allocated_bytes":torch.cuda.memory_allocated(),
                             "reserved_bytes":torch.cuda.memory_reserved(),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                             "text_ok":float(text_status=="ok")}
        self.warmed_up=True

    def close(self):
        for name in ("detector","landmarker"):
            component=getattr(self,name,None)
            if component is not None:
                component.close()


METRIC_NAMES={
    "left_eye_width_face":"左眼宽占面宽比",
    "inner_canthal_face":"内眦距占面宽比",
    "right_eye_width_face":"右眼宽占面宽比",
    "left_temple_face":"左侧太阳穴占面宽比",
    "right_temple_face":"右侧太阳穴占面宽比",
    "left_brow_eye_ratio":"左眉眼比",
    "right_brow_eye_ratio":"右眉眼比",
    "middle_lower_ratio":"中下庭比",
    "nose_inner_canthal_ratio":"鼻翼内眦比",
    "upper_lower_lip_ratio":"上下唇厚度比",
    "mirror_asymmetry":"镜像不对称偏差",
}


def load_gender_stats(weights:Path):
    stats_path=weights/"gender_stats.json"
    if not stats_path.exists():
        return None
    payload=json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(payload,dict) or not isinstance(payload.get("splits",{}).get("train"),dict):
        raise ValueError("Invalid gender_stats.json schema: missing splits.train")
    return payload


def _format_gender_trend(gender_stats,gender_key:str)->tuple[str,str]:
    if gender_stats is None:
        return "","课程任务书建议的关注维度；训练集分组相关性统计尚未提供，不作统计趋势推断。"
    metrics=gender_stats.get("splits",{}).get("train",{}).get(gender_key,{}).get("metrics",{})
    confirmed=[]
    for name,item in metrics.items():
        if isinstance(item,dict) and item.get("display_trend") is True and isinstance(item.get("pearson"),dict):
            r=item["pearson"].get("r")
            q=item["pearson"].get("q_bh")
            if isinstance(r,(int,float)) and isinstance(q,(int,float)):
                confirmed.append(f"{METRIC_NAMES.get(name,name)}(r={r:+.2f}, q={q:.3f})")
    if not confirmed:
        return "","课程任务书建议的关注维度；当前统计文件无通过留出折复核与多重检验校正的显著相关性指标，不展示统计趋势。"
    return f" 训练集确认关联指标：{'、'.join(confirmed[:3])}。","课程任务书建议维度 + SCUT-FBP5500 训练集分组统计相关性（经留出折同方向确认与 BH 校正，非因果结论）。"


def gender_views(geometry,explanation,gender_stats=None):
    sensitive="、".join(region["region_name"] for region in explanation["region_sensitivity"] if region["is_sensitive"]) or "暂无超过参照阈值的区域"
    f_trend,f_note=_format_gender_trend(gender_stats,"female")
    m_trend,m_note=_format_gender_trend(gender_stats,"male")
    return {
        "general":{"title":"通用综合参考","focus_keys":["eyes","nose","mouth","brows","contour","expression"],
                   "summary":f"中下庭比 {geometry['mid_to_lower_face_ratio']:.2f}，镜像对称度 {geometry['mirror_symmetry_score']:.1%}；模型区域敏感性：{sensitive}。",
                   "evidence_note":"像素几何测量 + 模型区域敏感性 + 可见外观描述"},
        "female":{"title":"女性侧重参考维度","focus_keys":["eyes","mouth","contour"],
                  "summary":f"按课程任务书建议关注眼型、唇形和轮廓；当前上下唇厚度比 {geometry['upper_to_lower_lip_ratio']:.2f}。{f_trend}",
                  "evidence_note":f_note},
        "male":{"title":"男性侧重参考维度","focus_keys":["contour","brows","nose"],
                "summary":f"按课程任务书建议关注轮廓、眉眼和鼻部；当前眉眼比 {geometry['brow_to_eye_ratio']:.2f}。{m_trend}",
                "evidence_note":m_note}}
