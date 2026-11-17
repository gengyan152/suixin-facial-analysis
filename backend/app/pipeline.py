import time

import cv2
import numpy as np
import torch
from .config import GateRejected
from .imaging import crop, match_candidates, iou
from .quality import check_quality
from .numeric import NumericalModels
from .facellm import FaceLLM


class Pipeline:
    def __init__(self, settings):
        self.settings = settings
        self.models = {
            "gate_detectors": {"loaded": False, "version": "mediapipe_face_detector + opencv_yunet"},
            "landmarks": {"loaded": False, "version": "mediapipe_face_landmarker_pose"},
            "facellm": {"loaded": False, "version": "Idiap/FaceLLM-1B", "precision": "bf16"},
            "beauty_2dfap": {"loaded": False, "version": "2d_fap_mobilenetv2", "weights_status": "untrained_init", "weights_ready": False},
            "apparent_age": {"loaded": False, "version": "se_resnext50_appa_real", "weights_status": "missing"},
        }
        self.device = "cuda:0"
        self.warmed_up = False
        self.startup_error = None

    def load(self):
        import mediapipe as mp
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the resident FaceLLM deployment")
        self.mp = mp
        # model_asset_buffer avoids Windows non-ASCII path problems in native APIs.
        base = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        self.detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=base(model_asset_buffer=(self.settings.weights / "blaze_face_short_range.tflite").read_bytes()),
            min_detection_confidence=self.settings.detector_score))
        model_bytes = np.frombuffer((self.settings.weights / "face_detection_yunet_2023mar.onnx").read_bytes(), dtype=np.uint8)
        self.yunet = cv2.FaceDetectorYN.create("onnx", model_bytes, np.array([], dtype=np.uint8), (320, 320), self.settings.detector_score, .3, 5000)
        self.models["gate_detectors"]["loaded"] = True
        self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=base(model_asset_buffer=(self.settings.weights / "face_landmarker.task").read_bytes()),
            num_faces=5, output_facial_transformation_matrixes=True))
        self.models["landmarks"]["loaded"] = True
        self.numeric = NumericalModels(self.settings.weights, self.device)
        self.models["beauty_2dfap"].update(loaded=True, weights_status=self.numeric.beauty_status, weights_ready=self.numeric.beauty_status != "untrained_init")
        self.models["apparent_age"].update(loaded=True, weights_status=self.numeric.age_status)
        self.llm = FaceLLM(self.settings, self.device)
        self.models["facellm"].update(loaded=True, precision=self.llm.precision)

    def detect(self, image):
        h, w = image.shape[:2]
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mp_boxes = []
        for detection in self.detector.detect(mp_image).detections:
            b = detection.bounding_box
            mp_boxes.append({"bbox_px": [b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height], "score": detection.categories[0].score})
        self.yunet.setInputSize((w, h))
        _, detections = self.yunet.detect(image)
        yn_boxes = [] if detections is None else [{"bbox_px": [float(d[0]), float(d[1]), float(d[0] + d[2]), float(d[1] + d[3])], "score": float(d[-1])} for d in detections]
        return match_candidates(mp_boxes, yn_boxes, w, h, self.settings.detector_iou, self.settings.min_face_side)

    def gate(self, image, cancelled=lambda: False):
        candidates = self.detect(image)
        humans = []
        for candidate in candidates:
            if cancelled():
                raise GateRejected("GATE_UNAVAILABLE", "task cancelled")
            face, _ = crop(image, candidate["bbox_px"])
            if self.llm.classify(face, cancelled) == "HUMAN":
                humans.append(candidate)
        if not humans:
            raise GateRejected("NO_HUMAN_FACE", "0 human candidates remained after per-candidate gate")
        return humans

    def target(self, image, candidate):
        face, bounds = crop(image, candidate["bbox_px"])
        h, w = face.shape[:2]
        if min(candidate["bbox_px"][2] - candidate["bbox_px"][0], candidate["bbox_px"][3] - candidate["bbox_px"][1]) < self.settings.min_face_side:
            raise GateRejected("FACE_TOO_SMALL", "selected candidate below minimum face side")
        detected = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))
        if not detected.face_landmarks:
            raise GateRejected("GATE_UNAVAILABLE", "selected human landmark extraction failed")
        meshes = [np.array([[p.x * w + bounds[0], p.y * h + bounds[1]] for p in mesh], dtype=np.float64) for mesh in detected.face_landmarks]
        overlaps = [iou([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], candidate["bbox_px"]) for p in meshes]
        index = int(np.argmax(overlaps))
        if overlaps[index] < .2:
            raise GateRejected("GATE_UNAVAILABLE", "landmark mesh does not match selected face")
        x1, y1, x2, y2 = candidate["bbox_px"]
        pose, blur, brightness = check_quality(detected.facial_transformation_matrixes[index], image[y1:y2, x1:x2], self.settings)
        target = {**candidate, "image_width": image.shape[1], "image_height": image.shape[0], "pose_deg": pose, "blur_laplacian_var": blur, "brightness_mean": brightness}
        target.pop("score", None)
        return {"face": face, "target_face": target}

    def scores(self, context):
        return self.numeric.score(context["face"])

    def describe(self, context, cancelled=lambda: False):
        return self.llm.describe(context["face"], cancelled)

    def warmup(self):
        raw = (self.settings.weights / "warmup.jpg").read_bytes()
        from .imaging import decode_image
        image = decode_image(raw, self.settings).bgr
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        candidates = self.gate(image)
        context = self.target(image, candidates[0])
        self.scores(context)
        text_status = "ok"
        try:
            self.describe(context)
        except Exception:
            # The post-text module does not control availability of the front gate.
            text_status = "failed"
        self.warmup_metrics = {
            "elapsed_ms": (time.monotonic() - start) * 1000,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "text_ok": float(text_status == "ok"),
        }
        self.warmed_up = True

    def close(self):
        for name in ("detector", "landmarker"):
            component = getattr(self, name, None)
            if component is not None:
                component.close()
