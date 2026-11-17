import json
import re
import time

import cv2
from PIL import Image
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError
import torch
from .config import GateRejected

FIELD_LABELS = {"brows":"眉部","eyes":"眼部","nose":"鼻部","mouth":"嘴部","contour":"脸部轮廓","expression":"可见表情"}
ATTRIBUTES = {
    "brows":{"STRAIGHT":"眉形较平直。","ARCHED":"眉形呈弧形。","ANGLED":"眉形有较明显转折。"},
    "eyes":{"OPEN":"眼睛睁开，眼裂轮廓可见。","NARROWED":"眼睛略微眯起。","CLOSED":"眼睛闭合。"},
    "nose":{"STRAIGHT":"可见鼻梁线条较直。","ROUNDED":"可见鼻部轮廓较圆润。","WIDE":"可见鼻翼轮廓较宽。"},
    "mouth":{"CLOSED":"双唇闭合。","PARTED":"双唇分开。","TEETH_VISIBLE":"嘴部张开，可见牙齿。"},
    "contour":{"OVAL":"可见面部轮廓偏椭圆。","ROUND":"可见面部轮廓偏圆。","ANGULAR":"可见下颌轮廓有较明显转折。","LONG":"可见面部轮廓偏长。"},
    "expression":{"SMILING":"可见嘴角上扬，呈微笑表情。","NEUTRAL":"可见表情较平静。","FROWNING":"可见眉间收紧。"},
}
ATTRIBUTE_QUESTIONS = {
    "brows":"What is the shape of the eyebrows: straight, arched or angled? Answer with a short visible description.",
    "eyes":"Are the eyes open, narrowed or closed? Answer briefly about only the eyes.",
    "nose":"Describe the visible shape of the nose in one short sentence.",
    "mouth":"Are teeth visible? Are the lips parted or closed? Answer briefly about the mouth.",
    "contour":"Is the face shape oval, round, long or angular? Answer briefly about facial outline only.",
    "expression":"Is the person visibly smiling, frowning, or neutral? Answer briefly about only visible facial expression.",
}


def controlled_attribute(key,text):
    """Map only whitelisted visible English attributes into Chinese templates."""
    text=text.lower()
    # Invisible teeth do not make an explicitly described lip position unknown.
    if re.search(r"unknown|cannot (tell|determine|see)|can't (tell|determine|see)|not (clear|visible)|unable to|whether .{0,28} or|obscured|occluded|covered|hidden",
                 re.sub(r"\bteeth\s+(?:are\s+)?not visible\b", "", text) if key == "mouth" else text):
        return "UNKNOWN"
    if key=="mouth":
        if re.search(r"\bteeth\b",text) and not re.search(r"teeth.{0,20}\b(not|n't|no|none|without)\b|\b(no|not|without|neither)\b.{0,20}teeth",text):
            return "TEETH_VISIBLE"
        if re.search(r"\bclosed\b",text) and not re.search(r"\b(not|no|isn't|aren't|neither|nor)\b.{0,16}\bclosed\b",text):
            return "CLOSED"
        if re.search(r"\b(parted|open)\b",text) and not re.search(r"\b(not|no|isn't|aren't|neither|nor)\b.{0,16}\b(parted|open)\b",text):
            return "PARTED"
        return "UNKNOWN"
    patterns={
        "brows":[("ARCHED",r"arch|curved"),("ANGLED",r"angled|angular"),("STRAIGHT",r"straight")],
        "eyes":[("CLOSED",r"\bclosed\b"),("NARROWED",r"narrow|squint"),("OPEN",r"\bopen\b")],
        "nose":[("ROUNDED",r"round"),("STRAIGHT",r"straight"),("WIDE",r"wide|broad")],
        "contour":[("OVAL",r"oval"),("ROUND",r"round"),("ANGULAR",r"angular|square"),("LONG",r"long|elongated")],
        "expression":[("FROWNING",r"frown"),("SMILING",r"smil"),("NEUTRAL",r"neutral")],
    }
    for value,pattern in patterns[key]:
        if re.search(pattern,text) and not re.search(r"(?:not|no|isn't|aren't)\s+(?:\w+\s+)?"+pattern,text):
            return value
    return "UNKNOWN"
GATE_PROMPT = """Classify only the central face in this crop. Is it an analyzable HUMAN face in a photograph?
Human faces in ordinary photographs are HUMAN. Dogs, cats, primates, other animals,
plush toys, dolls, cartoons, illustrations and objects are NON_HUMAN.
If the central face cannot be determined, use UNKNOWN. Output exactly one token:
HUMAN or NON_HUMAN or UNKNOWN. No explanation."""
TEXT_PROMPT = """Observe only visible appearance of this face. Return one JSON object with exactly six keys:
brows, eyes, nose, mouth, contour, expression. Each value must have label (Chinese part name),
description (one short Chinese sentence about visible appearance), visible (boolean).
If obscured or uncertain use description 无法判断 and visible false. Do not guess unseen features.
Describe shape, size, edges or visible expression only. No attractiveness, score, identity,
ethnicity, gender, health or personality. Output JSON only."""


class VisibleField(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: str
    description: str
    visible: StrictBool


def parse_gate(text):
    value = text.strip().strip('`"\' .\n').upper()
    if value not in {"HUMAN","NON_HUMAN","UNKNOWN"}:
        raise GateRejected("GATE_UNAVAILABLE", "unparseable front-gate enumeration")
    return value


def parse_fields(text):
    # JSON decoder supports fenced output/preamble, but never evals generated text.
    start = text.find("{")
    if start < 0:
        raise ValueError("description JSON absent")
    data,_ = json.JSONDecoder().raw_decode(text[start:])
    if isinstance(data,dict) and "fields" in data:
        data = data["fields"]
    if not isinstance(data,dict) or set(data) != set(FIELD_LABELS):
        raise ValueError("description requires all six fields")
    result = {}
    for key,label in FIELD_LABELS.items():
        field = VisibleField.model_validate(data[key])
        description = field.description.strip()
        if not description or len(description)>300:
            raise ValueError("description length invalid")
        if not re.search(r"[\u4e00-\u9fff]",description) or re.search(r"智慧|聪明|性格|命运|善良|亲和力|友善|人格|智商|性取向",description):
            raise ValueError("description outside visible Chinese appearance contract")
        result[key] = {"label":label,"description":description if field.visible else "无法判断","visible":field.visible}
    return result


class FaceLLM:
    def __init__(self,settings,device):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.settings = settings
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.precision = "bf16" if self.dtype==torch.bfloat16 else "fp16"
        path = settings.weights / "FaceLLM-1B" / "model"
        self.processor = AutoProcessor.from_pretrained(path,local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(path,torch_dtype=self.dtype,
                        local_files_only=True,low_cpu_mem_usage=True,attn_implementation="sdpa").to(device).eval()
        # Bound InternVL dynamic tiles; only crop content is sent to the local model.
        self.processor.image_processor.max_patches = 1

    def generate(self,image,prompt,tokens,timeout,cancelled=lambda:False):
        from transformers import StoppingCriteria,StoppingCriteriaList
        deadline = time.monotonic()+timeout
        class Deadline(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                return time.monotonic()>deadline or cancelled()
        pil = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        messages=[{"role":"user","content":[{"type":"image","image":pil},{"type":"text","text":prompt}]}]
        with torch.no_grad():
            inputs = self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                           return_dict=True,return_tensors="pt").to(self.device,dtype=self.dtype)
            generated = self.model.generate(**inputs,max_new_tokens=tokens,do_sample=False,
                             pad_token_id=self.processor.tokenizer.eos_token_id,
                             stopping_criteria=StoppingCriteriaList([Deadline()]))
            text = self.processor.decode(generated[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        if cancelled() or time.monotonic()>deadline:
            raise TimeoutError("generation cancelled or deadline exceeded")
        return text

    def classify(self,image,cancelled=lambda:False):
        try:
            return parse_gate(self.generate(image,GATE_PROMPT,12,self.settings.gate_timeout,cancelled))
        except GateRejected:
            raise
        except Exception as error:
            raise GateRejected("GATE_UNAVAILABLE",type(error).__name__) from error

    def describe(self,image,cancelled=lambda:False):
        started=time.monotonic()
        answer=self.generate(image,TEXT_PROMPT,220,self.settings.text_timeout,cancelled)
        try:
            return parse_fields(answer)
        except (ValueError,ValidationError):
            # Controlled fallback: ask for six short observations separately; do not
            # fabricate missing descriptions from geometry or a beauty score.
            result={}
            for key,label in FIELD_LABELS.items():
                remaining=self.settings.text_timeout-(time.monotonic()-started)
                if remaining<=0:
                    raise TimeoutError("description fallback timed out")
                options=ATTRIBUTES[key]
                answer=self.generate(image,ATTRIBUTE_QUESTIONS[key],48,remaining,cancelled)
                observation=controlled_attribute(key,answer)
                visible=observation!="UNKNOWN"
                result[key]={"label":label,"description":options[observation] if visible else "无法判断","visible":visible}
            return result


_POLISH_MAP = {
    "眼睛睁开，眼裂轮廓可见": "双眸自然舒展、眼型清晰",
    "眼睛略微眯起": "眉眼微弯、神态放松",
    "眼睛闭合": "双目轻阖",
    "眉形较平直": "眉形平直干净",
    "眉形呈弧形": "眉形柔和呈自然弧度",
    "眉形有较明显转折": "眉峰线条利落分明",
    "可见鼻梁线条较直": "鼻梁线条平直挺括",
    "可见鼻部轮廓较圆润": "鼻头线条柔和圆润",
    "可见鼻翼轮廓较宽": "鼻部轮廓舒展大气",
    "双唇闭合": "双唇轻合",
    "双唇分开": "双唇微启",
    "嘴部张开，可见牙齿": "唇齿微露",
    "可见面部轮廓偏椭圆": "脸部轮廓柔和偏椭圆",
    "可见面部轮廓偏圆": "脸部线条圆润饱满",
    "可见下颌轮廓有较明显转折": "下颌轮廓清晰利落",
    "可见面部轮廓偏长": "脸部线条修长舒展",
    "可见嘴角上扬，呈微笑表情": "唇角轻扬带着浅浅笑意",
    "可见表情较平静": "神态从容平和",
    "可见眉间收紧": "眉宇间略带专注神色",
}


def _clean_clause(raw: str) -> str:
    s = raw.strip().rstrip("。！；，,. ")
    if s in _POLISH_MAP:
        return _POLISH_MAP[s]
    s = re.sub(r"^可见(?=面部|脸部|下颌|鼻|眉|眼|双唇|嘴)", "", s)
    return s


def compose_narrative(scores: dict, fields: dict | None) -> str:
    age = int(scores.get("apparent_age_display", round(float(scores.get("apparent_age", 22)))))
    beauty = float(scores.get("beauty_display", round(float(scores.get("beauty_score", 3.0)), 1)))
    raw_score = float(scores.get("beauty_score", beauty))
    if raw_score >= 4.0:
        vibe = "五官与骨相协调感十分出众，整体气质清亮灵动"
    elif raw_score >= 3.4:
        vibe = "五官比例舒展协调，整体气质清爽耐看"
    elif raw_score >= 2.8:
        vibe = "面容线条自然大方，整体观感亲切温和"
    else:
        vibe = "面容神态真实松弛，整体风格质朴自然"

    lead = f"画面中的人物表观年龄约 {age} 岁，颜值评分为 {beauty:.1f} 分（满分 5.0 分），{vibe}。"
    if not isinstance(fields, dict):
        return lead

    ordered_keys = ("contour", "brows", "eyes", "nose", "mouth", "expression")
    clauses = []
    for k in ordered_keys:
        item = fields.get(k)
        if isinstance(item, dict) and item.get("visible") is True:
            desc = str(item.get("description", "")).strip()
            if desc and desc != "无法判断":
                clauses.append(_clean_clause(desc))

    if not clauses:
        return lead
    detail = "，".join(clauses)
    return f"{lead}细看五官细节，{detail}，整体面容舒展和谐。"
