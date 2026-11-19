import json
import re
import time

import cv2
from PIL import Image
import torch
from .config import GateRejected

FIELD_LABELS = {"brows":"眉部","eyes":"眼部","nose":"鼻部","mouth":"嘴部","contour":"脸部轮廓","expression":"可见表情"}
FIELD_ALIASES = {
    "brows": ("眉毛",),
    "eyes": ("眼睛",),
    "nose": ("鼻子",),
    "mouth": ("嘴唇", "嘴巴"),
    "contour": ("脸型",),
    "expression": ("表情",),
}

GATE_PROMPT = (
    "Classify the most prominent subject in the image. Answer with one label: "
    "HUMAN (a real human face), NON_HUMAN (an animal, a toy, or an illustration), "
    "UNKNOWN (no recognizable subject)."
)
TEXT_PROMPT = (
    "请用简体中文描述照片中的五官。每个部位写一句话，说清看得见的外形。看不清的部位写 null。"
    "只输出一个 JSON 对象，六个键分别是 brows（眉毛）、eyes（眼睛）、nose（鼻子）、"
    "mouth（嘴唇）、contour（脸型）、expression（可见表情）。"
    "眉部写双眼上方的眉毛。可见表情写嘴角或眉间的动作。"
    "省略审美评价以及对身份、年龄、性别、族裔、健康、情绪和性格的推测。"
)
TEXT_TOKENS = 220
INTERNVL_VERSION = "OpenGVLab/InternVL3-2B-hf"


def parse_gate(text):
    value = text.strip().strip('`"\' .\n').upper()
    if value not in {"HUMAN","NON_HUMAN","UNKNOWN"}:
        raise GateRejected("GATE_UNAVAILABLE", "unparseable front-gate enumeration")
    return value


def parse_fields(text):
    # Decode the first JSON object, including when the model wraps it in a fence.
    start = text.find("{")
    if start < 0:
        raise ValueError("description JSON absent")
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    result = {}
    has_usable_field = False
    for key, label in FIELD_LABELS.items():
        value = next((data[name] for name in (key, label, *FIELD_ALIASES[key]) if name in data), "")
        description = value.strip() if isinstance(value, str) else ""
        content = description.rstrip("。！？.!? ")
        unavailable = (
            value is None
            or content in {"无", "不可见"}
            or content.endswith(("无法判断", "无法看清", "无法辨认", "看不清"))
        )
        visible = (
            not unavailable
            and len(description) <= 64
            and re.search(r"[\u4e00-\u9fff]", description) is not None
            and re.search(r"[A-Za-z]|智慧|聪明|性格|命运|善良|亲和力|友善|人格|智商|性取向|健康|年龄|性别|族裔|颜值|气质|情绪", description) is None
        )
        has_usable_field = has_usable_field or unavailable or visible
        result[key] = {"label": label, "description": description if visible else "无法判断", "visible": visible}
    if not has_usable_field:
        raise ValueError("description contains no usable fields")
    return result


class InternVL:
    def __init__(self, settings, device, path, adapter_path=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        if adapter_path is not None:
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                if not (adapter_path / name).is_file():
                    raise FileNotFoundError(f"Missing LoRA file: {adapter_path / name}")
        self.settings = settings
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.precision = "bf16" if self.dtype == torch.bfloat16 else "fp16"
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        # Bound InternVL dynamic tiles; only crop content is sent to the local model.
        self.processor.image_processor.max_patches = 1
        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=self.dtype, local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa",
        )
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True, is_trainable=False)
        self.model = model.to(device).eval()

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

    def describe(self, image, cancelled=lambda: False):
        return parse_fields(self.generate(image, TEXT_PROMPT, TEXT_TOKENS, self.settings.text_timeout, cancelled))


    def classify(self,image,cancelled=lambda:False):
        try:
            with self.model.disable_adapter():
                return parse_gate(self.generate(image, GATE_PROMPT, 12, self.settings.gate_timeout, cancelled))
        except GateRejected:
            raise
        except Exception as error:
            raise GateRejected("GATE_UNAVAILABLE",type(error).__name__) from error


def compose_narrative(scores: dict, fields: dict | None) -> str:
    age = int(scores.get("apparent_age_display", round(float(scores.get("apparent_age", 22)))))
    beauty = float(scores.get("beauty_display", round(float(scores.get("beauty_score", 3.0)), 1)))
    lead = f"画面中的人物表观年龄约 {age} 岁，颜值评分为 {beauty:.1f} 分（满分 5.0 分）。"
    if fields is None:
        return lead

    observations = [
        fields[key]["description"].rstrip("。！？.!? ")
        for key in ("contour", "brows", "eyes", "nose", "mouth", "expression")
        if fields[key]["visible"]
    ]
    return lead + ("。".join(observations) + "。" if observations else "")
