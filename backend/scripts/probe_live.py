"""Diagnostic real-model run. Writes measurements only, never user images."""
from pathlib import Path
import sys
import json
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import Settings
from app.pipeline import Pipeline
from app.imaging import decode_image,crop
import torch


def main():
    settings=Settings()
    pipeline=Pipeline(settings)
    started=time.monotonic()
    print("LOAD",flush=True)
    pipeline.load()
    print("LOADED",time.monotonic()-started,torch.cuda.mem_get_info(),flush=True)
    report={}
    for name,path in [("astronaut",settings.weights/"warmup.jpg"),("grace",settings.weights/"fixtures/grace_hopper.jpg"),("dog",settings.weights/"fixtures/dog.jpg"),("cat",settings.weights/"fixtures/cat.png")]:
        image=decode_image(path.read_bytes(),settings).bgr
        candidates=pipeline.detect(image)
        print(name,"candidates",candidates,flush=True)
        report[name]={"candidates":candidates}
        for c in candidates:
            face,_=crop(image,c["bbox_px"])
            gate=pipeline.llm.classify(face)
            print(name,c["face_id"],gate,flush=True)
            if gate=="HUMAN":
                context=pipeline.target(image,c)
                print("QUALITY",context["target_face"],flush=True)
                scores=pipeline.scores(context)
                fields=pipeline.describe(context)
                report[name].update(scores=scores,target=context["target_face"],fields=fields)
                print("FULL",json.dumps(report[name],ensure_ascii=True),flush=True)
    pipeline.warmup()
    report["warmup"]=pipeline.warmup_metrics
    (settings.weights.parent/"live_probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    pipeline.close()
    print("DONE",time.monotonic()-started,flush=True)


if __name__=="__main__": main()
