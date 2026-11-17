from collections import deque
from copy import deepcopy
from dataclasses import dataclass,field
from datetime import datetime
import logging
import threading
import time
import uuid

from .config import ETHICS,REJECTIONS,GateRejected
from .facellm import FIELD_LABELS,compose_narrative
from .imaging import jpeg
from .pipeline import gender_views

logger=logging.getLogger(__name__)


class ServiceError(Exception):
    def __init__(self,status,code,message):
        self.status,self.code,self.message=status,code,message


@dataclass
class Task:
    task_id: str
    image_id: str
    image: object
    mapping: dict
    client_request_id: str | None
    created_at: str = field(default_factory=lambda:datetime.now().astimezone().isoformat())
    created: float = field(default_factory=time.monotonic)
    result: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    selected: str | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    history: list = field(default_factory=lambda:["queued"])


class TaskService:
    def __init__(self,settings,pipeline):
        self.settings,self.pipeline=settings,pipeline
        self.condition=threading.Condition(threading.RLock())
        self.tasks={}
        self.queue=deque()
        self.stopped=threading.Event()
        self.initialized=threading.Event()
        self.worker=threading.Thread(target=self._worker,name="single-gpu-worker",daemon=True)
        self.cleaner=threading.Thread(target=self._cleaner,name="task-ttl-cleaner",daemon=True)

    def start(self):
        self.worker.start()
        self.cleaner.start()

    def close(self):
        self.stopped.set()
        with self.condition:
            for task in list(self.tasks.values()):
                self._destroy(task)
            self.condition.notify_all()
        self.worker.join(timeout=60)
        self.cleaner.join(timeout=2)

    def _lookup(self,task_id):
        task=self.tasks.get(task_id)
        if task is None:
            raise ServiceError(404,"TASK_EXPIRED","临时图像已过期销毁，请重新选择或拍摄照片")
        if time.monotonic()-task.created >= self.settings.ttl_seconds:
            self._destroy(task)
            raise ServiceError(410,"TASK_EXPIRED","临时图像已过期销毁，请重新选择或拍摄照片")
        return task

    def _destroy(self,task):
        task.cancelled.set()
        task.image=None
        task.artifacts.clear()
        task.candidates.clear()
        task.mapping.clear()
        task.result.clear()
        self.tasks.pop(task.task_id,None)
        self.queue=deque(item for item in self.queue if item != task.task_id)

    def submit(self,normalized,client_request_id=None):
        with self.condition:
            for old in list(self.tasks.values()):
                if time.monotonic()-old.created>=self.settings.ttl_seconds:
                    self._destroy(old)
            if client_request_id:
                for old in self.tasks.values():
                    if old.client_request_id==client_request_id and old.result["status"] not in {"completed","partial_error","rejected"}:
                        return self._submission(old)
            if len(self.queue)>=self.settings.queue_capacity or len(self.tasks)>=self.settings.max_tasks:
                raise ServiceError(429,"QUEUE_FULL","当前分析任务队列已满，请稍后再试")
            token=uuid.uuid4().hex[:16]
            task=Task("tsk_"+token,"img_"+token,normalized.bgr,normalized.mapping,client_request_id)
            task.result={"task_id":task.task_id,"image_id":task.image_id,"status":"queued","is_offline_replay":False,
                         "rejection":None,"selection":None,"target_face":None,"scores":None,"geometry":None,
                         "explanation":None,"text_analysis":None,"gender_views":None,"ethics_notice":ETHICS,"timings_ms":{}}
            self.tasks[task.task_id]=task
            self.queue.append(task.task_id)
            reply=self._submission(task)
            self.condition.notify()
            return reply

    def _submission(self,task):
        return {"task_id":task.task_id,"image_id":task.image_id,"status":task.result["status"],
                "queue_position":list(self.queue).index(task.task_id) if task.task_id in self.queue else 0,"created_at":task.created_at}

    def select(self,task_id,image_id,face_id):
        with self.condition:
            task=self._lookup(task_id)
            if task.image_id!=image_id or task.result["status"]!="need_selection" or not any(c["face_id"]==face_id for c in task.candidates):
                raise ServiceError(400,"INVALID_FACE_ID","所选目标人脸编号无效")
            if len(self.queue)>=self.settings.queue_capacity:
                raise ServiceError(429,"QUEUE_FULL","当前分析任务队列已满，请稍后再试")
            task.selected=face_id
            self._publish(task,"processing",selection=None,ethics_notice=ETHICS)
            self.queue.append(task.task_id)
            self.condition.notify()
            return {"task_id":task_id,"status":"processing","selected_face_id":face_id}

    def get(self,task_id):
        with self.condition:
            task=self._lookup(task_id)
            result=deepcopy(task.result)
            result["timings_ms"]["total_elapsed_ms"]=(time.monotonic()-task.created)*1000 if result["status"] not in {"completed","partial_error","rejected"} else result["timings_ms"].get("total_elapsed_ms",0)
            return result

    def artifact(self,task_id,name):
        with self.condition:
            task=self._lookup(task_id)
            if name not in task.artifacts:
                raise ServiceError(404,"ARTIFACT_NOT_READY","图像资源尚未就绪或不存在")
            return task.artifacts[name]

    def delete(self,task_id):
        with self.condition:
            self._destroy(self._lookup(task_id))
        return {"task_id":task_id,"deleted":True}

    def _publish(self,task,status,artifacts=None,**fields):
        with self.condition:
            if task.cancelled.is_set():
                return False
            task.result.update(fields,status=status)
            task.result["timings_ms"]["total_elapsed_ms"]=(time.monotonic()-task.created)*1000
            if task.history[-1]!=status:
                task.history.append(status)
            if artifacts:
                task.artifacts.update(artifacts)
            return True

    def _timing(self,task,key,start):
        with self.condition:
            if not task.cancelled.is_set():
                task.result["timings_ms"][key]=(time.monotonic()-start)*1000

    def _worker(self):
        try:
            try:
                self.pipeline.load()
                self.pipeline.warmup()
            except Exception as error:
                self.pipeline.startup_error=f"{type(error).__name__}: {error}"
                logger.exception("Model startup or warmup failed")
            finally:
                self.initialized.set()
            while not self.stopped.is_set():
                with self.condition:
                    self.condition.wait_for(lambda:self.queue or self.stopped.is_set(),timeout=1)
                    if self.stopped.is_set():
                        break
                    if not self.queue:
                        continue
                    task=self.tasks.get(self.queue.popleft())
                if task and not task.cancelled.is_set():
                    self._run(task)
        finally:
            self.pipeline.close()

    def _run(self,task):
        context=None
        image=None
        try:
            self._publish(task,"processing")
            if not self.pipeline.warmed_up:
                raise GateRejected("GATE_UNAVAILABLE",self.pipeline.startup_error or "startup incomplete")
            with self.condition:
                image=task.image
            if image is None or task.cancelled.is_set():
                return
            if task.selected is None:
                start=time.monotonic()
                try:
                    candidates=self.pipeline.gate(image,task.cancelled.is_set)
                finally:
                    self._timing(task,"gate_ms",start)
                with self.condition:
                    if task.cancelled.is_set():
                        return
                    task.candidates=candidates
                normalized=jpeg(image)
                if len(candidates)>1:
                    self._publish(task,"need_selection",selection={"image_width":image.shape[1],"image_height":image.shape[0],
                           "preview_image_url":f"/api/v1/tasks/{task.task_id}/artifacts/normalized.jpg","candidates":candidates},
                           artifacts={"normalized.jpg":normalized},ethics_notice="检测到多张人脸，请点击选择单人目标继续分析。")
                    return
                task.selected=candidates[0]["face_id"]
                self._publish(task,"processing",artifacts={"normalized.jpg":normalized})
            candidate=next(c for c in task.candidates if c["face_id"]==task.selected)
            start=time.monotonic()
            context=self.pipeline.target(image,candidate)
            scores=self.pipeline.scores(context)
            self._timing(task,"scores_ms",start)
            versions={"beauty":f"2d_fap_mobilenetv2 ({self.pipeline.models['beauty_2dfap']['weights_status']})",
                      "age":f"se_resnext50_appa_real ({self.pipeline.models['apparent_age']['weights_status']})","text":"Idiap/FaceLLM-1B"}
            if not self._publish(task,"scores_ready",scores=scores,geometry=context["geometry"],target_face=context["target_face"],model_versions=versions):
                return
            start=time.monotonic()
            explanation,artifacts=self.pipeline.explanation(image,context)
            explanation.update({key:f"/api/v1/tasks/{task.task_id}/artifacts/{name}" for key,name in {
                "original_image_url":"normalized.jpg","landmarks_overlay_url":"landmarks_overlay.jpg","gradcam_overlay_url":"gradcam_overlay.jpg"}.items()})
            self._timing(task,"explanation_ms",start)
            if not self._publish(task,"explanation_ready",explanation=explanation,artifacts=artifacts):
                return
            start=time.monotonic()
            # Only this post-gate description boundary is allowed to partially fail.
            try:
                fields=self.pipeline.describe(context,task.cancelled.is_set)
                text={"status":"ok","error_message":None,"fields":fields,"narrative":compose_narrative(scores,fields)}
                final="completed"
            except Exception as error:
                logger.warning("Post-description failed: %s",type(error).__name__)
                empty_fields={key:{"label":label,"description":"无法判断","visible":False} for key,label in FIELD_LABELS.items()}
                text={"status":"error","error_message":"五官文字分析暂不可用，数值与解释图已保留。",
                      "fields":empty_fields,"narrative":compose_narrative(scores,None)}
                final="partial_error"
            self._timing(task,"text_ms",start)
            self._publish(task,final,text_analysis=text,gender_views=gender_views(context["geometry"],explanation,getattr(self.pipeline,"gender_stats_data",None)),generated_at=datetime.now().astimezone().isoformat())
        except GateRejected as error:
            self._publish(task,"rejected",rejection={"code":error.code,"message":REJECTIONS[error.code],"detail":error.detail},
                          scores=None,geometry=None,explanation=None,text_analysis=None,gender_views=None,target_face=None)
        except Exception as error:
            logger.exception("Inference stage failed")
            # Contract has no numeric/attribution failure state; fail closed and
            # remove all partial payloads instead of misusing post-text partial_error.
            self._publish(task,"rejected",rejection={"code":"GATE_UNAVAILABLE","message":REJECTIONS["GATE_UNAVAILABLE"],"detail":f"inference failure: {type(error).__name__}"},
                          scores=None,geometry=None,explanation=None,text_analysis=None,gender_views=None,target_face=None)
        finally:
            with self.condition:
                if task.result.get("status")!="need_selection":
                    task.image=None
                    task.candidates.clear()
                    if task.result.get("status")=="rejected":
                        task.artifacts.clear()
            context=None
            image=None

    def _cleaner(self):
        while not self.stopped.wait(self.settings.cleanup_interval):
            with self.condition:
                for task in list(self.tasks.values()):
                    if time.monotonic()-task.created>=self.settings.ttl_seconds:
                        self._destroy(task)
