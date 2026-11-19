from contextlib import asynccontextmanager
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,Response
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .imaging import decode_image
from .pipeline import Pipeline
from .schemas import Health,Submission,FaceSelection,TaskResult
from .service import TaskService,ServiceError


def create_app(settings=None,pipeline=None):
    settings=settings or Settings()
    service=TaskService(settings,pipeline or Pipeline(settings))

    @asynccontextmanager
    async def lifespan(app):
        service.start()
        yield
        await run_in_threadpool(service.close)

    app=FastAPI(title="Local Face Analysis",version="1.2",lifespan=lifespan)
    app.state.service=service

    @app.exception_handler(ServiceError)
    async def service_error(request,error):
        return JSONResponse({"error_code":error.code,"message":error.message},status_code=error.status,headers={"Cache-Control":"no-store"})

    @app.get("/api/v1/health",response_model=Health,response_model_exclude_none=True)
    async def health():
        import torch
        device=f"cuda:0 ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"
        return {"status":"ok" if service.pipeline.warmed_up else "loading" if not service.initialized.is_set() else "degraded",
                "warmed_up":service.pipeline.warmed_up,"device":device,"queue_length":len(service.queue),
                "queue_capacity":settings.queue_capacity,"ttl_seconds":settings.ttl_seconds,
                "models":service.pipeline.models,"startup_error":service.pipeline.startup_error,
                "warmup_metrics":getattr(service.pipeline,"warmup_metrics",None)}

    @app.post("/api/v1/tasks",status_code=202,response_model=Submission)
    async def submit(request:Request,client_request_id:str|None=None):
        if request.headers.get("content-type","").split(";")[0].lower() not in {"image/jpeg","image/png","application/octet-stream"}:
            raise ServiceError(400,"INVALID_IMAGE","无法解码图像数据，请选择有效的照片")
        raw=bytearray()
        async for chunk in request.stream():
            if len(raw)+len(chunk)>settings.max_body_bytes:
                raise ServiceError(413,"IMAGE_TOO_LARGE","图片大小超过 8 MB 限制，请压缩后重试")
            raw.extend(chunk)
        try:
            normalized=await run_in_threadpool(decode_image,raw,settings)
        except (ValueError,OverflowError):
            raise ServiceError(400,"INVALID_IMAGE","无法解码图像数据，请选择有效的照片") from None
        return service.submit(normalized,client_request_id)

    @app.post("/api/v1/tasks/{task_id}/select-face",status_code=202)
    async def select(task_id:str,selection:FaceSelection):
        return service.select(task_id,selection.image_id,selection.face_id)

    @app.get("/api/v1/tasks/{task_id}",response_model=TaskResult)
    async def result(task_id:str):
        return service.get(task_id)

    @app.get("/api/v1/tasks/{task_id}/artifacts/{artifact_name}")
    async def artifact(task_id:str,artifact_name:str):
        return Response(service.artifact(task_id,artifact_name),media_type="image/jpeg",headers={"Cache-Control":"no-store"})

    @app.delete("/api/v1/tasks/{task_id}")
    async def delete(task_id:str):
        return service.delete(task_id)

    return app


app=create_app()
