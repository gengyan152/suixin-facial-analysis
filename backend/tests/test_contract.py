import time
from dataclasses import replace
from io import BytesIO
import threading

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.config import Settings, GateRejected
from app.main import create_app
from app.pipeline import Pipeline
from app.schemas import Health, TaskResult
from app.vlm import FIELD_LABELS


def photo(color=100):
    stream = BytesIO()
    Image.new("RGB", (300, 300), (color, color, color)).save(stream, format="PNG")
    return stream.getvalue()


class ControlledPipeline:
    """Deterministic API state-machine fixture, never a runtime fallback."""

    def __init__(self, count=1, rejection=None, text_error=False, block=None):
        self.models = Pipeline(Settings()).models
        self.models["beauty_2dfap"].update(loaded=True, weights_status="untrained_init")
        self.models["apparent_age"].update(loaded=True, weights_status="appa_real_pretrained")
        self.warmed_up = False
        self.startup_error = None
        self.count, self.rejection, self.text_error, self.block = count, rejection, text_error, block
        self.thread_ids = []

    def load(self):
        pass

    def warmup(self):
        self.warmed_up = True

    def close(self):
        pass

    def gate(self, image, cancelled):
        self.thread_ids.append(threading.get_ident())
        if self.block:
            while not self.block.wait(.01):
                if cancelled():
                    raise GateRejected("GATE_UNAVAILABLE")
        if self.rejection:
            raise GateRejected(self.rejection)
        return [{"face_id": f"face_{i}", "bbox_px": [10 + i * 90, 10, 90 + i * 90, 110], "bbox_norm": [.1, .1, .4, .5], "score": .9} for i in range(self.count)]

    def target(self, image, candidate):
        return {
            "target_face": {
                **candidate, "pose_deg": {"yaw": 0., "pitch": 0., "roll": 0.},
                "image_width": 300, "image_height": 300, "blur_laplacian_var": 100., "brightness_mean": 100.,
            }
        }

    def scores(self, context):
        return {
            "beauty_score": 3., "beauty_display": 3., "scale_min": 1, "scale_max": 5,
            "distribution_label": "模型预测评分分布（1—5 档）", "rating_probabilities": [.2] * 5,
            "distribution_std": 1., "apparent_age": 30., "apparent_age_display": 30,
            "weights_ready": False, "weights_status": "untrained_init",
        }

    def describe(self, context, cancelled):
        time.sleep(.03)
        if self.text_error:
            raise ValueError("malformed JSON")
        return {key: {"label": label, "description": "清晰可见", "visible": True} for key, label in FIELD_LABELS.items()}


def wait(client, task_id, states=("completed", "partial_error", "rejected", "need_selection")):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in states:
            return result
        time.sleep(.005)
    raise AssertionError("task did not finish")


def submit(client, **kwargs):
    response = client.post("/api/v1/tasks", content=photo(), headers={"Content-Type": "image/png"}, **kwargs)
    assert response.status_code == 202, response.text
    return response.json()


def test_health_and_single_person_stages():
    app = create_app(pipeline=ControlledPipeline())
    with TestClient(app) as client:
        assert app.state.service.initialized.wait(2)
        health = Health.model_validate(client.get("/api/v1/health").json())
        assert health.warmed_up and health.queue_capacity == 8 and health.ttl_seconds == 120
        assert health.models["beauty_2dfap"].weights_status == "untrained_init"
        task = submit(client)
        result = wait(client, task["task_id"])
        TaskResult.model_validate(result)
        assert result["status"] == "completed" and result["scores"]["weights_ready"] is False
        assert app.state.service.tasks[task["task_id"]].history == ["queued", "processing", "scores_ready", "completed"]
        artifact = client.get(result["original_image_url"])
        assert artifact.status_code == 200 and artifact.headers["cache-control"] == "no-store"
        assert app.state.service.tasks[task["task_id"]].image is None


@pytest.mark.parametrize("category", ["dog", "cat", "monkey", "doll", "cartoon", "pattern", "scenery"])
def test_nonhuman_contract_never_scores(category):
    with TestClient(create_app(pipeline=ControlledPipeline(rejection="NO_HUMAN_FACE"))) as client:
        result = wait(client, submit(client)["task_id"])
        assert result["rejection"]["code"] == "NO_HUMAN_FACE"
        assert result["rejection"]["message"] == "未识别到人脸"
        assert all(result[key] is None for key in ("scores", "text_analysis", "original_image_url"))


def test_multiface_selection_and_wrong_ids():
    app = create_app(pipeline=ControlledPipeline(count=2))
    with TestClient(app) as client:
        task = submit(client)
        result = wait(client, task["task_id"])
        assert result["status"] == "need_selection" and len(result["selection"]["candidates"]) == 2
        route = f"/api/v1/tasks/{task['task_id']}/select-face"
        for image_id, face_id in [("wrong", "face_0"), (task["image_id"], "face_999")]:
            assert client.post(route, json={"image_id": image_id, "face_id": face_id}).status_code == 400
        assert client.post(route, json={"image_id": task["image_id"], "face_id": "face_1"}).status_code == 202
        result = wait(client, task["task_id"])
        assert result["target_face"]["face_id"] == "face_1"


def test_gate_failure_and_post_text_failure_are_separate():
    with TestClient(create_app(pipeline=ControlledPipeline(rejection="GATE_UNAVAILABLE"))) as client:
        result = wait(client, submit(client)["task_id"])
        assert result["status"] == "rejected" and result["scores"] is None
    with TestClient(create_app(pipeline=ControlledPipeline(text_error=True))) as client:
        result = wait(client, submit(client)["task_id"])
        assert result["status"] == "partial_error" and result["scores"] and result["original_image_url"]
        assert set(result["text_analysis"]["fields"]) == set(FIELD_LABELS)
        assert result["text_analysis"]["narrative"] == "画面中的人物表观年龄约 30 岁，颜值评分为 3.0 分（满分 5.0 分）。"


def test_size_decode_and_content_type():
    with TestClient(create_app(pipeline=ControlledPipeline())) as client:
        assert client.post("/api/v1/tasks", content=b"x" * 8_388_609, headers={"Content-Type": "image/jpeg"}).status_code == 413
        assert client.post("/api/v1/tasks", content=b"garbage", headers={"Content-Type": "image/png"}).json()["error_code"] == "INVALID_IMAGE"
        assert client.post("/api/v1/tasks", content=photo(), headers={"Content-Type": "multipart/form-data"}).status_code == 400


def test_delete_ttl_and_no_resurrection():
    blocker = threading.Event()
    app = create_app(settings=replace(Settings(), ttl_seconds=.25, cleanup_interval=.02), pipeline=ControlledPipeline(block=blocker))
    with TestClient(app) as client:
        task = submit(client)
        route = f"/api/v1/tasks/{task['task_id']}"
        assert client.delete(route).json()["deleted"]
        assert client.get(route).status_code in (404, 410)
        second = submit(client)
        time.sleep(.3)
        assert client.get(f"/api/v1/tasks/{second['task_id']}").status_code in (404, 410)
        blocker.set()
        time.sleep(.05)
        assert not app.state.service.tasks and not app.state.service.queue


def test_bounded_queue_idempotency_and_single_worker():
    blocker = threading.Event()
    pipeline = ControlledPipeline(block=blocker)
    app = create_app(pipeline=pipeline)
    with TestClient(app) as client:
        first = submit(client, params={"client_request_id": "same"})
        again = submit(client, params={"client_request_id": "same"})
        assert first["task_id"] == again["task_id"]
        wait(client, first["task_id"], states=("processing",))
        for _ in range(8):
            submit(client)
        response = client.post("/api/v1/tasks", content=photo(), headers={"Content-Type": "image/png"})
        assert response.status_code == 429
        blocker.set()
        wait(client, first["task_id"])
        assert len(set(pipeline.thread_ids)) == 1
