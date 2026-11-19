import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, GateRejected
from app.imaging import decode_image, jpeg
from app.main import create_app
from app.schemas import Health, TaskResult

pytestmark = pytest.mark.live


def read(path):
    return decode_image(path.read_bytes(), Settings()).bgr


def wait_live(client, task_id, stop=("completed", "partial_error", "rejected", "need_selection")):
    deadline = time.monotonic() + 110
    seen = []
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200, response.text
        result = response.json()
        if not seen or seen[-1] != result["status"]:
            seen.append(result["status"])
        if result["status"] in stop:
            return result, seen
        time.sleep(.02)
    raise AssertionError(f"live task timeout, states: {seen}")


def test_real_gate_faces_animals_patterns(live_pipeline):
    p = live_pipeline
    w = p.settings.weights
    for name in ("dog.jpg", "cat.png"):
        with pytest.raises(GateRejected) as error:
            p.gate(read(w / "fixtures" / name))
        assert error.value.code == "NO_HUMAN_FACE"
        # Force no candidate detection shortcut: real LLM also sees the animal.
        animal = read(w / "fixtures" / name)
        assert p.llm.classify(animal) in {"NON_HUMAN", "UNKNOWN"}
    pattern = np.repeat(((np.indices((512, 512)).sum(0) // 16) % 2 * 255).astype(np.uint8)[..., None], 3, 2)
    with pytest.raises(GateRejected) as error:
        p.gate(pattern)
    assert error.value.code == "NO_HUMAN_FACE"
    assert len(p.gate(read(w / "warmup.jpg"))) == 1


def test_real_person_pet_and_multiface(live_pipeline):
    p = live_pipeline
    w = p.settings.weights
    person = read(w / "warmup.jpg")[40:200, 150:300]
    person = cv2.resize(person, (400, 400))
    dog = cv2.resize(read(w / "fixtures/dog.jpg"), (400, 400))
    mixed = np.concatenate([person, dog], axis=1)
    humans = p.gate(mixed)
    assert len(humans) == 1 and humans[0]["bbox_px"][0] < 400
    other = read(w / "fixtures/grace_hopper.jpg")[80:380, 130:400]
    pair = np.concatenate([person, cv2.resize(other, (400, 400))], axis=1)
    assert len(p.gate(pair)) == 2


def test_real_api_complete_and_selection(live_pipeline):
    # Reuse resident real models; service owns all inference during this test.
    p = live_pipeline
    load, warmup, close = p.load, p.warmup, p.close
    p.load = lambda: None
    p.warmup = lambda: None
    p.close = lambda: None
    try:
        app = create_app(pipeline=p)
        with TestClient(app) as client:
            health = Health.model_validate(client.get("/api/v1/health").json())
            assert health.warmed_up
            assert health.models["face_gate"].version == "OpenGVLab/InternVL3-2B-hf"
            assert health.models["description"].loaded
            assert health.models["description"].version == "OpenGVLab/InternVL3-2B-hf + LoRA"
            assert p.llm.processor.image_processor.max_patches == 1
            assert health.models["beauty_2dfap"].weights_status == "trained_fold1"
            assert health.models["beauty_2dfap"].weights_ready is True
            raw = (p.settings.weights / "warmup.jpg").read_bytes()
            response = client.post("/api/v1/tasks", content=raw, headers={"Content-Type": "image/png"})
            assert response.status_code == 202
            task = response.json()
            result, _ = wait_live(client, task["task_id"])
            TaskResult.model_validate(result)
            assert result["status"] == "completed", result
            assert result["scores"]["weights_ready"] is True
            assert result["scores"]["weights_status"] == "trained_fold1"
            assert result["model_versions"]["age"].endswith("(appa_real_pretrained)")
            assert result["model_versions"]["text"] == health.models["description"].version
            history = app.state.service.tasks[task["task_id"]].history
            assert history == ["queued", "processing", "scores_ready", "completed"]
            artifact = client.get(result["original_image_url"])
            assert artifact.status_code == 200 and artifact.content[:2] == b"\xff\xd8"
            assert artifact.headers["cache-control"] == "no-store"
            person = read(p.settings.weights / "warmup.jpg")[40:200, 150:300]
            other = read(p.settings.weights / "fixtures/grace_hopper.jpg")[80:380, 130:400]
            pair = np.concatenate([cv2.resize(person, (400, 400)), cv2.resize(other, (400, 400))], axis=1)
            task = client.post("/api/v1/tasks", content=jpeg(pair), headers={"Content-Type": "image/jpeg"}).json()
            result, _ = wait_live(client, task["task_id"])
            assert result["status"] == "need_selection"
            selected = result["selection"]["candidates"][1]["face_id"]
            assert client.post(f"/api/v1/tasks/{task['task_id']}/select-face", json={"image_id": task["image_id"], "face_id": selected}).status_code == 202
            result, _ = wait_live(client, task["task_id"])
            assert result["status"] == "completed", result
            assert result["target_face"]["face_id"] == selected
            for name in ("dog.jpg", "cat.png"):
                task = client.post("/api/v1/tasks", content=(p.settings.weights / "fixtures" / name).read_bytes(), headers={"Content-Type": "application/octet-stream"}).json()
                result, _ = wait_live(client, task["task_id"])
                assert result["rejection"]["message"] == "未识别到人脸" and result["scores"] is None
    finally:
        p.load, p.warmup, p.close = load, warmup, close
