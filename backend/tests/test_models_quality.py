import json
from io import BytesIO
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from PIL import Image
import torch

from app.config import Settings, GateRejected
from app.imaging import decode_image, match_candidates
from app.vlm import compose_narrative, FIELD_LABELS, parse_gate, parse_fields
from app.numeric import BeautyModel, age_tensor
from app.pipeline import Pipeline
from app.quality import check_quality


def test_exif_and_non_square_mapping():
    image = Image.new("RGB", (2000, 1000))
    exif = image.getexif()
    exif[274] = 6
    stream = BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    result = decode_image(stream.getvalue(), Settings())
    assert result.bgr.shape == (1280, 640, 3)
    assert result.mapping["oriented_size"] == (1000, 2000)
    assert result.mapping["scale_x"] == result.mapping["scale_y"] == .64


def test_bgr_raw_age_input():
    image = np.full((80, 120, 3), [10, 100, 240], np.uint8)
    tensor = age_tensor(image)
    assert tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32
    assert tensor[:, 0, 0].tolist() == [10., 100., 240.]


def test_40_bin_sigmoid_l1():
    model = BeautyModel().eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 40)
    assert torch.allclose(output.sum(1), torch.ones(1))
    assert isinstance(model.classifier[2], torch.nn.Sigmoid)


def test_full_vector_matches_upstream_migration():
    from app.config import WEIGHTS
    source = (WEIGHTS / "upstream/MobileNetv2.py").read_text(encoding="utf-8")
    source = source.replace("from torchvision.models.utils import load_state_dict_from_url", "from torch.hub import load_state_dict_from_url")
    namespace = {}
    exec(compile(source, "upstream/MobileNetv2.py", "exec"), namespace)
    torch.manual_seed(20)
    deployed = BeautyModel().eval()
    reference = namespace["MobileNetV2"](num_classes=40).eval()
    reference.load_state_dict(deployed.state_dict(), strict=True)
    tensor = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        actual, expected = deployed(tensor), reference(tensor)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-8)


def test_nms_one_to_one_and_minimum_side():
    a = {"bbox_px": [10, 10, 110, 110], "score": .9}
    b = {"bbox_px": [15, 15, 115, 115], "score": .8}
    small = {"bbox_px": [150, 150, 180, 180], "score": .9}
    assert len(match_candidates([a, a, small], [b, small], 300, 300)) == 1
    assert not match_candidates([a], [small], 300, 300)


@pytest.mark.parametrize("answer", ["HUMAN", "NON_HUMAN", "UNKNOWN"])
def test_gate_enumerations(answer):
    assert parse_gate(answer) == answer


@pytest.mark.parametrize("answer", ["It is HUMAN", "NON_HUMAN HUMAN", "", "yes"])
def test_unparseable_gate_fails_closed(answer):
    with pytest.raises(GateRejected):
        parse_gate(answer)


def test_human_holding_pet_is_candidate_filtered():
    pipeline = Pipeline(Settings())
    pipeline.detect = Mock(return_value=[{"face_id": "face_0", "bbox_px": [0, 0, 100, 100]}, {"face_id": "face_1", "bbox_px": [100, 0, 200, 100]}])
    pipeline.llm = Mock()
    pipeline.llm.classify.side_effect = ["HUMAN", "NON_HUMAN"]
    humans = pipeline.gate(np.zeros((200, 200, 3), np.uint8))
    assert [c["face_id"] for c in humans] == ["face_0"]
    assert pipeline.llm.classify.call_count == 2


def test_unknown_and_exceptions_never_score():
    pipeline = Pipeline(Settings())
    pipeline.detect = Mock(return_value=[{"face_id": "face_0", "bbox_px": [0, 0, 100, 100]}])
    pipeline.llm = Mock()
    pipeline.llm.classify.return_value = "UNKNOWN"
    with pytest.raises(GateRejected, match="0 human") as error:
        pipeline.gate(np.zeros((100, 100, 3), np.uint8))
    assert error.value.code == "NO_HUMAN_FACE"
    pipeline.llm.classify.side_effect = GateRejected("GATE_UNAVAILABLE", "timeout")
    with pytest.raises(GateRejected) as error:
        pipeline.gate(np.zeros((100, 100, 3), np.uint8))
    assert error.value.code == "GATE_UNAVAILABLE"


def test_text_parser_assembles_short_observations_and_unknown_fields():
    raw = json.dumps({
        "眉部": "帽檐遮住眉毛，眉形无法看清。",
        "眼睛": " 眼裂稍长，两端略收窄。 ",
        "nose": "鼻梁挺直，显得很有气质",
        "mouth": False,
        "contour": "无。",
        "expression": "neutral",
    })
    fields = parse_fields("```json\n" + raw + "\n```")
    assert set(fields) == set(FIELD_LABELS)
    assert fields["eyes"] == {"label": "眼部", "description": "眼裂稍长，两端略收窄。", "visible": True}
    for key in fields.keys() - {"eyes"}:
        assert fields[key] == {"label": FIELD_LABELS[key], "description": "无法判断", "visible": False}


def test_text_parser_distinguishes_abstention_from_invalid_output():
    fields = parse_fields(json.dumps(dict.fromkeys(FIELD_LABELS)))
    assert all(not field["visible"] for field in fields.values())
    for raw in ('{"nose": "straight"}', '{"contour": {"description": "椭圆"}}', '{"eyes":', '{}'):
        with pytest.raises(ValueError):
            parse_fields(raw)


def test_narrative_uses_scores_and_visible_observations():
    fields = parse_fields('{"brows": null, "eyes": "眼裂偏长。", "nose": "鼻尖较圆！"}')
    for beauty in (2.3, 4.2):
        scores = {"beauty_display": beauty, "apparent_age_display": 30}
        assert compose_narrative(scores, fields) == (
            f"画面中的人物表观年龄约 30 岁，颜值评分为 {beauty:.1f} 分（满分 5.0 分）。眼裂偏长。鼻尖较圆。"
        )


@pytest.mark.parametrize("case,expected", [
    ("dark", "POOR_LIGHTING"),
    ("bright", "POOR_LIGHTING"),
    ("blur", "IMAGE_TOO_BLURRY"),
    ("pose", "POSE_TOO_LARGE"),
    ("reflected", "GATE_UNAVAILABLE"),
])
def test_quality_rejections(case, expected):
    image = np.full((300, 300, 3), 128, np.uint8)
    matrix = np.eye(4)
    if case == "dark":
        image[:] = 5
    elif case == "bright":
        image[:] = 250
    elif case == "pose":
        matrix[:3, :3] = cv2.Rodrigues(np.array([0., .8, 0.]))[0]
    elif case == "reflected":
        matrix[0, 0] = -1.0
    if case != "blur":
        # Add mild texture so non-blur cases pass sharpness check first
        rng = np.random.default_rng(7)
        if case not in ("dark", "bright"):
            image = rng.integers(80, 180, (300, 300, 3), dtype=np.uint8)
    with pytest.raises(GateRejected) as error:
        check_quality(matrix, image, Settings())
    assert error.value.code == expected
