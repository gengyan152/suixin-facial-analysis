from dataclasses import replace
from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
import torch

from app.config import Settings,GateRejected
from app.imaging import decode_image,match_candidates
from app.facellm import parse_gate,parse_fields,controlled_attribute
from app.numeric import BeautyModel,age_tensor
from app.pipeline import Pipeline


def test_exif_and_non_square_mapping():
    image=Image.new("RGB",(2000,1000))
    exif=image.getexif();exif[274]=6
    stream=BytesIO();image.save(stream,format="JPEG",exif=exif)
    result=decode_image(stream.getvalue(),Settings())
    assert result.bgr.shape==(1280,640,3)
    assert result.mapping["oriented_size"]==(1000,2000)
    assert result.mapping["scale_x"]==result.mapping["scale_y"]==.64


def test_bgr_raw_age_input():
    image=np.full((80,120,3),[10,100,240],np.uint8)
    tensor=age_tensor(image)
    assert tensor.shape==(3,224,224) and tensor.dtype==torch.float32
    assert tensor[:,0,0].tolist()==[10.,100.,240.]


def test_40_bin_sigmoid_l1():
    model=BeautyModel().eval()
    with torch.no_grad():
        output=model(torch.zeros(1,3,224,224))
    assert output.shape==(1,40)
    assert torch.allclose(output.sum(1),torch.ones(1))
    assert isinstance(model.classifier[2],torch.nn.Sigmoid)


def test_full_vector_matches_upstream_migration():
    from app.config import WEIGHTS
    source=(WEIGHTS/"upstream/MobileNetv2.py").read_text(encoding="utf-8")
    source=source.replace("from torchvision.models.utils import load_state_dict_from_url","from torch.hub import load_state_dict_from_url")
    namespace={}
    exec(compile(source,"upstream/MobileNetv2.py","exec"),namespace)
    torch.manual_seed(20)
    deployed=BeautyModel().eval()
    reference=namespace["MobileNetV2"](num_classes=40).eval()
    reference.load_state_dict(deployed.state_dict(),strict=True)
    tensor=torch.rand(2,3,224,224)
    with torch.no_grad():
        actual,expected=deployed(tensor),reference(tensor)
    torch.testing.assert_close(actual,expected,rtol=0,atol=1e-8)


def test_nms_one_to_one_and_minimum_side():
    a={"bbox_px":[10,10,110,110],"score":.9}
    b={"bbox_px":[15,15,115,115],"score":.8}
    small={"bbox_px":[150,150,180,180],"score":.9}
    assert len(match_candidates([a,a,small],[b,small],300,300))==1
    assert not match_candidates([a],[small],300,300)


@pytest.mark.parametrize("answer",["HUMAN","NON_HUMAN","UNKNOWN"])
def test_gate_enumerations(answer):
    assert parse_gate(answer)==answer


@pytest.mark.parametrize("answer",["It is HUMAN", "NON_HUMAN HUMAN", "", "yes"])
def test_unparseable_gate_fails_closed(answer):
    with pytest.raises(GateRejected): parse_gate(answer)


def test_human_holding_pet_is_candidate_filtered():
    pipeline=Pipeline(Settings())
    pipeline.detect=Mock(return_value=[{"face_id":"face_0","bbox_px":[0,0,100,100]}, {"face_id":"face_1","bbox_px":[100,0,200,100]}])
    pipeline.llm=Mock()
    pipeline.llm.classify.side_effect=["HUMAN","NON_HUMAN"]
    humans=pipeline.gate(np.zeros((200,200,3),np.uint8))
    assert [c["face_id"] for c in humans]==["face_0"]
    assert pipeline.llm.classify.call_count==2


def test_unknown_and_exceptions_never_score():
    pipeline=Pipeline(Settings())
    pipeline.detect=Mock(return_value=[{"face_id":"face_0","bbox_px":[0,0,100,100]}])
    pipeline.llm=Mock()
    pipeline.llm.classify.return_value="UNKNOWN"
    with pytest.raises(GateRejected,match="0 human") as error: pipeline.gate(np.zeros((100,100,3),np.uint8))
    assert error.value.code=="NO_HUMAN_FACE"
    pipeline.llm.classify.side_effect=GateRejected("GATE_UNAVAILABLE","timeout")
    with pytest.raises(GateRejected) as error: pipeline.gate(np.zeros((100,100,3),np.uint8))
    assert error.value.code=="GATE_UNAVAILABLE"


def test_text_parser_six_fields_strict_bool():
    import json
    fields={key:{"label":key,"description":"轮廓清晰","visible":True} for key in ("brows","eyes","nose","mouth","contour","expression")}
    assert len(parse_fields("```json\n"+json.dumps(fields)+"\n```"))==6
    fields["eyes"]["visible"]="false"
    with pytest.raises(ValueError): parse_fields(json.dumps(fields))


def test_controlled_description_drops_personality_and_handles_negation():
    assert controlled_attribute("mouth","The lips are closed, and the teeth are not visible.")=="CLOSED"
    assert controlled_attribute("mouth","The lips are parted, revealing the upper teeth.")=="TEETH_VISIBLE"
    assert controlled_attribute("brows","The eyebrows are arched.")=="ARCHED"
    assert controlled_attribute("eyes","He is wise and kind.")=="UNKNOWN"
