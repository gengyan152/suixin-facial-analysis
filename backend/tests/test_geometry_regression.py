from dataclasses import replace
import cv2
import numpy as np
import pytest

from app.config import Settings,GateRejected
from app.geometry import measure,REGIONS


def mesh():
    # Explicit synthetic geometry, in pixel space, used only for analytic invariants.
    points=np.full((478,2),[200.,230.])
    oval=REGIONS["contour"]
    for i,index in enumerate(oval):
        theta=-np.pi/2+2*np.pi*i/len(oval)
        points[index]=[200+110*np.cos(theta),240+160*np.sin(theta)]
    positions={33:(130,200),133:(170,200),362:(230,200),263:(270,200),159:(150,190),145:(150,210),386:(250,190),374:(250,210),
               52:(145,170),65:(155,170),282:(245,170),295:(255,170),9:(200,160),168:(200,180),6:(200,210),2:(200,260),
               152:(200,400),98:(175,250),327:(225,250),61:(160,310),291:(240,310),0:(200,300),13:(200,308),14:(200,312),17:(200,328),
               234:(90,240),454:(310,240),172:(110,340),397:(290,340)}
    for index,xy in positions.items(): points[index]=xy
    return points


def test_roll_scale_translation_invariance_and_lip_ratio():
    points=mesh()
    rng=np.random.default_rng(8)
    image=rng.integers(60,190,(300,300,3),dtype=np.uint8)
    base,*_=measure(points,np.eye(4),image,Settings())
    angle=.21
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    changed,*_=measure(points@rotation.T*1.8+np.array([100,70]),np.eye(4),image,Settings())
    assert base["upper_to_lower_lip_ratio"]==pytest.approx(.5)
    assert base["mid_to_lower_face_ratio"]==pytest.approx(100/140)
    for key in base.keys()-{"measurement_note","five_eyes"}:
        assert changed[key]==pytest.approx(base[key],abs=1e-6)
    for key in base["five_eyes"]:
        assert changed["five_eyes"][key]==pytest.approx(base["five_eyes"][key])


def test_true_2d_symmetry_detects_vertical_displacement():
    points=mesh()
    image=np.random.default_rng(9).integers(50,190,(300,300,3),dtype=np.uint8)
    base,*_=measure(points,np.eye(4),image,Settings())
    points[263,1]+=8
    changed,*_=measure(points,np.eye(4),image,Settings())
    assert changed["mirror_deviation_norm"]>base["mirror_deviation_norm"]+.005


@pytest.mark.parametrize("case,expected",[("dark","POOR_LIGHTING"),("bright","POOR_LIGHTING"),("blur","IMAGE_TOO_BLURRY"),("pose","POSE_TOO_LARGE"),("degenerate","GATE_UNAVAILABLE"),("flipped","GATE_UNAVAILABLE")])
def test_quality_rejections(case,expected):
    points=mesh()
    image=np.full((300,300,3),128,np.uint8)
    matrix=np.eye(4)
    if case=="dark": image[:]=5
    if case=="bright": image[:]=250
    if case=="pose": matrix[:3,:3]=cv2.Rodrigues(np.array([0.,.8,0.]))[0]
    if case=="degenerate": points[:]=0
    if case=="flipped": points[[33,133]]=points[[133,33]]
    with pytest.raises(GateRejected) as error: measure(points,matrix,image,Settings())
    assert error.value.code==expected
