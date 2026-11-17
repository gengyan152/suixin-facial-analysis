from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

Status = Literal["queued","processing","rejected","need_selection","scores_ready","explanation_ready","completed","partial_error"]


class ModelHealth(BaseModel):
    model_config=ConfigDict(extra="allow")
    loaded: bool
    version: str | None = None
    weights_status: str | None = None
    precision: str | None = None
    source: str | None = None


class Health(BaseModel):
    status: str
    warmed_up: bool
    device: str
    queue_length: int
    queue_capacity: int
    ttl_seconds: float
    models: dict[str,ModelHealth]
    startup_error: str | None = None
    warmup_metrics: dict[str,float] | None = None


class Submission(BaseModel):
    task_id: str
    image_id: str
    status: Status
    queue_position: int
    created_at: str


class FaceSelection(BaseModel):
    image_id: str = Field(min_length=1,max_length=100)
    face_id: str = Field(min_length=1,max_length=100)


class Pose(BaseModel):
    yaw: float
    pitch: float
    roll: float


class Candidate(BaseModel):
    face_id: str
    bbox_px: list[int] = Field(min_length=4,max_length=4)
    bbox_norm: list[float] = Field(min_length=4,max_length=4)
    score: float


class Selection(BaseModel):
    image_width: int
    image_height: int
    preview_image_url: str
    candidates: list[Candidate]


class TargetFace(BaseModel):
    face_id: str
    image_width: int
    image_height: int
    bbox_px: list[int] = Field(min_length=4,max_length=4)
    bbox_norm: list[float] = Field(min_length=4,max_length=4)
    pose_deg: Pose
    blur_laplacian_var: float
    brightness_mean: float


class Scores(BaseModel):
    beauty_score: float
    beauty_display: float
    scale_min: int
    scale_max: int
    distribution_label: str
    rating_probabilities: list[float] = Field(min_length=5,max_length=5)
    distribution_std: float
    apparent_age: float
    apparent_age_display: int
    weights_ready: bool
    weights_status: str


class FiveEyes(BaseModel):
    left_eye_to_face_ratio: float
    intercanthal_to_face_ratio: float
    right_eye_to_face_ratio: float
    inner_canthus_to_eye_width_ratio: float


class Geometry(BaseModel):
    measurement_note: str
    mid_to_lower_face_ratio: float
    five_eyes: FiveEyes
    brow_to_eye_ratio: float
    nose_wing_to_intercanthal_ratio: float
    upper_to_lower_lip_ratio: float
    mirror_symmetry_score: float
    mirror_deviation_norm: float


class RegionSensitivity(BaseModel):
    region_key: str
    region_name: str
    perturbed_score: float
    delta_score: float
    abs_delta: float
    relative_to_random: float
    is_sensitive: bool


class Explanation(BaseModel):
    original_image_url: str
    landmarks_overlay_url: str
    gradcam_overlay_url: str
    random_baseline_mean_abs_delta: float
    region_sensitivity: list[RegionSensitivity]


class TextField(BaseModel):
    label: str
    description: str
    visible: bool


class SixFields(BaseModel):
    brows: TextField
    eyes: TextField
    nose: TextField
    mouth: TextField
    contour: TextField
    expression: TextField


class TextAnalysis(BaseModel):
    status: str
    error_message: str | None
    fields: SixFields
    narrative: str | None = None


class GenderView(BaseModel):
    title: str
    focus_keys: list[str]
    summary: str
    evidence_note: str


class GenderViews(BaseModel):
    general: GenderView
    female: GenderView
    male: GenderView


class Rejection(BaseModel):
    code: str
    message: str
    detail: str
    preview_image_url: str | None = None


class TaskResult(BaseModel):
    task_id: str
    image_id: str
    status: Status
    is_offline_replay: bool = False
    model_versions: dict[str,str] | None = None
    generated_at: str | None = None
    rejection: Rejection | None = None
    selection: Selection | None = None
    target_face: TargetFace | None = None
    scores: Scores | None = None
    geometry: Geometry | None = None
    explanation: Explanation | None = None
    text_analysis: TextAnalysis | None = None
    gender_views: GenderViews | None = None
    ethics_notice: str
    timings_ms: dict[str,float]
