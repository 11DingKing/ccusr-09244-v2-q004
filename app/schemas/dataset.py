from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class DatasetBase(BaseModel):
    name: str = Field(..., max_length=200, description="数据集名称")
    description: Optional[str] = Field(None, description="数据集描述")
    version: str = Field("1.0", max_length=20, description="版本号")
    robot_model_id: int = Field(..., description="机型ID")
    scene_id: int = Field(..., description="场景ID")
    skill_id: Optional[int] = Field(None, description="技能ID")
    owner_team: str = Field(..., max_length=100, description="所属团队")
    contact_person: Optional[str] = Field(None, max_length=100, description="联系人")
    tags: Optional[List[str]] = Field(None, description="标签")
    license_info: Optional[str] = Field(None, max_length=200, description="许可证信息")


class DatasetCreate(DatasetBase):
    operation_data_ids: Optional[List[int]] = Field(None, description="初始包含的作业数据ID列表")


class DatasetUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None
    version: Optional[str] = Field(None, max_length=20)
    robot_model_id: Optional[int] = None
    scene_id: Optional[int] = None
    skill_id: Optional[int] = None
    owner_team: Optional[str] = Field(None, max_length=100)
    contact_person: Optional[str] = Field(None, max_length=100)
    is_published: Optional[bool] = None
    tags: Optional[List[str]] = None
    license_info: Optional[str] = Field(None, max_length=200)
    data_grade: Optional[str] = Field(None, max_length=10)


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    version: str
    robot_model_id: int
    scene_id: int
    skill_id: Optional[int] = None
    owner_team: str
    contact_person: Optional[str] = None
    review_status: str = "draft"
    is_published: bool
    published_at: Optional[datetime] = None
    current_version: int = 1
    total_items: int
    success_count: int
    failure_count: int
    annotation_complete_rate: float
    average_quality_score: Optional[float] = None
    reuse_count: int
    data_grade: Optional[str] = None
    tags: Optional[List[str]] = None
    license_info: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DatasetItemAddRequest(BaseModel):
    operation_data_ids: List[int] = Field(..., description="要添加的作业数据ID列表")


class DatasetItemRemoveRequest(BaseModel):
    operation_data_ids: List[int] = Field(..., description="要移除的作业数据ID列表")


class DatasetReuseBase(BaseModel):
    dataset_id: int = Field(..., description="数据集ID")
    dataset_version_id: Optional[int] = Field(None, description="数据集版本ID")
    reusing_team: str = Field(..., max_length=100, description="复用团队")
    purpose: Optional[str] = Field(None, max_length=200, description="复用目的")
    project_name: Optional[str] = Field(None, max_length=200, description="项目名称")
    notes: Optional[str] = Field(None, description="备注")


class DatasetReuseCreate(DatasetReuseBase):
    pass


class DatasetReuseResponse(BaseModel):
    id: int
    dataset_id: int
    dataset_version_id: Optional[int] = None
    reusing_team: str
    purpose: Optional[str] = None
    project_name: Optional[str] = None
    reuse_date: datetime
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class DataGradeStats(BaseModel):
    grade: str
    count: int
    percentage: float


class SceneStats(BaseModel):
    scene_id: int
    scene_name: str
    scene_category: str
    total_data_count: int
    annotated_count: int
    annotation_complete_rate: float
    success_count: int
    failure_count: int
    dataset_count: int
    total_reuse_count: int
    reuse_rate: float


class RobotModelStats(BaseModel):
    robot_model_id: int
    robot_model_name: str
    manufacturer: str
    total_data_count: int
    annotated_count: int
    annotation_complete_rate: float
    success_count: int
    failure_count: int
    dataset_count: int
    total_reuse_count: int
    reuse_rate: float
    grade_distribution: List[DataGradeStats]


class OverallStats(BaseModel):
    total_operation_data: int
    total_annotated: int
    overall_annotation_rate: float
    total_datasets: int
    total_published_datasets: int
    total_reuse_count: int
    total_success_count: int
    total_failure_count: int
    robot_model_count: int
    scene_count: int
    skill_count: int


class QualityGradeRequest(BaseModel):
    completeness_weight: float = Field(0.5, ge=0, le=1, description="完整度权重")
    annotation_weight: float = Field(0.5, ge=0, le=1, description="标注质量权重")
    grade_a_threshold: float = Field(0.9, ge=0, le=1, description="A级阈值")
    grade_b_threshold: float = Field(0.7, ge=0, le=1, description="B级阈值")
    grade_c_threshold: float = Field(0.5, ge=0, le=1, description="C级阈值")


class DatasetVersionCreate(BaseModel):
    change_description: Optional[str] = Field(None, description="版本变更说明")
    created_by: Optional[str] = Field(None, max_length=100, description="创建人")


class DatasetVersionResponse(BaseModel):
    id: int
    dataset_id: int
    version_number: int
    version_label: str
    change_description: Optional[str] = None
    total_items: int
    success_count: int
    failure_count: int
    annotation_complete_rate: float
    average_quality_score: Optional[float] = None
    data_grade: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime
    is_active: bool = True
    revoked_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DatasetReviewAction(BaseModel):
    action: str = Field(..., description="审核动作：submit/approve/reject/revoke")
    reviewer: Optional[str] = Field(None, max_length=100, description="审核人")
    review_notes: Optional[str] = Field(None, description="审核意见")


class DatasetReviewResponse(BaseModel):
    id: int
    dataset_id: int
    dataset_version_id: Optional[int] = None
    action: str
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetSubscriptionCreate(BaseModel):
    subscriber_team: str = Field(..., max_length=100, description="订阅团队")
    contact_person: Optional[str] = Field(None, max_length=100, description="联系人")
    notify_on_new_version: bool = Field(True, description="新版本时是否通知")


class DatasetSubscriptionResponse(BaseModel):
    id: int
    dataset_id: int
    subscriber_team: str
    contact_person: Optional[str] = None
    notify_on_new_version: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewStatusStats(BaseModel):
    draft: int = 0
    pending_review: int = 0
    approved: int = 0
    rejected: int = 0
    published: int = 0


class DatasetNotificationResponse(BaseModel):
    id: int
    dataset_id: int
    dataset_version_id: int
    subscriber_team: str
    contact_person: Optional[str] = None
    notification_type: str
    message: Optional[str] = None
    status: str
    created_at: datetime
    cancelled_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ActiveVersionInfo(BaseModel):
    """撤回到达终态（被阻止或幂等命中）时返回的当前有效版本。"""

    version_id: Optional[int] = None
    version_number: Optional[int] = None
    version_label: Optional[str] = None
    is_published: bool = False
    review_status: str


class DatasetRevokeResponse(BaseModel):
    success: bool
    idempotent: bool = False
    reason: Optional[str] = None
    dataset_id: int
    review_status: str
    is_published: bool
    revoked_version_id: Optional[int] = None
    active_version: Optional[ActiveVersionInfo] = None
    reuse_count: int = 0
    reused_by_teams: List[str] = []
    cancelled_notification_count: int = 0
    review: Optional[DatasetReviewResponse] = None
