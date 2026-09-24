from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    Dataset, DatasetItem, DatasetReuse,
    DatasetVersion, DatasetReview, DatasetSubscription, DatasetNotification,
    OperationData, Annotation, RobotModel, Scene
)
from app.services.aggregation import compute_dataset_quality_stats
from app.services.review import (
    DatasetRevokeError,
    RevokeResult,
    revoke_dataset,
)
from app.schemas.dataset import (
    DatasetCreate, DatasetUpdate, DatasetResponse,
    DatasetItemAddRequest, DatasetItemRemoveRequest,
    DatasetReuseCreate, DatasetReuseResponse,
    DatasetVersionCreate, DatasetVersionResponse,
    DatasetReviewAction, DatasetReviewResponse,
    DatasetSubscriptionCreate, DatasetSubscriptionResponse,
    DatasetNotificationResponse, DatasetRevokeResponse, ActiveVersionInfo,
)

router = APIRouter()

VALID_REVIEW_ACTIONS = {"submit", "approve", "reject", "revoke"}

REVIEW_TRANSITIONS = {
    "submit": {"draft", "rejected"},
    "approve": {"pending_review"},
    "reject": {"pending_review"},
    "revoke": {"pending_review", "approved"},
}


def recalculate_dataset_stats(db: Session, dataset: Dataset):
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset.id).all()
    op_ids = [item.operation_data_id for item in items]

    dataset.total_items = len(op_ids)

    if op_ids:
        stats = compute_dataset_quality_stats(db, op_ids)
        dataset.success_count = stats.success_count
        dataset.failure_count = stats.failure_count
        dataset.annotation_complete_rate = stats.annotation_complete_rate
        dataset.average_quality_score = stats.average_quality_score
        dataset.data_grade = stats.data_grade

    db.commit()


def _snapshot_version_stats(dataset: Dataset) -> dict:
    return {
        "total_items": dataset.total_items,
        "success_count": dataset.success_count,
        "failure_count": dataset.failure_count,
        "annotation_complete_rate": dataset.annotation_complete_rate,
        "average_quality_score": dataset.average_quality_score,
        "data_grade": dataset.data_grade,
    }


def _create_version_snapshot(db: Session, dataset: Dataset, change_description: str = None, created_by: str = None) -> DatasetVersion:
    version_number = dataset.current_version
    version_label = dataset.version
    stats = _snapshot_version_stats(dataset)

    version = DatasetVersion(
        dataset_id=dataset.id,
        version_number=version_number,
        version_label=version_label,
        change_description=change_description,
        created_by=created_by,
        **stats
    )
    db.add(version)
    db.flush()
    return version


def _notify_subscribers(db: Session, dataset: Dataset, version: DatasetVersion) -> List[dict]:
    """为订阅方落库待发送通知，返回通知内容。

    通知持久化在同一事务内：发布若最终未提交，通知不会残留；
    撤回时这些 pending 通知可在同一事务中取消。
    (dataset, version, subscriber_team) 上有唯一约束，重复触发不会新增。
    """
    subscriptions = db.query(DatasetSubscription).filter(
        DatasetSubscription.dataset_id == dataset.id,
        DatasetSubscription.notify_on_new_version == True
    ).all()

    notifications = []
    for sub in subscriptions:
        message = f"数据集 '{dataset.name}' 已发布新版本 {version.version_label}"
        existing = db.query(DatasetNotification).filter(
            DatasetNotification.dataset_id == dataset.id,
            DatasetNotification.dataset_version_id == version.id,
            DatasetNotification.subscriber_team == sub.subscriber_team,
        ).first()
        if existing is None:
            db.add(DatasetNotification(
                dataset_id=dataset.id,
                dataset_version_id=version.id,
                subscriber_team=sub.subscriber_team,
                contact_person=sub.contact_person,
                notification_type="new_version",
                message=message,
                status="pending",
            ))
        notifications.append({
            "subscriber_team": sub.subscriber_team,
            "contact_person": sub.contact_person,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "new_version": version.version_label,
            "message": message
        })
    return notifications


def _revoke_response(result: RevokeResult) -> DatasetRevokeResponse:
    active = result.active_version
    return DatasetRevokeResponse(
        success=result.success,
        idempotent=result.idempotent,
        reason=result.reason,
        dataset_id=result.dataset_id,
        review_status=result.review_status,
        is_published=result.is_published,
        revoked_version_id=result.revoked_version_id,
        active_version=ActiveVersionInfo(
            version_id=active.id,
            version_number=active.version_number,
            version_label=active.version_label,
            is_published=result.is_published,
            review_status=result.review_status,
        ) if active else ActiveVersionInfo(
            is_published=result.is_published,
            review_status=result.review_status,
        ),
        reuse_count=result.reuse_count,
        reused_by_teams=result.reused_by_teams,
        cancelled_notification_count=result.cancelled_notification_count,
        review=DatasetReviewResponse.model_validate(result.review) if result.review else None,
    )


@router.get("/datasets", response_model=List[DatasetResponse], tags=["数据集管理"])
def list_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    robot_model_id: Optional[int] = Query(None, description="机型ID过滤"),
    scene_id: Optional[int] = Query(None, description="场景ID过滤"),
    skill_id: Optional[int] = Query(None, description="技能ID过滤"),
    is_published: Optional[bool] = Query(None, description="是否已发布"),
    review_status: Optional[str] = Query(None, description="审核状态过滤：draft/pending_review/approved/rejected"),
    owner_team: Optional[str] = Query(None, description="所属团队"),
    data_grade: Optional[str] = Query(None, description="数据等级"),
    keyword: Optional[str] = Query(None, description="搜索关键词"),
    db: Session = Depends(get_db)
):
    query = db.query(Dataset)
    if robot_model_id:
        query = query.filter(Dataset.robot_model_id == robot_model_id)
    if scene_id:
        query = query.filter(Dataset.scene_id == scene_id)
    if skill_id:
        query = query.filter(Dataset.skill_id == skill_id)
    if is_published is not None:
        query = query.filter(Dataset.is_published == is_published)
    if review_status:
        query = query.filter(Dataset.review_status == review_status)
    if owner_team:
        query = query.filter(Dataset.owner_team == owner_team)
    if data_grade:
        query = query.filter(Dataset.data_grade == data_grade)
    if keyword:
        query = query.filter(
            (Dataset.name.contains(keyword)) |
            (Dataset.description.contains(keyword))
        )
    return query.order_by(Dataset.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse, tags=["数据集管理"])
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return dataset


@router.post("/datasets", response_model=DatasetResponse, tags=["数据集管理"])
def create_dataset(data: DatasetCreate, db: Session = Depends(get_db)):
    robot_model = db.query(RobotModel).filter(RobotModel.id == data.robot_model_id).first()
    if not robot_model:
        raise HTTPException(status_code=400, detail="机型不存在")
    scene = db.query(Scene).filter(Scene.id == data.scene_id).first()
    if not scene:
        raise HTTPException(status_code=400, detail="场景不存在")

    dataset_data = data.model_dump(exclude={"operation_data_ids"})
    dataset = Dataset(**dataset_data)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    version = DatasetVersion(
        dataset_id=dataset.id,
        version_number=1,
        version_label=dataset.version,
        change_description="初始版本",
        **_snapshot_version_stats(dataset)
    )
    db.add(version)
    db.commit()
    db.refresh(dataset)

    if data.operation_data_ids:
        items = []
        for op_id in data.operation_data_ids:
            op = db.query(OperationData).filter(OperationData.id == op_id).first()
            if op:
                items.append(DatasetItem(dataset_id=dataset.id, operation_data_id=op_id))
        if items:
            db.bulk_save_objects(items)
            db.commit()
            db.refresh(dataset)
            recalculate_dataset_stats(db, dataset)

    return dataset


@router.put("/datasets/{dataset_id}", response_model=DatasetResponse, tags=["数据集管理"])
def update_dataset(dataset_id: int, data: DatasetUpdate, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    if dataset.review_status == "pending_review":
        raise HTTPException(status_code=400, detail="数据集正在审核中，无法修改")

    update_data = data.model_dump(exclude_unset=True)

    if dataset.review_status in ("approved", "published") and dataset.is_published:
        for field in ("name", "description", "robot_model_id", "scene_id", "skill_id", "tags", "license_info"):
            if field in update_data:
                raise HTTPException(status_code=400, detail="已发布的数据集需先提交新版本才能修改内容")

    if update_data.get("is_published") and not dataset.is_published:
        if dataset.review_status != "approved":
            raise HTTPException(status_code=400, detail="数据集尚未通过审核，无法发布")
        update_data["published_at"] = datetime.now(timezone.utc)

    for field, value in update_data.items():
        setattr(dataset, field, value)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.delete("/datasets/{dataset_id}", tags=["数据集管理"])
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if dataset.review_status == "pending_review":
        raise HTTPException(status_code=400, detail="数据集正在审核中，无法删除")
    db.delete(dataset)
    db.commit()
    return {"message": "删除成功"}


@router.get("/datasets/{dataset_id}/operation-ids", response_model=List[int], tags=["数据集管理"])
def get_dataset_operation_ids(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()
    return [item.operation_data_id for item in items]


@router.post("/datasets/{dataset_id}/items", response_model=DatasetResponse, tags=["数据集管理"])
def add_items_to_dataset(dataset_id: int, req: DatasetItemAddRequest, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    if dataset.is_published and dataset.review_status in ("approved", "published"):
        raise HTTPException(status_code=400, detail="已发布的数据集需先创建新版本才能添加数据")

    existing_ids = set(
        db.query(DatasetItem.operation_data_id)
        .filter(DatasetItem.dataset_id == dataset_id)
        .all()
    )
    existing_ids = {x[0] for x in existing_ids}

    items = []
    for op_id in req.operation_data_ids:
        if op_id in existing_ids:
            continue
        op = db.query(OperationData).filter(OperationData.id == op_id).first()
        if op:
            items.append(DatasetItem(dataset_id=dataset_id, operation_data_id=op_id))

    if items:
        db.bulk_save_objects(items)
        db.commit()

    db.refresh(dataset)
    recalculate_dataset_stats(db, dataset)
    db.refresh(dataset)
    return dataset


@router.delete("/datasets/{dataset_id}/items", response_model=DatasetResponse, tags=["数据集管理"])
def remove_items_from_dataset(dataset_id: int, req: DatasetItemRemoveRequest, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    if dataset.is_published and dataset.review_status in ("approved", "published"):
        raise HTTPException(status_code=400, detail="已发布的数据集需先创建新版本才能移除数据")

    db.query(DatasetItem).filter(
        DatasetItem.dataset_id == dataset_id,
        DatasetItem.operation_data_id.in_(req.operation_data_ids)
    ).delete(synchronize_session=False)
    db.commit()

    db.refresh(dataset)
    recalculate_dataset_stats(db, dataset)
    db.refresh(dataset)
    return dataset


@router.post(
    "/datasets/{dataset_id}/review",
    response_model=DatasetReviewResponse | DatasetRevokeResponse,
    tags=["数据集审核"],
)
def review_dataset(dataset_id: int, req: DatasetReviewAction, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    if req.action not in VALID_REVIEW_ACTIONS:
        raise HTTPException(status_code=400, detail=f"无效的审核动作，允许值：{', '.join(VALID_REVIEW_ACTIONS)}")

    # 撤回的阶段校验（含重复请求幂等）由 review 服务统一处理
    if req.action == "revoke":
        try:
            result = revoke_dataset(db, dataset, reviewer=req.reviewer, review_notes=req.review_notes)
        except DatasetRevokeError as exc:
            db.rollback()
            blocked = exc.result
            raise HTTPException(
                status_code=409 if blocked.reason == "version_reused" else 400,
                detail={
                    "message": blocked.message,
                    "reason": blocked.reason,
                    "result": _revoke_response(blocked).model_dump(mode="json"),
                },
            )
        db.commit()
        db.refresh(dataset)
        if result.review is not None:
            db.refresh(result.review)
        # active_version / dataset 字段以提交后状态为准
        result.review_status = dataset.review_status
        result.is_published = bool(dataset.is_published)
        result.reuse_count = dataset.reuse_count or 0
        return _revoke_response(result)

    if dataset.review_status not in REVIEW_TRANSITIONS.get(req.action, set()):
        raise HTTPException(
            status_code=400,
            detail=f"当前状态 '{dataset.review_status}' 不允许执行 '{req.action}' 操作"
        )

    if req.action == "submit":
        if dataset.total_items == 0:
            raise HTTPException(status_code=400, detail="数据集为空，无法提交审核")
        dataset.review_status = "pending_review"
        dataset.is_published = False

    elif req.action == "approve":
        # 批准只产生版本快照并进入 approved；发布由 /publish 完成，
        # 届时才落库订阅通知，因此存在"已批准未发布"的可撤回阶段。
        dataset.review_status = "approved"
        dataset.is_published = False
        dataset.published_at = None
        review = DatasetReview(
            dataset_id=dataset.id,
            action="approve",
            reviewer=req.reviewer,
            review_notes=req.review_notes,
        )
        db.add(review)
        db.flush()

        version = _create_version_snapshot(
            db, dataset,
            change_description=req.review_notes or "审核通过，生成待发布版本",
            created_by=req.reviewer
        )
        dataset.current_version = version.version_number
        dataset.version = version.version_label

        review.dataset_version_id = version.id
        db.commit()
        db.refresh(review)
        return review

    elif req.action == "reject":
        dataset.review_status = "rejected"
        dataset.is_published = False

    review = DatasetReview(
        dataset_id=dataset.id,
        action=req.action,
        reviewer=req.reviewer,
        review_notes=req.review_notes,
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


@router.get("/datasets/{dataset_id}/reviews", response_model=List[DatasetReviewResponse], tags=["数据集审核"])
def list_dataset_reviews(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return db.query(DatasetReview).filter(
        DatasetReview.dataset_id == dataset_id
    ).order_by(DatasetReview.created_at.desc()).all()


@router.post("/datasets/{dataset_id}/publish", response_model=DatasetResponse, tags=["数据集管理"])
def publish_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if dataset.is_published:
        raise HTTPException(status_code=400, detail="数据集已发布")
    if dataset.review_status != "approved":
        raise HTTPException(status_code=400, detail="数据集尚未通过审核，无法发布")
    if dataset.total_items == 0:
        raise HTTPException(status_code=400, detail="数据集为空，无法发布")

    version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.version_number == dataset.current_version,
        )
        .order_by(DatasetVersion.id.desc())
        .first()
    )
    dataset.is_published = True
    dataset.published_at = datetime.now(timezone.utc)
    if version is not None:
        # 发布与待发送通知在同一事务内提交
        _notify_subscribers(db, dataset, version)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.post("/datasets/{dataset_id}/unpublish", response_model=DatasetResponse, tags=["数据集管理"])
def unpublish_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    # 与撤回共用同一套事务协调：复用拦截、快照失效、通知取消
    try:
        revoke_dataset(db, dataset, reviewer=None, review_notes="通过下架接口撤回")
    except DatasetRevokeError as exc:
        db.rollback()
        blocked = exc.result
        raise HTTPException(
            status_code=409 if blocked.reason == "version_reused" else 400,
            detail=blocked.message,
        )
    db.commit()
    db.refresh(dataset)
    return dataset


@router.post("/datasets/{dataset_id}/versions", response_model=DatasetVersionResponse, tags=["数据集版本"])
def create_dataset_version(dataset_id: int, req: DatasetVersionCreate, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    if dataset.review_status == "pending_review":
        raise HTTPException(status_code=400, detail="数据集正在审核中，无法创建新版本")

    _create_version_snapshot(db, dataset, change_description=req.change_description, created_by=req.created_by)

    dataset.current_version += 1
    new_version_number = dataset.current_version
    version_parts = dataset.version.split(".")
    if len(version_parts) == 2:
        minor = int(version_parts[1]) + 1
        new_label = f"{version_parts[0]}.{minor}"
    else:
        new_label = str(new_version_number)
    dataset.version = new_label
    dataset.review_status = "draft"
    dataset.is_published = False
    dataset.published_at = None

    db.commit()
    db.flush()

    new_version = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_id,
        DatasetVersion.version_number == new_version_number
    ).first()
    db.refresh(new_version)
    return new_version


@router.get("/datasets/{dataset_id}/versions", response_model=List[DatasetVersionResponse], tags=["数据集版本"])
def list_dataset_versions(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_id
    ).order_by(DatasetVersion.version_number.desc()).all()


@router.get("/datasets/{dataset_id}/versions/{version_id}", response_model=DatasetVersionResponse, tags=["数据集版本"])
def get_dataset_version(dataset_id: int, version_id: int, db: Session = Depends(get_db)):
    version = db.query(DatasetVersion).filter(
        DatasetVersion.id == version_id,
        DatasetVersion.dataset_id == dataset_id
    ).first()
    if not version:
        raise HTTPException(status_code=404, detail="版本不存在")
    return version


@router.post("/datasets/{dataset_id}/subscriptions", response_model=DatasetSubscriptionResponse, tags=["数据集订阅"])
def subscribe_dataset(dataset_id: int, req: DatasetSubscriptionCreate, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")

    existing = db.query(DatasetSubscription).filter(
        DatasetSubscription.dataset_id == dataset_id,
        DatasetSubscription.subscriber_team == req.subscriber_team
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="该团队已订阅此数据集")

    subscription = DatasetSubscription(
        dataset_id=dataset_id,
        subscriber_team=req.subscriber_team,
        contact_person=req.contact_person,
        notify_on_new_version=req.notify_on_new_version
    )
    db.add(subscription)
    db.commit()
    db.refresh(subscription)
    return subscription


@router.delete("/datasets/{dataset_id}/subscriptions/{subscription_id}", tags=["数据集订阅"])
def unsubscribe_dataset(dataset_id: int, subscription_id: int, db: Session = Depends(get_db)):
    subscription = db.query(DatasetSubscription).filter(
        DatasetSubscription.id == subscription_id,
        DatasetSubscription.dataset_id == dataset_id
    ).first()
    if not subscription:
        raise HTTPException(status_code=404, detail="订阅不存在")
    db.delete(subscription)
    db.commit()
    return {"message": "取消订阅成功"}


@router.get("/datasets/{dataset_id}/subscriptions", response_model=List[DatasetSubscriptionResponse], tags=["数据集订阅"])
def list_dataset_subscriptions(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return db.query(DatasetSubscription).filter(
        DatasetSubscription.dataset_id == dataset_id
    ).order_by(DatasetSubscription.created_at.desc()).all()


@router.get("/datasets/{dataset_id}/notifications", response_model=List[DatasetNotificationResponse], tags=["数据集订阅"])
def list_dataset_notifications(
    dataset_id: int,
    status: Optional[str] = Query(None, description="通知状态过滤：pending/sent/cancelled"),
    db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    query = db.query(DatasetNotification).filter(DatasetNotification.dataset_id == dataset_id)
    if status:
        query = query.filter(DatasetNotification.status == status)
    return query.order_by(DatasetNotification.created_at.desc()).all()


@router.get("/datasets/{dataset_id}/reuses", response_model=List[DatasetReuseResponse], tags=["数据集复用"])
def list_dataset_reuses(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return db.query(DatasetReuse).filter(DatasetReuse.dataset_id == dataset_id).order_by(DatasetReuse.reuse_date.desc()).all()


@router.post("/dataset-reuses", response_model=DatasetReuseResponse, tags=["数据集复用"])
def create_dataset_reuse(data: DatasetReuseCreate, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == data.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=400, detail="数据集不存在")
    if not dataset.is_published:
        raise HTTPException(status_code=400, detail="数据集尚未发布，无法复用")

    if data.dataset_version_id:
        version = db.query(DatasetVersion).filter(
            DatasetVersion.id == data.dataset_version_id,
            DatasetVersion.dataset_id == data.dataset_id
        ).first()
        if not version:
            raise HTTPException(status_code=400, detail="指定的数据集版本不存在")
        if not version.is_active:
            raise HTTPException(status_code=400, detail="指定的数据集版本已撤回失效，无法复用")
    else:
        # 默认复用当前有效（未撤回）的最新版本
        version = (
            db.query(DatasetVersion)
            .filter(
                DatasetVersion.dataset_id == data.dataset_id,
                DatasetVersion.is_active == True,  # noqa: E712
            )
            .order_by(DatasetVersion.id.desc())
            .first()
        )

    reuse_data = data.model_dump()
    if version is not None:
        reuse_data["dataset_version_id"] = version.id
    reuse = DatasetReuse(**reuse_data)
    db.add(reuse)
    dataset.reuse_count = (dataset.reuse_count or 0) + 1
    db.commit()
    db.refresh(reuse)
    return reuse


@router.get("/dataset-reuses", response_model=List[DatasetReuseResponse], tags=["数据集复用"])
def list_all_dataset_reuses(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    dataset_id: Optional[int] = Query(None, description="数据集ID"),
    reusing_team: Optional[str] = Query(None, description="复用团队"),
    db: Session = Depends(get_db)
):
    query = db.query(DatasetReuse)
    if dataset_id:
        query = query.filter(DatasetReuse.dataset_id == dataset_id)
    if reusing_team:
        query = query.filter(DatasetReuse.reusing_team == reusing_team)
    return query.order_by(DatasetReuse.reuse_date.desc()).offset(skip).limit(limit).all()
