"""数据集审核撤回（revoke）领域逻辑。

撤回必须在**同一个数据库事务**内协调以下状态，避免外部查询同时看到
“已撤回”和“可复用/已发布”两种相互矛盾的状态：

- 允许撤回的审核阶段：审核中（pending_review）或审核通过（approved，含随后
  已通过 /publish 发布、is_published=True 的数据集）；
- 发布标志：is_published / published_at 随撤回收起；
- 版本快照：审核通过时生成的快照标记为失效（is_active=False），历史快照保留留痕；
- 复用关系与复用计数：当前有效版本一旦被其他团队实际复用则禁止撤回；计数始终
  以实际留存的复用记录为准，因此重复请求不会再次扣减；
- 待发送通知：发布时排队、尚未发送的订阅通知随撤回一并删除；已发送通知作为
  历史事实保留。
"""

from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import (
    Dataset,
    DatasetVersion,
    DatasetReview,
    DatasetReuse,
    DatasetSubscription,
    DatasetNotification,
)
from app.schemas.dataset import DatasetReuseResponse, DatasetVersionResponse


# 允许撤回的审核阶段：审核中，或审核通过（含“审核通过即发布”的已发布数据集）
REVOKE_ALLOWED_REVIEW_STATUSES = {"pending_review", "approved"}


def get_active_version(db: Session, dataset: Dataset) -> Optional[DatasetVersion]:
    """返回数据集当前有效版本快照（is_active=True 中版本号/ID 最新者）。"""
    return (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset.id,
            DatasetVersion.is_active == True,  # noqa: E712 - SQLAlchemy 布尔比较
        )
        .order_by(DatasetVersion.version_number.desc(), DatasetVersion.id.desc())
        .first()
    )


def queue_new_version_notifications(
    db: Session, dataset: Dataset, version: DatasetVersion
) -> None:
    """发布新版本时为订阅方排队待发送通知，与发布操作在同一事务内提交。"""
    subscriptions = (
        db.query(DatasetSubscription)
        .filter(
            DatasetSubscription.dataset_id == dataset.id,
            DatasetSubscription.notify_on_new_version == True,  # noqa: E712
        )
        .all()
    )
    for sub in subscriptions:
        db.add(
            DatasetNotification(
                dataset_id=dataset.id,
                dataset_version_id=version.id,
                subscriber_team=sub.subscriber_team,
                contact_person=sub.contact_person,
                new_version=version.version_label,
                message=f"数据集 '{dataset.name}' 已发布新版本 {version.version_label}",
                is_sent=False,
            )
        )


def _latest_review(
    db: Session, dataset: Dataset, action: Optional[str] = None
) -> Optional[DatasetReview]:
    query = db.query(DatasetReview).filter(DatasetReview.dataset_id == dataset.id)
    if action is not None:
        query = query.filter(DatasetReview.action == action)
    return query.order_by(DatasetReview.id.desc()).first()


def _latest_version(db: Session, dataset: Dataset) -> Optional[DatasetVersion]:
    return (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset.id)
        .order_by(DatasetVersion.version_number.desc(), DatasetVersion.id.desc())
        .first()
    )


def _existing_revoke_for_cycle(
    db: Session,
    dataset: Dataset,
    latest_version: Optional[DatasetVersion],
) -> Optional[DatasetReview]:
    """当前版本周期是否已完成撤回（用于幂等）。

    条件：最近一条审核记录是撤回，且其关联版本仍是数据集最新版本——
    一旦又提交审核或创建了更新版本，即视为进入下一周期，旧撤回不再幂等命中。
    """
    last_review = _latest_review(db, dataset)
    if last_review is None or last_review.action != "revoke":
        return None
    if latest_version is None:
        return last_review if last_review.dataset_version_id is None else None
    if last_review.dataset_version_id == latest_version.id:
        return last_review
    return None


def _reuses_of_version(
    db: Session, dataset: Dataset, version: DatasetVersion
) -> List[DatasetReuse]:
    """统计挂在指定有效版本上的实际复用记录。

    显式指定版本的复用按版本匹配；dataset_version_id 为空的历史复用记录视为
    复用了当时（即当前）有效版本，同样参与阻止判断。
    """
    return (
        db.query(DatasetReuse)
        .filter(
            DatasetReuse.dataset_id == dataset.id,
            (DatasetReuse.dataset_version_id == version.id)
            | (DatasetReuse.dataset_version_id.is_(None)),
        )
        .order_by(DatasetReuse.id.desc())
        .all()
    )


def _raise_blocked(
    active_version: Optional[DatasetVersion], reuses: List[DatasetReuse]
) -> None:
    teams = "、".join(sorted({r.reusing_team for r in reuses}))
    label = active_version.version_label if active_version else "未知版本"
    reason = (
        f"当前有效版本 {label} 已存在 {len(reuses)} 条实际复用记录"
        f"（复用团队：{teams}），为保证复用方可追溯，禁止直接撤回；"
        "请先协调复用方迁移，或发布新版本进行替代。"
    )
    raise HTTPException(
        status_code=409,
        detail={
            "blocked": True,
            "reason": reason,
            "reuse_count": len(reuses),
            "reuses": [
                DatasetReuseResponse.model_validate(r).model_dump(mode="json")
                for r in reuses
            ],
            "active_version": (
                DatasetVersionResponse.model_validate(active_version).model_dump(
                    mode="json"
                )
                if active_version is not None
                else None
            ),
        },
    )


def revoke_dataset_review(
    db: Session,
    dataset: Dataset,
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
) -> DatasetReview:
    """在单事务内执行审核撤回，返回流痕用的撤回审核记录。

    阶段不允许时抛 400；有效版本已被复用时抛 409（含阻止原因和当前有效版本）；
    提交失败时回滚全部变更并抛 500。对当前审核周期内已完成的撤回重复调用是幂等的：
    直接返回该次撤回记录，不新增审核记录、不重复扣减复用计数。
    """
    # 幂等检查在最前：当前周期已撤回过（draft 态）时直接返回该次撤回记录
    latest_version = _latest_version(db, dataset)
    if dataset.review_status not in REVOKE_ALLOWED_REVIEW_STATUSES:
        existing_revoke = _existing_revoke_for_cycle(db, dataset, latest_version)
        if dataset.review_status == "draft" and existing_revoke is not None:
            return existing_revoke
        raise HTTPException(
            status_code=400,
            detail=(
                f"当前审核状态 '{dataset.review_status}' 不允许撤回，"
                "仅审核中（pending_review）或审核通过（approved）阶段可撤回"
            ),
        )

    # 本次撤回针对的在途版本快照：审核中为提交时对齐的待审快照，
    # 审核通过后为该次批准关联的当前有效快照（此刻两者都是数据集最新版本）
    target_version = latest_version

    # 当前有效版本一旦被其他团队实际复用，阻止撤回（无论发布标志是否仍在）
    active_version = get_active_version(db, dataset)
    if active_version is not None:
        reuses = _reuses_of_version(db, dataset, active_version)
        if reuses:
            _raise_blocked(active_version, reuses)

    # —— 以下变更全部在同一事务内提交，任一失败整体回滚 ——
    dataset.review_status = "draft"
    dataset.is_published = False
    dataset.published_at = None

    if target_version is not None:
        # 撤回审核通过产生的快照，使其不再作为当前有效版本对外暴露
        target_version.is_active = False
        # 删除该版本已排队但尚未发送的订阅通知；已发送通知无法收回，作为历史保留
        (
            db.query(DatasetNotification)
            .filter(
                DatasetNotification.dataset_id == dataset.id,
                DatasetNotification.dataset_version_id == target_version.id,
                DatasetNotification.is_sent == False,  # noqa: E712
            )
            .delete(synchronize_session=False)
        )

    # 复用记录是其他团队的历史事实，撤回不会删除；计数以留存记录为准重新对账，
    # 因此重复撤回不会再次扣减，也不会产生孤儿计数
    dataset.reuse_count = (
        db.query(func.count(DatasetReuse.id))
        .filter(DatasetReuse.dataset_id == dataset.id)
        .scalar()
        or 0
    )

    review = DatasetReview(
        dataset_id=dataset.id,
        action="revoke",
        reviewer=reviewer,
        review_notes=review_notes,
        dataset_version_id=target_version.id if target_version is not None else None,
    )
    db.add(review)

    try:
        db.flush()
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="撤回失败，数据库事务已回滚，数据集状态未发生改变",
        )

    db.refresh(review)
    return review
