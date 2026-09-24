"""数据集审核撤回的事务协调逻辑。

撤回不是简单地把 ``review_status`` 改回草稿：允许撤回的阶段、发布标志、
版本快照、复用关系和待发送通知必须在同一个事务内协调变化：

- 仅 ``pending_review`` / ``approved`` 阶段允许撤回；
- 目标版本已被其他团队实际复用时阻止撤回，并返回阻止原因与当前有效版本；
- 撤回成功时发布标志复位、批准产生的版本快照失效、版本指针回退、
  该版本的待发送订阅通知全部取消；
- 重复撤回是幂等的：直接返回上次结果，不再扣减计数或写入任何记录。

本模块只做校验与状态变更（``flush``），由调用方负责 ``commit``，
提交失败时整个事务回滚，不会留下半撤回状态。
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models import (
    Dataset,
    DatasetNotification,
    DatasetReview,
    DatasetReuse,
    DatasetVersion,
)

# 允许发起撤回的审核阶段
REVOKABLE_STATUSES = ("pending_review", "approved")

# 撤回阻止原因
REASON_INVALID_STATUS = "invalid_status"
REASON_VERSION_REUSED = "version_reused"
REASON_ALREADY_REVOKED = "already_revoked"


@dataclass
class RevokeResult:
    success: bool
    reason: Optional[str]
    message: str
    dataset_id: int
    review_status: str
    is_published: bool
    idempotent: bool = False
    revoked_version_id: Optional[int] = None
    active_version: Optional[DatasetVersion] = None
    reuse_count: int = 0
    reused_by_teams: List[str] = field(default_factory=list)
    cancelled_notification_count: int = 0
    review: Optional[DatasetReview] = None


class DatasetRevokeError(Exception):
    """撤回被业务规则阻止（阶段不允许或版本已被复用）。"""

    def __init__(self, result: RevokeResult):
        super().__init__(result.message)
        self.result = result


def latest_active_version(db: Session, dataset_id: int) -> Optional[DatasetVersion]:
    """当前指向的最新有效版本快照（按写入顺序，而非可能重复的版本号）。"""
    return (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.is_active == True,  # noqa: E712
        )
        .order_by(DatasetVersion.id.desc())
        .first()
    )


def _last_review(db: Session, dataset_id: int) -> Optional[DatasetReview]:
    return (
        db.query(DatasetReview)
        .filter(DatasetReview.dataset_id == dataset_id)
        .order_by(DatasetReview.id.desc())
        .first()
    )


def _reuses_of_version(db: Session, dataset_id: int, version_id: int) -> List[DatasetReuse]:
    return (
        db.query(DatasetReuse)
        .filter(
            DatasetReuse.dataset_id == dataset_id,
            DatasetReuse.dataset_version_id == version_id,
        )
        .all()
    )


def _blocked(result: RevokeResult) -> DatasetRevokeError:
    return DatasetRevokeError(result)


def revoke_dataset(
    db: Session,
    dataset: Dataset,
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
) -> RevokeResult:
    """在当前事务内执行撤回，不提交；失败时抛出 :class:`DatasetRevokeError`。

    抛出异常时事务保持可回滚状态，调用方可直接 rollback 或交由依赖处理。
    """
    now = datetime.now(timezone.utc)
    current_active = latest_active_version(db, dataset.id)

    def terminal_result(message: str, idempotent: bool, reason: Optional[str]) -> RevokeResult:
        db.refresh(dataset)
        return RevokeResult(
            success=True,
            idempotent=idempotent,
            reason=reason,
            message=message,
            dataset_id=dataset.id,
            review_status=dataset.review_status,
            is_published=bool(dataset.is_published),
            active_version=latest_active_version(db, dataset.id),
            reuse_count=dataset.reuse_count or 0,
        )

    # 已在草稿态：区分"从未提交"与"撤回后的重复请求"，后者幂等返回
    if dataset.review_status == "draft":
        last_review = _last_review(db, dataset.id)
        if last_review is not None and last_review.action == "revoke":
            return terminal_result(
                "数据集已撤回，重复请求不产生任何变更",
                idempotent=True,
                reason=REASON_ALREADY_REVOKED,
            )
        raise _blocked(RevokeResult(
            success=False,
            reason=REASON_INVALID_STATUS,
            message="当前为草稿状态，未进入审核流程，无需撤回",
            dataset_id=dataset.id,
            review_status=dataset.review_status,
            is_published=bool(dataset.is_published),
            active_version=current_active,
            reuse_count=dataset.reuse_count or 0,
        ))

    if dataset.review_status not in REVOKABLE_STATUSES:
        raise _blocked(RevokeResult(
            success=False,
            reason=REASON_INVALID_STATUS,
            message=f"当前状态 '{dataset.review_status}' 不允许撤回，"
                    f"仅审核中(pending_review)或审核通过(approved)阶段可撤回",
            dataset_id=dataset.id,
            review_status=dataset.review_status,
            is_published=bool(dataset.is_published),
            active_version=current_active,
            reuse_count=dataset.reuse_count or 0,
        ))

    # 已被其他团队实际复用的版本不得撤回
    reused_teams: List[str] = []
    if current_active is not None:
        reuses = _reuses_of_version(db, dataset.id, current_active.id)
        reused_teams = sorted({r.reusing_team for r in reuses})
        if reuses:
            raise _blocked(RevokeResult(
                success=False,
                reason=REASON_VERSION_REUSED,
                message=f"当前有效版本 {current_active.version_label} 已被 "
                        f"{len(reused_teams)} 个团队复用，不能直接撤回",
                dataset_id=dataset.id,
                review_status=dataset.review_status,
                is_published=bool(dataset.is_published),
                active_version=current_active,
                reuse_count=dataset.reuse_count or 0,
                reused_by_teams=reused_teams,
            ))

    cancelled = 0
    revoked_version_id: Optional[int] = None

    if dataset.review_status == "approved" and current_active is not None:
        # 批准产生的快照失效，指针回退到上一个仍有效的快照
        target = current_active
        target.is_active = False
        target.revoked_at = now
        revoked_version_id = target.id

        fallback = (
            db.query(DatasetVersion)
            .filter(
                DatasetVersion.dataset_id == dataset.id,
                DatasetVersion.is_active == True,  # noqa: E712
                DatasetVersion.id < target.id,
            )
            .order_by(DatasetVersion.id.desc())
            .first()
        )
        if fallback is not None:
            dataset.current_version = fallback.version_number
            dataset.version = fallback.version_label

        # 取消该版本尚未发送的订阅通知（已发送的不动）
        pending = (
            db.query(DatasetNotification)
            .filter(
                DatasetNotification.dataset_id == dataset.id,
                DatasetNotification.dataset_version_id == target.id,
                DatasetNotification.status == "pending",
            )
            .all()
        )
        for note in pending:
            note.status = "cancelled"
            note.cancelled_at = now
            cancelled += 1
    # pending_review 阶段批准尚未发生：没有快照与通知需要处理

    dataset.review_status = "draft"
    dataset.is_published = False
    dataset.published_at = None

    review = DatasetReview(
        dataset_id=dataset.id,
        action="revoke",
        dataset_version_id=revoked_version_id,
        reviewer=reviewer,
        review_notes=review_notes,
    )
    db.add(review)
    db.flush()

    return RevokeResult(
        success=True,
        reason=None,
        message="撤回成功，数据集已回到草稿状态",
        dataset_id=dataset.id,
        review_status=dataset.review_status,
        is_published=False,
        revoked_version_id=revoked_version_id,
        active_version=latest_active_version(db, dataset.id),
        reuse_count=dataset.reuse_count or 0,
        reused_by_teams=reused_teams,
        cancelled_notification_count=cancelled,
        review=review,
    )
