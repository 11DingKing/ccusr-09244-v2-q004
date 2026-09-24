"""数据集撤回（revoke）语义的端到端测试。

覆盖：
- 审核中撤回（pending_review）
- 审核通过但未发布时撤回（approved, is_published=False）
- 已发布且无复用时撤回：发布标志、版本快照、待发送通知在同一事务协调
- 已被其他团队复用时撤回被阻止，返回原因与当前有效版本
- 数据库提交失败时整体回滚，不产生半撤回状态
- 重复撤回幂等：不新增审核记录、不重复扣减或新增记录
"""
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.models import (
    Base,
    Dataset,
    DatasetItem,
    DatasetNotification,
    DatasetReuse,
    DatasetReview,
    DatasetSubscription,
    DatasetVersion,
    OperationData,
    RobotModel,
    Scene,
    Skill,
)
from main import app


class CommitFailureController:
    """测试开关：置位后下一次 Session.commit 抛出异常。"""

    armed = False


class FailingSession(Session):
    def commit(self):
        if CommitFailureController.armed:
            CommitFailureController.armed = False
            raise RuntimeError("simulated commit failure")
        return super().commit()


@pytest.fixture()
def db_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = __import__(
        "sqlalchemy.orm", fromlist=["sessionmaker"]
    ).sessionmaker(bind=engine, class_=FailingSession, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = factory()
        try:
            yield db
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


@pytest.fixture()
def client(db_factory):
    # 提交失败场景需要拿到 500 响应而不是让异常在客户端重抛
    return TestClient(app, raise_server_exceptions=False)


def _make_dataset(db, *, published=False, approved=False, with_reuse=False):
    token = uuid.uuid4().hex[:8]
    robot = RobotModel(name=f"RM-{token}", manufacturer="M")
    scene = Scene(name=f"S-{token}", category="c")
    skill = Skill(name=f"SK-{token}", category="c")
    db.add_all([robot, scene, skill])
    db.flush()
    op = OperationData(
        robot_model_id=robot.id,
        scene_id=scene.id,
        skill_id=skill.id,
        motion_trajectory={"waypoints": []},
        perception_records={"camera_images_captured": 1},
        timestamp_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        timestamp_end=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )
    db.add(op)
    db.flush()

    dataset = Dataset(
        name=f"DS-{token}",
        version="1.0",
        robot_model_id=robot.id,
        scene_id=scene.id,
        skill_id=skill.id,
        owner_team="owner",
        total_items=1,
        review_status="draft",
        is_published=False,
    )
    db.add(dataset)
    db.flush()
    db.add(DatasetItem(dataset_id=dataset.id, operation_data_id=op.id))
    db.add(DatasetVersion(
        dataset_id=dataset.id,
        version_number=1,
        version_label="1.0",
        change_description="初始版本",
        total_items=1,
    ))
    db.add(DatasetSubscription(
        dataset_id=dataset.id,
        subscriber_team="订阅组",
        contact_person="联系人",
        notify_on_new_version=True,
    ))
    db.commit()

    if approved or published:
        dataset.review_status = "pending_review"
        db.commit()
        approved_version = DatasetVersion(
            dataset_id=dataset.id,
            version_number=dataset.current_version,
            version_label=dataset.version,
            change_description="审核通过",
            total_items=1,
        )
        review = DatasetReview(dataset_id=dataset.id, action="approve")
        db.add_all([approved_version, review])
        db.flush()
        review.dataset_version_id = approved_version.id
        dataset.review_status = "approved"
        dataset.is_published = bool(published)
        dataset.published_at = datetime.now(timezone.utc) if published else None
        if published:
            db.add(DatasetNotification(
                dataset_id=dataset.id,
                dataset_version_id=approved_version.id,
                subscriber_team="订阅组",
                contact_person="联系人",
                notification_type="new_version",
                message="新版本 1.0 已发布",
                status="pending",
            ))
        db.commit()

        if with_reuse:
            db.add(DatasetReuse(
                dataset_id=dataset.id,
                dataset_version_id=approved_version.id,
                reusing_team="外部算法组",
                purpose="模型训练",
                project_name="P-X",
            ))
            dataset.reuse_count = (dataset.reuse_count or 0) + 1
            db.commit()

    db.refresh(dataset)
    return dataset.id


def _review(client, dataset_id, action, reviewer="owner", expected=None):
    resp = client.post(
        f"/api/v1/datasets/{dataset_id}/review",
        json={"action": action, "reviewer": reviewer},
    )
    if expected is not None:
        assert resp.status_code == expected, resp.text
    return resp


def _approved_version(db, ds_id):
    return db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == ds_id,
        DatasetVersion.change_description == "审核通过",
    ).one()


def test_revoke_during_pending_review(client, db_factory):
    db = db_factory()
    ds_id = _make_dataset(db)
    db.close()

    _review(client, ds_id, "submit", expected=200)
    body = _review(client, ds_id, "revoke", expected=200).json()

    assert body["success"] is True
    assert body["review_status"] == "draft"
    assert body["is_published"] is False
    assert body["revoked_version_id"] is None
    assert body["cancelled_notification_count"] == 0

    db = db_factory()
    ds = db.get(Dataset, ds_id)
    assert ds.review_status == "draft"
    assert ds.is_published is False
    assert ds.published_at is None
    # 审核中尚未产生批准快照，初始快照保持有效
    versions = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == ds_id).all()
    assert all(v.is_active for v in versions)
    assert db.query(DatasetReview).filter(
        DatasetReview.dataset_id == ds_id, DatasetReview.action == "revoke"
    ).count() == 1
    db.close()


def test_revoke_approved_not_published(client, db_factory):
    db = db_factory()
    ds_id = _make_dataset(db, approved=True)
    db.close()

    body = _review(client, ds_id, "revoke", expected=200).json()
    assert body["success"] is True
    assert body["review_status"] == "draft"
    assert body["is_published"] is False

    db = db_factory()
    ds = db.get(Dataset, ds_id)
    assert ds.review_status == "draft"
    assert ds.is_published is False
    approved_version = _approved_version(db, ds_id)
    assert approved_version.is_active is False
    assert approved_version.revoked_at is not None
    initial = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == ds_id,
        DatasetVersion.change_description == "初始版本",
    ).one()
    assert initial.is_active is True
    # 版本指针回退到仍有效的初始快照
    assert ds.current_version == initial.version_number
    assert ds.version == initial.version_label
    # 未发布过，不存在任何待发送通知
    assert db.query(DatasetNotification).filter(
        DatasetNotification.dataset_id == ds_id
    ).count() == 0
    db.close()


def test_revoke_published_without_reuse_cancels_notifications(client, db_factory):
    db = db_factory()
    ds_id = _make_dataset(db, published=True)
    db.close()

    body = _review(client, ds_id, "revoke", expected=200).json()
    assert body["success"] is True
    assert body["is_published"] is False
    assert body["cancelled_notification_count"] == 1

    db = db_factory()
    ds = db.get(Dataset, ds_id)
    assert ds.is_published is False
    assert ds.published_at is None
    assert ds.review_status == "draft"
    assert (ds.reuse_count or 0) == 0
    assert _approved_version(db, ds_id).is_active is False

    notes = db.query(DatasetNotification).filter(
        DatasetNotification.dataset_id == ds_id
    ).all()
    assert len(notes) == 1
    assert notes[0].status == "cancelled"
    assert notes[0].cancelled_at is not None

    active = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == ds_id,
        DatasetVersion.is_active == True,  # noqa: E712
    ).all()
    assert {v.change_description for v in active} == {"初始版本"}
    db.close()


def test_publish_then_revoke_full_api_flow(client, db_factory):
    """真实 approve -> publish -> revoke 链路：通知由发布落库，由撤回取消。"""
    db = db_factory()
    ds_id = _make_dataset(db)
    db.close()

    _review(client, ds_id, "submit", expected=200)
    _review(client, ds_id, "approve", expected=200)
    db = db_factory()
    ds = db.get(Dataset, ds_id)
    assert ds.review_status == "approved"
    assert ds.is_published is False  # 批准不等于发布
    db.close()

    resp = client.post(f"/api/v1/datasets/{ds_id}/publish")
    assert resp.status_code == 200, resp.text
    db = db_factory()
    note = db.query(DatasetNotification).filter(DatasetNotification.dataset_id == ds_id).one()
    assert note.status == "pending"
    db.close()

    body = _review(client, ds_id, "revoke", expected=200).json()
    assert body["success"] is True
    assert body["cancelled_notification_count"] == 1
    db = db_factory()
    assert db.query(DatasetNotification).filter(
        DatasetNotification.dataset_id == ds_id,
        DatasetNotification.status == "cancelled",
    ).count() == 1
    db.close()


def test_revoke_blocked_when_version_reused(client, db_factory):
    db = db_factory()
    ds_id = _make_dataset(db, published=True, with_reuse=True)
    approved_version = _approved_version(db, ds_id)
    approved_version_id = approved_version.id
    db.close()

    resp = _review(client, ds_id, "revoke", expected=409)
    detail = resp.json()["detail"]
    assert detail["reason"] == "version_reused"
    result = detail["result"]
    assert result["success"] is False
    assert result["active_version"]["version_id"] == approved_version_id
    assert result["active_version"]["version_label"] == "1.0"
    assert "外部算法组" in result["reused_by_teams"]
    assert result["reuse_count"] == 1

    db = db_factory()
    ds = db.get(Dataset, ds_id)
    # 状态保持已发布、可复用，快照、通知、复用记录都未被动过
    assert ds.review_status == "approved"
    assert ds.is_published is True
    assert ds.reuse_count == 1
    assert db.get(DatasetVersion, approved_version_id).is_active is True
    note = db.query(DatasetNotification).filter(DatasetNotification.dataset_id == ds_id).one()
    assert note.status == "pending"
    assert db.query(DatasetReuse).filter(DatasetReuse.dataset_id == ds_id).count() == 1
    assert db.query(DatasetReview).filter(
        DatasetReview.dataset_id == ds_id, DatasetReview.action == "revoke"
    ).count() == 0
    db.close()


def test_revoke_rolls_back_when_commit_fails(client, db_factory):
    db = db_factory()
    ds_id = _make_dataset(db, published=True)
    approved_version_id = _approved_version(db, ds_id).id
    db.close()

    CommitFailureController.armed = True
    resp = _review(client, ds_id, "revoke")
    assert resp.status_code == 500

    db = db_factory()
    ds = db.get(Dataset, ds_id)
    # 全部回滚：仍处于已发布状态
    assert ds.review_status == "approved"
    assert ds.is_published is True
    assert ds.published_at is not None
    version = db.get(DatasetVersion, approved_version_id)
    assert version.is_active is True
    assert version.revoked_at is None
    note = db.query(DatasetNotification).filter(DatasetNotification.dataset_id == ds_id).one()
    assert note.status == "pending"
    assert note.cancelled_at is None
    assert db.query(DatasetReview).filter(
        DatasetReview.dataset_id == ds_id, DatasetReview.action == "revoke"
    ).count() == 0
    db.close()


def test_revoke_is_idempotent(client, db_factory):
    db = db_factory()
    ds_id = _make_dataset(db, published=True)
    db.close()

    first = _review(client, ds_id, "revoke", expected=200).json()
    assert first["success"] is True
    assert first["idempotent"] is False

    second = _review(client, ds_id, "revoke", expected=200).json()
    assert second["success"] is True
    assert second["idempotent"] is True
    assert second["reason"] == "already_revoked"
    assert second["review_status"] == "draft"
    # 当前有效版本指针在两次请求间保持一致
    assert second["active_version"]["version_label"] == "1.0"

    db = db_factory()
    assert db.query(DatasetReview).filter(
        DatasetReview.dataset_id == ds_id, DatasetReview.action == "revoke"
    ).count() == 1
    # 复用计数既不被扣减也不新增（始终为 0）
    assert (db.get(Dataset, ds_id).reuse_count or 0) == 0
    # 通知只被取消一次，没有重复记录
    notes = db.query(DatasetNotification).filter(DatasetNotification.dataset_id == ds_id).all()
    assert len(notes) == 1
    assert notes[0].status == "cancelled"
    db.close()


def test_reuse_rejected_after_revoke_and_inactive_version_unusable(client, db_factory):
    db = db_factory()
    ds_id = _make_dataset(db, published=True)
    approved_version_id = _approved_version(db, ds_id).id
    db.close()

    _review(client, ds_id, "revoke", expected=200)

    # 撤回后数据集未发布，不能再登记复用
    resp = client.post("/api/v1/dataset-reuses", json={
        "dataset_id": ds_id,
        "reusing_team": "新来的组",
    })
    assert resp.status_code == 400

    # 直接发布一个新版本后，显式指定已失效的旧版本仍被拒绝
    _review(client, ds_id, "submit", expected=200)
    _review(client, ds_id, "approve", expected=200)
    client.post(f"/api/v1/datasets/{ds_id}/publish")
    resp = client.post("/api/v1/dataset-reuses", json={
        "dataset_id": ds_id,
        "dataset_version_id": approved_version_id,
        "reusing_team": "新来的组",
    })
    assert resp.status_code == 400
    assert "已撤回失效" in resp.json()["detail"]

    # 默认复用应自动落到新的有效版本，且计数只增加一次
    resp = client.post("/api/v1/dataset-reuses", json={
        "dataset_id": ds_id,
        "reusing_team": "新来的组",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["dataset_version_id"] != approved_version_id
    db = db_factory()
    assert db.get(Dataset, ds_id).reuse_count == 1
    db.close()


def test_revoke_rejected_in_draft_and_rejected(client, db_factory):
    db = db_factory()
    fresh_id = _make_dataset(db)
    rejected_id = _make_dataset(db)
    db.get(Dataset, rejected_id).review_status = "pending_review"
    db.commit()
    db.close()

    # 从未提交审核的草稿不能撤回
    resp = _review(client, fresh_id, "revoke", expected=400)
    assert resp.json()["detail"]["reason"] == "invalid_status"

    # rejected 阶段也不能撤回
    _review(client, rejected_id, "reject", expected=200)
    resp = _review(client, rejected_id, "revoke", expected=400)
    detail = resp.json()["detail"]
    assert detail["reason"] == "invalid_status"
    assert detail["result"]["active_version"] is not None

    db = db_factory()
    assert db.query(DatasetReview).filter(
        DatasetReview.dataset_id == rejected_id, DatasetReview.action == "revoke"
    ).count() == 0
    db.close()
