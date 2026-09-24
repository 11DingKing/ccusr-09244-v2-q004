"""撤回（revoke）语义测试。

覆盖：
- 审核中撤回（pending_review -> draft）
- 审核通过但未发布时撤回（approved / is_published=False）
- 已发布且无复用时撤回（发布标志、版本快照、待发送通知在一个事务内协调变化）
- 已发布且被其他团队实际复用时阻止撤回（返回阻止原因与当前有效版本，状态不变）
- 撤回被实际复用阻止后，状态/快照/通知保持不变
- 重复撤回幂等：不新增审核记录、不重复扣减复用计数
- 数据库提交失败时整体回滚，状态保持撤回前
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models import (
    RobotModel,
    Scene,
    Skill,
    OperationData,
    Dataset,
    DatasetItem,
    DatasetSubscription,
    DatasetVersion,
    DatasetReview,
    DatasetNotification,
    DatasetReuse,
)
from datetime import datetime, timezone, timedelta

from main import app

API = "/api/v1"


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSession()

    def override_get_db():
        try:
            yield session
        finally:
            pass  # 测试统一管理 session 生命周期

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield session
    finally:
        app.dependency_overrides.clear()
        session.close()
        engine.dispose()


@pytest.fixture()
def client(db_session):
    return TestClient(app)


def _make_base_data(db, *, with_item=True):
    """创建机型/场景/技能和一个带数据项的数据集（草稿态），返回 dataset。"""
    rm = RobotModel(name="RM-TEST", manufacturer="RealMotion")
    scene = Scene(name="测试场景", category="测试")
    skill = Skill(name="测试技能", category="操作")
    db.add_all([rm, scene, skill])
    db.flush()

    now = datetime.now(timezone.utc)
    op = OperationData(
        robot_model_id=rm.id,
        scene_id=scene.id,
        skill_id=skill.id,
        motion_trajectory={"waypoints": []},
        perception_records={"camera_images_captured": 1},
        timestamp_start=now - timedelta(seconds=10),
        timestamp_end=now,
    )
    db.add(op)

    dataset = Dataset(
        name="测试数据集",
        robot_model_id=rm.id,
        scene_id=scene.id,
        skill_id=skill.id,
        owner_team="数据组",
        review_status="draft",
        is_published=False,
        total_items=1,
        current_version=1,
        version="1.0",
        reuse_count=0,
    )
    db.add(dataset)
    db.flush()
    db.add(DatasetVersion(
        dataset_id=dataset.id,
        version_number=1,
        version_label="1.0",
        change_description="初始版本",
        is_active=False,
        total_items=0,
    ))
    if with_item:
        db.add(DatasetItem(dataset_id=dataset.id, operation_data_id=op.id))
        dataset.total_items = 1
    db.commit()
    return dataset


def _review(client, dataset_id, action, **kw):
    return client.post(f"{API}/datasets/{dataset_id}/review", json={"action": action, **kw})


def _active_version(db, dataset_id):
    return (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.is_active == True)
        .order_by(DatasetVersion.version_number.desc())
        .first()
    )


# ---------- 1. 审核中撤回 ----------

def test_revoke_during_pending_review(client, db_session):
    dataset = _make_base_data(db_session)

    assert _review(client, dataset.id, "submit", reviewer="负责人").status_code == 200
    db_session.expire_all()
    assert db_session.get(Dataset, dataset.id).review_status == "pending_review"

    resp = _review(client, dataset.id, "revoke", reviewer="负责人", review_notes="信息填错")
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "revoke"

    db_session.expire_all()
    refreshed = db_session.get(Dataset, dataset.id)
    assert refreshed.review_status == "draft"
    assert refreshed.is_published is False
    assert refreshed.published_at is None
    # 审核中尚未产生有效版本，撤回不应凭空激活快照
    assert _active_version(db_session, dataset.id) is None


# ---------- 2. 批准未发布撤回 ----------

def test_revoke_approved_not_published(client, db_session):
    dataset = _make_base_data(db_session)

    _review(client, dataset.id, "submit")
    approve = _review(client, dataset.id, "approve", reviewer="审核员")
    assert approve.status_code == 200

    db_session.expire_all()
    refreshed = db_session.get(Dataset, dataset.id)
    assert refreshed.review_status == "approved"
    assert refreshed.is_published is False
    active = _active_version(db_session, dataset.id)
    assert active is not None  # 审核通过把待审快照提升为当前有效版本

    resp = _review(client, dataset.id, "revoke", reviewer="负责人")
    assert resp.status_code == 200

    db_session.expire_all()
    refreshed = db_session.get(Dataset, dataset.id)
    assert refreshed.review_status == "draft"
    assert refreshed.is_published is False
    assert refreshed.published_at is None
    # 版本快照保留留痕，但已失效
    assert _active_version(db_session, dataset.id) is None
    snapshot = db_session.query(DatasetVersion).filter_by(id=active.id).first()
    assert snapshot.is_active is False

    # 撤回记录关联被撤回的版本
    revoke_review = (
        db_session.query(DatasetReview)
        .filter(DatasetReview.dataset_id == dataset.id, DatasetReview.action == "revoke")
        .one()
    )
    assert revoke_review.dataset_version_id == active.id


# ---------- 3. 已发布、无复用撤回 ----------

def test_revoke_published_without_reuse_coordinates_all_state(client, db_session):
    dataset = _make_base_data(db_session)
    db_session.add(DatasetSubscription(
        dataset_id=dataset.id, subscriber_team="算法组", notify_on_new_version=True
    ))
    db_session.commit()

    _review(client, dataset.id, "submit")
    _review(client, dataset.id, "approve", reviewer="审核员")
    pub = client.post(f"{API}/datasets/{dataset.id}/publish")
    assert pub.status_code == 200

    db_session.expire_all()
    assert db_session.get(Dataset, dataset.id).is_published is True
    pending = db_session.query(DatasetNotification).filter_by(
        dataset_id=dataset.id, is_sent=False
    ).all()
    assert len(pending) == 1  # 发布已排队待发送通知

    resp = _review(client, dataset.id, "revoke", reviewer="负责人")
    assert resp.status_code == 200

    db_session.expire_all()
    refreshed = db_session.get(Dataset, dataset.id)
    # 发布标志一并收起，外部查询不会再看到“已撤回 + 可复用”矛盾状态
    assert refreshed.review_status == "draft"
    assert refreshed.is_published is False
    assert refreshed.published_at is None
    assert _active_version(db_session, dataset.id) is None
    # 待发送通知随撤回在同一事务内删除
    assert db_session.query(DatasetNotification).filter_by(
        dataset_id=dataset.id, is_sent=False
    ).count() == 0


# ---------- 4. 已发布、存在复用，阻止撤回 ----------

def test_revoke_blocked_when_version_reused(client, db_session):
    dataset = _make_base_data(db_session)
    db_session.add(DatasetSubscription(
        dataset_id=dataset.id, subscriber_team="算法组", notify_on_new_version=True
    ))
    db_session.commit()

    _review(client, dataset.id, "submit")
    _review(client, dataset.id, "approve", reviewer="审核员")
    client.post(f"{API}/datasets/{dataset.id}/publish")

    active = _active_version(db_session, dataset.id)
    reuse_resp = client.post(f"{API}/dataset-reuses", json={
        "dataset_id": dataset.id,
        "reusing_team": "其他团队",
        "purpose": "模型训练",
    })
    assert reuse_resp.status_code == 200
    # 复用记录挂在当前有效版本上
    assert reuse_resp.json()["dataset_version_id"] == active.id

    resp = _review(client, dataset.id, "revoke", reviewer="负责人")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["blocked"] is True
    assert detail["reuse_count"] == 1
    assert "其他团队" in detail["reason"]
    # 调用方拿到当前有效版本
    assert detail["active_version"]["id"] == active.id
    assert detail["active_version"]["version_label"] == "1.0"
    assert detail["reuses"][0]["reusing_team"] == "其他团队"

    # 状态、版本快照、复用计数、待发通知全部保持不变
    db_session.expire_all()
    refreshed = db_session.get(Dataset, dataset.id)
    assert refreshed.review_status == "approved"
    assert refreshed.is_published is True
    assert refreshed.published_at is not None
    assert refreshed.reuse_count == 1
    assert _active_version(db_session, dataset.id).id == active.id
    assert db_session.query(DatasetNotification).filter_by(
        dataset_id=dataset.id, is_sent=False
    ).count() == 1
    assert db_session.query(DatasetReuse).filter_by(dataset_id=dataset.id).count() == 1
    assert db_session.query(DatasetReview).filter_by(
        dataset_id=dataset.id, action="revoke"
    ).count() == 0


# ---------- 5. 重复撤回幂等：不新增记录、不重复扣减 ----------

def test_repeated_blocked_revoke_never_mutates(client, db_session):
    dataset = _make_base_data(db_session)
    db_session.add(DatasetSubscription(
        dataset_id=dataset.id, subscriber_team="算法组", notify_on_new_version=True
    ))
    db_session.commit()

    _review(client, dataset.id, "submit")
    _review(client, dataset.id, "approve", reviewer="审核员")
    client.post(f"{API}/datasets/{dataset.id}/publish")
    active = _active_version(db_session, dataset.id)

    client.post(f"{API}/dataset-reuses", json={
        "dataset_id": dataset.id, "reusing_team": "其他团队", "purpose": "训练"
    })

    for _ in range(3):
        resp = _review(client, dataset.id, "revoke", reviewer="负责人")
        assert resp.status_code == 409

    db_session.expire_all()
    refreshed = db_session.get(Dataset, dataset.id)
    # 多次被阻止的请求不扣减计数、不新增撤回记录、不删除通知
    assert refreshed.is_published is True
    assert refreshed.reuse_count == 1
    assert _active_version(db_session, dataset.id).id == active.id
    assert db_session.query(DatasetReuse).filter_by(dataset_id=dataset.id).count() == 1
    assert db_session.query(DatasetNotification).filter_by(
        dataset_id=dataset.id, is_sent=False
    ).count() == 1
    assert db_session.query(DatasetReview).filter_by(
        dataset_id=dataset.id, action="revoke"
    ).count() == 0


def test_repeat_revoke_is_idempotent(client, db_session):
    dataset = _make_base_data(db_session)
    # 预留一个与实际复用记录不符的陈旧计数，验证撤回按实际记录对账且不重复扣减
    dataset.reuse_count = 3
    db_session.commit()

    _review(client, dataset.id, "submit")
    _review(client, dataset.id, "approve", reviewer="审核员")

    first = _review(client, dataset.id, "revoke", reviewer="负责人")
    assert first.status_code == 200
    first_id = first.json()["id"]

    db_session.expire_all()
    assert db_session.get(Dataset, dataset.id).reuse_count == 0

    # 第二次重复撤回：返回同一条记录，不新增、不再次扣减
    second = _review(client, dataset.id, "revoke", reviewer="负责人")
    assert second.status_code == 200
    assert second.json()["id"] == first_id

    db_session.expire_all()
    refreshed = db_session.get(Dataset, dataset.id)
    assert refreshed.review_status == "draft"
    assert refreshed.is_published is False
    assert refreshed.reuse_count == 0
    assert db_session.query(DatasetReview).filter_by(
        dataset_id=dataset.id, action="revoke"
    ).count() == 1


def test_new_version_snapshot_numbering_and_active_invariant(client, db_session):
    dataset = _make_base_data(db_session)

    # 走完 v1 发布
    _review(client, dataset.id, "submit")
    _review(client, dataset.id, "approve", reviewer="审核员")
    client.post(f"{API}/datasets/{dataset.id}/publish")
    v1 = _active_version(db_session, dataset.id)
    assert v1.version_number == 1 and v1.version_label == "1.0"

    # 创建 v2 草稿：旧快照失效，新草稿快照号/标签正确且未生效
    resp = client.post(f"{API}/datasets/{dataset.id}/versions", json={"change_description": "加数据"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["version_number"] == 2 and body["version_label"] == "1.1"
    assert body["is_active"] is False
    db_session.expire_all()
    ds = db_session.get(Dataset, dataset.id)
    assert ds.current_version == 2 and ds.version == "1.1"
    assert ds.is_published is False and ds.review_status == "draft"
    assert _active_version(db_session, dataset.id) is None
    assert db_session.query(DatasetVersion).filter_by(
        dataset_id=dataset.id, version_number=1
    ).count() == 1  # 旧版本号未被重复占用

    # v2 通过并发布后成为唯一有效版本
    _review(client, dataset.id, "submit")
    _review(client, dataset.id, "approve", reviewer="审核员")
    client.post(f"{API}/datasets/{dataset.id}/publish")
    db_session.expire_all()
    actives = db_session.query(DatasetVersion).filter_by(
        dataset_id=dataset.id, is_active=True
    ).all()
    assert len(actives) == 1
    assert actives[0].version_number == 2 and actives[0].version_label == "1.1"


def test_revoke_from_invalid_stage_rejected(client, db_session):
    dataset = _make_base_data(db_session)
    # 草稿态直接撤回（此前没有撤回记录）应被拒绝
    resp = _review(client, dataset.id, "revoke", reviewer="负责人")
    assert resp.status_code == 400
    db_session.expire_all()
    assert db_session.get(Dataset, dataset.id).review_status == "draft"


def test_revoke_idempotency_scoped_to_review_cycle(client, db_session):
    # 撤回后再次提交审核进入下一周期，旧撤回不应被幂等命中
    dataset = _make_base_data(db_session)
    _review(client, dataset.id, "submit")
    _review(client, dataset.id, "approve", reviewer="审核员")
    first = _review(client, dataset.id, "revoke", reviewer="负责人")
    assert first.status_code == 200

    # 重新提交 -> 审核通过；这是一个新的审核周期
    assert _review(client, dataset.id, "submit").status_code == 200
    assert _review(client, dataset.id, "approve", reviewer="审核员").status_code == 200

    again = _review(client, dataset.id, "revoke", reviewer="负责人")
    assert again.status_code == 200
    assert again.json()["id"] != first.json()["id"]
    assert db_session.query(DatasetReview).filter_by(
        dataset_id=dataset.id, action="revoke"
    ).count() == 2


# ---------- 6. 数据库提交失败，整体回滚 ----------

def test_revoke_rolls_back_on_commit_failure(client, db_session, monkeypatch):
    dataset = _make_base_data(db_session)
    db_session.add(DatasetSubscription(
        dataset_id=dataset.id, subscriber_team="算法组", notify_on_new_version=True
    ))
    db_session.commit()

    _review(client, dataset.id, "submit")
    _review(client, dataset.id, "approve", reviewer="审核员")
    client.post(f"{API}/datasets/{dataset.id}/publish")
    active = _active_version(db_session, dataset.id)

    # 让撤回事务在提交时失败
    def failing_commit():
        raise RuntimeError("simulated database outage")

    monkeypatch.setattr(db_session, "commit", failing_commit)

    resp = _review(client, dataset.id, "revoke", reviewer="负责人")
    assert resp.status_code == 500
    assert "回滚" in resp.json()["detail"]

    monkeypatch.undo()
    db_session.rollback()
    db_session.expire_all()

    # 所有状态保持撤回前
    refreshed = db_session.get(Dataset, dataset.id)
    assert refreshed.review_status == "approved"
    assert refreshed.is_published is True
    assert refreshed.published_at is not None
    assert _active_version(db_session, dataset.id).id == active.id
    assert db_session.query(DatasetNotification).filter_by(
        dataset_id=dataset.id, is_sent=False
    ).count() == 1
    assert db_session.query(DatasetReview).filter_by(
        dataset_id=dataset.id, action="revoke"
    ).count() == 0
