from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base
from app.routers import common, operation, dataset, analytics


def create_tables():
    import os
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    if not os.path.exists(db_path):
        Base.metadata.create_all(bind=engine)
    else:
        Base.metadata.create_all(bind=engine)


create_tables()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# 机器人真机作业数据回流后端服务

## 功能模块：

### 基础资源管理
- **机型管理**：管理机器人型号信息
- **场景管理**：生产制造、餐饮零售等作业场景
- **技能管理**：抓取、装配、焊接等作业技能

### 作业数据采集
- 按机型、场景、作业技能归类入库
- 动作轨迹、感知记录、抓取成败数据采集

### 人工标注
- 成功/失败标注
- 失败分类（感知异常、运动控制异常、环境干扰等）
- 标注审核流程

### 数据集管理
- 数据集打包与发布
- 优质数据集公开供其他团队复用
- 复用次数统计（关联具体版本）

### 数据集审核
- 提交审核 → 审核（通过/驳回） → 发布
- 审核状态：draft / pending_review / approved / rejected
- 支持在审核中或审核通过后撤回：发布标志、版本快照、待发送订阅通知在同一事务内协调
- 已被其他团队复用的版本不可撤回；重复撤回请求幂等，不重复扣减或新增记录

### 数据集版本
- 每次修改自动创建新版本快照
- 版本历史可追溯、可查询
- 复用方明确使用的版本

### 数据集订阅
- 团队可订阅感兴趣的数据集
- 数据集发布新版本时通知订阅方

### 数据质量分级
- 按完整度和标注质量自动评分分级（A/B/C/D）

### 统计分析
- 按机型、场景统计数据量
- 标注完成率、复用率
- 失败原因分析
- 按审核状态统计（待审/已发布等）
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_prefix = settings.API_V1_PREFIX

app.include_router(common.router, prefix=api_prefix)
app.include_router(operation.router, prefix=api_prefix)
app.include_router(dataset.router, prefix=api_prefix)
app.include_router(analytics.router, prefix=api_prefix)


@app.get("/", tags=["首页"])
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "api_prefix": api_prefix,
        "message": "机器人真机作业数据回流服务已启动"
    }


@app.get("/health", tags=["首页"])
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
