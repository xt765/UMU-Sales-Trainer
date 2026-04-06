"""API Router 模块。

提供销售训练对话系统的 RESTful API 接口，包括会话管理、消息收发、
评估查询和健康检查等端点。使用 FastAPI + Pydantic 实现请求响应验证。
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from umu_sales_trainer.core.workflow import WorkflowState, invoke
from umu_sales_trainer.models.conversation import ConversationSession, Message
from umu_sales_trainer.models.customer import CustomerProfile
from umu_sales_trainer.models.evaluation import EvaluationResult
from umu_sales_trainer.models.product import ProductInfo
from umu_sales_trainer.models.semantic import SemanticPoint
from umu_sales_trainer.services.database import DatabaseService, get_db_service

router = APIRouter(prefix="/api/v1", tags=["api"])


class CreateSessionRequest(BaseModel):
    """创建会话请求模型。

    Attributes:
        customer_profile: 客户画像信息
        product_info: 产品信息
        semantic_points: 语义点列表（可选）
    """

    customer_profile: dict[str, Any]
    product_info: dict[str, Any]
    semantic_points: Optional[list[dict[str, Any]]] = None


class CreateSessionResponse(BaseModel):
    """创建会话响应模型。

    Attributes:
        session_id: 会话ID
        status: 会话状态
        created_at: 创建时间
    """

    session_id: str
    status: str
    created_at: datetime


class SendMessageRequest(BaseModel):
    """发送消息请求模型。

    Attributes:
        content: 消息内容
    """

    content: str = Field(..., min_length=1, description="销售发言内容")


class SendMessageResponse(BaseModel):
    """发送消息响应模型。

    Attributes:
        session_id: 会话ID
        turn: 消息轮次
        ai_response: AI客户回复
        evaluation: 评估结果
    """

    session_id: str
    turn: int
    ai_response: str
    evaluation: dict[str, Any]


class EvaluationResponse(BaseModel):
    """评估结果响应模型。

    Attributes:
        session_id: 会话ID
        coverage_status: 语义点覆盖状态
        coverage_rate: 覆盖率
        overall_score: 综合评分
        expression_analysis: 表达能力分析
    """

    session_id: str
    coverage_status: dict[str, str]
    coverage_rate: float
    overall_score: float
    expression_analysis: dict[str, int]


class SessionStatusResponse(BaseModel):
    """会话状态响应模型。

    Attributes:
        session_id: 会话ID
        status: 会话状态
        created_at: 创建时间
        message_count: 消息数量
    """

    session_id: str
    status: str
    created_at: datetime
    message_count: int


class HealthResponse(BaseModel):
    """健康检查响应模型。

    Attributes:
        status: 服务状态
        timestamp: 检查时间
    """

    status: str
    timestamp: datetime


class MessageResponse(BaseModel):
    """消息响应模型。

    Attributes:
        id: 消息ID
        session_id: 会话ID
        role: 消息角色
        content: 消息内容
        turn: 轮次
        created_at: 创建时间
    """

    id: int
    session_id: str
    role: str
    content: str
    turn: int
    created_at: datetime


def _build_customer_profile(data: dict[str, Any]) -> CustomerProfile:
    """构建客户画像对象。

    Args:
        data: 客户画像数据字典

    Returns:
        CustomerProfile 实例
    """
    return CustomerProfile(
        industry=data.get("industry", ""),
        position=data.get("position", ""),
        concerns=data.get("concerns", []),
        personality=data.get("personality", ""),
        objection_tendencies=data.get("objection_tendencies", []),
    )


def _build_product_info(data: dict[str, Any]) -> ProductInfo:
    """构建产品信息对象。

    Args:
        data: 产品信息数据字典

    Returns:
        ProductInfo 实例
    """
    selling_points: dict[str, Any] = {}
    for sp_id, sp_data in data.get("key_selling_points", {}).items():
        selling_points[sp_id] = type("SellingPoint", (), sp_data)()
    return ProductInfo(
        name=data.get("name", ""),
        description=data.get("description", ""),
        core_benefits=data.get("core_benefits", []),
        key_selling_points=selling_points,
    )


def _build_semantic_points(
    data: Optional[list[dict[str, Any]]],
) -> list[SemanticPoint]:
    """构建语义点列表。

    Args:
        data: 语义点数据列表

    Returns:
        SemanticPoint 列表
    """
    if not data:
        return []
    return [
        SemanticPoint(
            point_id=sp.get("point_id", ""),
            description=sp.get("description", ""),
            keywords=sp.get("keywords", []),
            weight=sp.get("weight", 1.0),
        )
        for sp in data
    ]


def _build_semantic_points_from_product(product: ProductInfo) -> list[SemanticPoint]:
    """从产品信息构建语义点列表。

    将产品的核心优势(key_selling_points)转换为语义点格式，
    用于三层检测机制评估。

    Args:
        product: 产品信息对象

    Returns:
        SemanticPoint 列表
    """
    semantic_points = []

    # 从核心优势生成语义点
    if product.core_benefits:
        for idx, benefit in enumerate(product.core_benefits, 1):
            # 解析benefit字符串（可能包含冒号分隔的描述）
            if isinstance(benefit, str) and ":" in benefit:
                parts = benefit.split(":", 1)
                title = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else title
            else:
                title = str(benefit)
                description = str(benefit)

            # 从标题和描述中提取关键词
            keywords = _extract_keywords(f"{title} {description}")

            semantic_points.append(
                SemanticPoint(
                    point_id=f"SP-{idx:03d}",
                    description=description,
                    keywords=keywords,
                    weight=1.0,
                )
            )

    # 从key_selling_points添加额外语义点
    if product.key_selling_points:
        for sp_id, sp in product.key_selling_points.items():
            # 跳过已存在的语义点
            existing_ids = {p.point_id for p in semantic_points}
            if sp_id in existing_ids:
                continue

            keywords = _extract_keywords(
                f"{sp.description} {' '.join(sp.keywords)} {' '.join(sp.sample_phrases)}"
            )

            semantic_points.append(
                SemanticPoint(
                    point_id=sp_id,
                    description=sp.description,
                    keywords=keywords,
                    weight=sp.weight if hasattr(sp, "weight") else 1.0,
                )
            )

    # 如果仍然没有语义点，提供默认的通用语义点
    if not semantic_points:
        default_points = [
            ("SP-001", "产品介绍", ["产品", "介绍", "特点"]),
            ("SP-002", "疗效效果", ["效果", "疗效", "改善"]),
            ("SP-003", "安全性", ["安全", "副作用", "不良反应"]),
        ]
        for pid, desc, kws in default_points:
            semantic_points.append(
                SemanticPoint(
                    point_id=pid,
                    description=desc,
                    keywords=kws,
                    weight=1.0,
                )
            )

    return semantic_points


def _extract_keywords(text: str) -> list[str]:
    """从文本中提取关键词。

    简单的关键词提取，基于中文分词和常见词汇模式。

    Args:
        text: 输入文本

    Returns:
        关键词列表
    """
    import re

    # 常见停用词
    stop_words = {
        "的",
        "了",
        "是",
        "在",
        "有",
        "和",
        "与",
        "或",
        "等",
        "及",
        "对",
        "为",
        "以",
        "被",
        "把",
        "从",
        "到",
    }

    # 提取中文词汇（2-6个字符）
    words = re.findall(r"[\u4e00-\u9fff]{2,6}", text)

    # 过滤停用词并去重
    keywords = []
    seen = set()
    for word in words:
        word_lower = word.lower()
        if word_lower not in stop_words and word_lower not in seen:
            keywords.append(word)
            seen.add(word_lower)

    return keywords[:10]  # 限制最多10个关键词


def _format_evaluation(eval_result: EvaluationResult) -> dict[str, Any]:
    """格式化评估结果为字典。

    Args:
        eval_result: 评估结果对象

    Returns:
        格式化的字典
    """
    expr = eval_result.expression_analysis
    return {
        "coverage_status": eval_result.coverage_status,
        "coverage_rate": eval_result.coverage_rate,
        "overall_score": eval_result.overall_score,
        "expression_analysis": {
            "clarity": expr.clarity,
            "professionalism": expr.professionalism,
            "persuasiveness": expr.persuasiveness,
        },
    }


@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """创建训练会话。

    创建一个新的销售训练会话，初始化客户画像和产品信息。

    Args:
        request: 创建会话请求，包含客户和产品信息

    Returns:
        创建的会话信息

    Raises:
        HTTPException: 数据库保存失败时抛出
    """
    import uuid

    session_id = str(uuid.uuid4())
    now = datetime.utcnow()

    session_data = ConversationSession(
        id=session_id,
        customer_profile=str(request.customer_profile),
        product_info=str(request.product_info),
        status="active",
        created_at=now,
    )

    db: DatabaseService = get_db_service()
    db.init_db()

    saved_session = db.save_session(
        session_id=session_data.id,
        customer_profile=request.customer_profile,
        product_info=request.product_info,
    )

    if not saved_session:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session",
        )

    return CreateSessionResponse(
        session_id=session_id,
        status="active",
        created_at=now,
    )


@router.post("/sessions/{session_id}/messages", response_model=SendMessageResponse)
def send_message(
    session_id: str,
    request: SendMessageRequest,
) -> SendMessageResponse:
    """发送销售发言，获取AI客户回复。

    处理销售人员的发言，通过工作流执行分析、评估和模拟，
    返回AI客户的回复以及当前评估结果。

    Args:
        session_id: 会话ID
        request: 发送消息请求

    Returns:
        AI回复和评估结果

    Raises:
        HTTPException: 会话不存在时抛出
    """
    db: DatabaseService = get_db_service()
    session = db.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    messages = db.get_messages(session_id)
    turn = len(messages) + 1

    db.save_message(
        session_id=session_id,
        role="user",
        content=request.content,
        turn=turn,
    )

    customer = _build_customer_profile(session.customer_profile)
    product = _build_product_info(session.product_info)

    # 构建语义点列表（从产品信息的core_benefits和selling_points生成）
    semantic_points = _build_semantic_points_from_product(product)

    messages_for_workflow = [
        Message(
            session_id=session_id,
            role=m.role,
            content=m.content,
            turn=m.turn,
        )
        for m in messages
    ]

    workflow_state: WorkflowState = {
        "session_id": session_id,
        "sales_message": request.content,
        "customer_profile": customer,
        "product_info": product,
        "conversation_history": messages_for_workflow,
        "semantic_points": semantic_points,
        "analysis_result": None,
        "evaluation_result": None,
        "guidance": None,
        "ai_response": None,
        "next_node": "",
    }

    result = invoke(workflow_state)

    ai_response = result.get("ai_response", "抱歉，我现在无法回复。")
    evaluation = result.get("evaluation_result")

    if evaluation:
        db.save_coverage_record(
            session_id=session_id,
            point_id="overall",
            is_covered=True,
            coverage_details=_format_evaluation(evaluation),
        )

    db.save_message(
        session_id=session_id,
        role="assistant",
        content=ai_response or "",
        turn=turn,
    )

    eval_dict = (
        _format_evaluation(evaluation)
        if evaluation
        else {
            "coverage_status": {},
            "coverage_rate": 0.0,
            "overall_score": 0.0,
            "expression_analysis": {"clarity": 0, "professionalism": 0, "persuasiveness": 0},
        }
    )

    return SendMessageResponse(
        session_id=session_id,
        turn=turn,
        ai_response=ai_response or "",
        evaluation=eval_dict,
    )


@router.get("/sessions/{session_id}/evaluation", response_model=EvaluationResponse)
def get_evaluation(session_id: str) -> EvaluationResponse:
    """获取会话评估结果。

    查询指定会话的当前评估结果，包括语义点覆盖状态和表达能力分析。

    Args:
        session_id: 会话ID

    Returns:
        评估结果

    Raises:
        HTTPException: 会话不存在时抛出
    """
    db: DatabaseService = get_db_service()
    session = db.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    messages = db.get_messages(session_id)

    if not messages:
        return EvaluationResponse(
            session_id=session_id,
            coverage_status={},
            coverage_rate=0.0,
            overall_score=0.0,
            expression_analysis={"clarity": 0, "professionalism": 0, "persuasiveness": 0},
        )

    last_user_msg = None
    for msg in reversed(messages):
        if msg.role == "user":
            last_user_msg = msg
            break

    if not last_user_msg:
        return EvaluationResponse(
            session_id=session_id,
            coverage_status={},
            coverage_rate=0.0,
            overall_score=0.0,
            expression_analysis={"clarity": 0, "professionalism": 0, "persuasiveness": 0},
        )

    customer = _build_customer_profile(session.customer_profile)
    product = _build_product_info(session.product_info)

    workflow_state: WorkflowState = {
        "session_id": session_id,
        "sales_message": last_user_msg.content,
        "customer_profile": customer,
        "product_info": product,
        "conversation_history": [],
        "semantic_points": [],
        "analysis_result": None,
        "evaluation_result": None,
        "guidance": None,
        "ai_response": None,
        "next_node": "",
    }

    result = invoke(workflow_state)
    evaluation = result.get("evaluation_result")

    if not evaluation:
        return EvaluationResponse(
            session_id=session_id,
            coverage_status={},
            coverage_rate=0.0,
            overall_score=0.0,
            expression_analysis={"clarity": 0, "professionalism": 0, "persuasiveness": 0},
        )

    return EvaluationResponse(
        session_id=session_id,
        coverage_status=evaluation.coverage_status,
        coverage_rate=evaluation.coverage_rate,
        overall_score=evaluation.overall_score,
        expression_analysis={
            "clarity": evaluation.expression_analysis.clarity,
            "professionalism": evaluation.expression_analysis.professionalism,
            "persuasiveness": evaluation.expression_analysis.persuasiveness,
        },
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str) -> None:
    """软删除会话。

    对指定会话执行软删除，将其标记为已删除状态。

    Args:
        session_id: 会话ID

    Raises:
        HTTPException: 会话不存在时抛出
    """
    db: DatabaseService = get_db_service()
    session = db.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    db.soft_delete_session(session_id)


@router.get("/sessions/{session_id}/status", response_model=SessionStatusResponse)
def get_session_status(session_id: str) -> SessionStatusResponse:
    """获取会话状态。

    查询指定会话的当前状态和基本信息。

    Args:
        session_id: 会话ID

    Returns:
        会话状态信息

    Raises:
        HTTPException: 会话不存在时抛出
    """
    db: DatabaseService = get_db_service()
    session = db.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    messages = db.get_messages(session_id)

    return SessionStatusResponse(
        session_id=session_id,
        status=session.status,
        created_at=session.created_at or datetime.utcnow(),
        message_count=len(messages),
    )


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """健康检查接口。

    返回服务健康状态，用于监控和负载均衡探测。

    Returns:
        健康状态响应
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
    )


api_router = router
