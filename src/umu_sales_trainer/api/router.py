"""API Router 模块。

提供销售训练对话系统的 RESTful API 接口，包括会话管理、消息收发、
评估查询和健康检查等端点。使用 FastAPI + Pydantic 实现请求响应验证。
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from umu_sales_trainer.core.workflow import WorkflowState, create_workflow
from umu_sales_trainer.models.conversation import ConversationSession, Message
from umu_sales_trainer.models.customer import CustomerProfile
from umu_sales_trainer.models.evaluation import EvaluationResult
from umu_sales_trainer.models.product import ProductInfo
from umu_sales_trainer.models.semantic import SemanticPoint
from umu_sales_trainer.services.database import DatabaseService, get_db_service
from umu_sales_trainer.services.embedding import EmbeddingService
from umu_sales_trainer.services.llm import create_llm

_workflow_instance = None


def _get_workflow():
    """延迟初始化工作流实例。

    首次调用时创建 EmbeddingService、LLMService 和编译后的 StateGraph，
    后续调用直接返回已创建的实例。

    Returns:
        编译好的 LangGraph 工作流实例
    """
    global _workflow_instance
    if _workflow_instance is None:
        embedding_service = EmbeddingService()
        llm_service = create_llm("dashscope")
        _workflow_instance = create_workflow(embedding_service, llm_service)
    return _workflow_instance


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
        evaluation: 评估结果（含对话分析、语义覆盖、表达质量、改进建议）
        guidance: 智能引导建议（覆盖率不足时生成）
    """

    session_id: str
    turn: int
    ai_response: str
    evaluation: dict[str, Any]
    guidance: Optional[dict[str, Any]] = None


class EvaluationResponse(BaseModel):
    """评估结果响应模型。

    Attributes:
        session_id: 会话ID
        coverage_status: 语义点覆盖状态
        coverage_labels: 语义点ID到中文标签的映射
        coverage_rate: 覆盖率
        overall_score: 综合评分
        expression_analysis: 表达能力分析
        suggestions: 改进建议列表
        conversation_analysis: 对话分析结果
    """

    session_id: str
    coverage_status: dict[str, str]
    coverage_labels: dict[str, str]
    coverage_rate: float
    overall_score: float
    expression_analysis: dict[str, int]
    suggestions: list[dict[str, Any]] = []


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


class SessionListItem(BaseModel):
    """会话列表项模型。

    Attributes:
        session_id: 会话ID
        status: 会话状态
        created_at: 创建时间
        message_count: 消息轮次数量
    """

    session_id: str
    status: str
    created_at: datetime
    message_count: int


class SessionListResponse(BaseModel):
    """会话列表响应模型。

    Attributes:
        sessions: 会话列表
        total: 总数
    """

    sessions: list[SessionListItem]
    total: int


class MessagesListResponse(BaseModel):
    """消息列表响应模型。

    Attributes:
        session_id: 会话ID
        messages: 消息列表
        total: 消息总数
    """

    session_id: str
    messages: list[MessageResponse]
    total: int


def _build_customer_profile(data: dict[str, Any]) -> CustomerProfile:
    """构建客户画像对象。

    支持前端传入的 name/hospital/personality_type 字段映射，
    并根据职位自动生成合理的关注点。

    Args:
        data: 客户画像数据字典

    Returns:
        CustomerProfile 实例
    """
    position = data.get("position", "")

    _PERSONALITY_MAP = {
        "ANALYTICAL": "专业严谨，注重数据和循证医学",
        "DRIVER": "果断直接，注重效率和结果",
        "EXPRESSIVE": "热情开放，注重创新和关系",
        "AMIABLE": "温和谨慎，注重安全和信任",
    }

    _CONCERNS_BY_POSITION: dict[str, list[str]] = {
        "内分泌科主任": ["HbA1c控制效果", "低血糖风险", "患者依从性"],
        "医生": ["疗效", "安全性", "副作用"],
        "采购经理": ["价格", "性价比", "供货稳定性"],
    }

    return CustomerProfile(
        name=data.get("name", ""),
        hospital=data.get("hospital", ""),
        industry=data.get("industry", "医疗"),
        position=position,
        concerns=data.get("concerns") or _CONCERNS_BY_POSITION.get(position, ["疗效", "安全性"]),
        personality=data.get("personality")
        or _PERSONALITY_MAP.get(data.get("personality_type", ""), ""),
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

    固定使用 core_benefits 中的语义点（通常3个），不动态增减。
    key_selling_points 中的关键词仅用于补充同义语义点的检测关键词，
    不生成额外语义点，确保前端显示的语义点数量始终一致。

    Args:
        product: 产品信息对象

    Returns:
        固定数量的 SemanticPoint 列表
    """
    semantic_points: list[SemanticPoint] = []

    if not product.core_benefits:
        return _get_default_semantic_points()

    for idx, benefit in enumerate(product.core_benefits, 1):
        if isinstance(benefit, dict):
            point_id = benefit.get("id", f"SP-{idx:03d}")
            description = benefit.get("description", str(benefit))
        elif isinstance(benefit, str) and ":" in benefit:
            parts = benefit.split(":", 1)
            point_id = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else point_id
        elif hasattr(benefit, "id") and hasattr(benefit, "description"):
            point_id = getattr(benefit, "id", f"SP-{idx:03d}")
            description = getattr(benefit, "description", "")
        else:
            point_id = f"SP-{idx:03d}"
            description = str(benefit)

        base_keywords = _extract_keywords(f"{point_id} {description}")

        extra_keywords = _find_matching_ksp_keywords(point_id, description, product)

        all_keywords = list(set(base_keywords + extra_keywords))

        weight = _infer_weight_from_id(point_id)

        semantic_points.append(
            SemanticPoint(
                point_id=point_id,
                description=description,
                keywords=all_keywords,
                weight=weight,
            )
        )

    if not semantic_points:
        return _get_default_semantic_points()

    return semantic_points


def _find_matching_ksp_keywords(
    benefit_id: str,
    benefit_description: str,
    product: ProductInfo,
) -> list[str]:
    """从 key_selling_points 中查找与当前 core_benefit 匹配的关键词进行补充。

    通过 ID 关键词和描述文本相似度双重匹配，将 key_selling_points 中
    更丰富的关键词信息合并到对应的 core_benefit 语义点上。

    Args:
        benefit_id: 当前 core_benefit 的 ID（如 SP_SAFETY）
        benefit_description: 当前 core_benefit 的中文描述
        product: 完整的产品信息对象

    Returns:
        补充的关键词列表
    """
    if not product.key_selling_points:
        return []

    matched_keywords: list[str] = []
    benefit_kw_set = set(_extract_keywords(f"{benefit_id} {benefit_description}"))

    for sp_id, sp in product.key_selling_points.items():
        sp_kws = getattr(sp, "keywords", [])
        sp_desc = getattr(sp, "description", "")

        sp_kw_set = set(_extract_keywords(f"{sp_id} {sp_desc} {' '.join(sp_kws)}"))

        if sp_kw_set and benefit_kw_set:
            overlap_ratio = len(sp_kw_set & benefit_kw_set) / max(len(sp_kw_set), 1)
            if overlap_ratio > 0.15:
                matched_keywords.extend(sp_kws)
                sample_phrases = getattr(sp, "sample_phrases", [])
                if sample_phrases:
                    matched_keywords.extend(_extract_keywords(" ".join(sample_phrases)))

    return matched_keywords


def _infer_weight_from_id(point_id: str) -> float:
    """根据语义点ID推断权重值。

    安全性和疗效类语义点赋予较高权重，便利性等次要维度权重略低。

    Args:
        point_id: 语义点标识符

    Returns:
        权重值（1.0-1.5）
    """
    id_upper = point_id.upper()
    if any(kw in id_upper for kw in ("SAFETY", "EFFICACY")):
        return 1.2
    return 1.0


def _get_default_semantic_points() -> list[SemanticPoint]:
    """当产品无 core_benefits 时返回默认语义点。

    Returns:
        包含3个通用语义点的列表
    """
    return [
        SemanticPoint(
            point_id="SP-001", description="产品介绍", keywords=["产品", "介绍", "特点"], weight=1.0
        ),
        SemanticPoint(
            point_id="SP-002", description="疗效效果", keywords=["效果", "疗效", "改善"], weight=1.2
        ),
        SemanticPoint(
            point_id="SP-003",
            description="安全性",
            keywords=["安全", "副作用", "不良反应"],
            weight=1.2,
        ),
    ]


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


def _format_evaluation(
    eval_result: EvaluationResult,
    coverage_labels: dict[str, str] | None = None,
    expression_result=None,
) -> dict[str, Any]:
    """格式化评估结果为字典。

    Args:
        eval_result: 评估结果对象
        coverage_labels: 语义点ID到中文描述的映射（可选）
        expression_result: ExpressionResult 对象（可选，用于提取 suggestions）

    Returns:
        格式化的字典
    """
    expr = eval_result.expression_analysis
    result = {
        "coverage_status": eval_result.coverage_status,
        "coverage_rate": eval_result.coverage_rate,
        "overall_score": eval_result.overall_score,
        "expression_analysis": {
            "clarity": expr.clarity,
            "professionalism": expr.professionalism,
            "persuasiveness": expr.persuasiveness,
        },
    }
    if coverage_labels is not None:
        result["coverage_labels"] = coverage_labels

    if expression_result is not None:
        result["suggestions"] = [
            {
                "dimension": s.dimension,
                "current_score": s.current_score,
                "advice": s.advice,
                "example": s.example,
            }
            for s in expression_result.suggestions
        ]

    return result


def _format_guidance(guidance_result) -> Optional[dict[str, Any]]:
    """格式化引导结果为字典。

    Args:
        guidance_result: GuidanceResult 对象

    Returns:
        格式化的字典，或 None（如果不需要引导）
    """
    if guidance_result is None or not guidance_result.is_actionable:
        return None

    return {
        "summary": guidance_result.summary,
        "is_actionable": guidance_result.is_actionable,
        "priority_list": [
            {
                "gap": item.gap,
                "urgency": item.urgency,
                "suggestion": item.suggestion,
                "talking_point": item.talking_point,
                "expected_effect": item.expected_effect,
            }
            for item in guidance_result.priority_list
        ],
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

    # 拼接全程用户消息用于累积式语义评估
    all_user_messages = [m.content for m in messages if m.role == "user"]
    all_user_messages.append(request.content)
    cumulative_sales_text = "\n".join(all_user_messages)

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
        "sales_message": cumulative_sales_text,
        "current_message": request.content,
        "customer_profile": customer,
        "product_info": product,
        "semantic_points": semantic_points,
        "messages": messages_for_workflow,
    }

    result = _get_workflow().invoke(workflow_state)

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

    coverage_labels = {sp.point_id: sp.description for sp in semantic_points}

    expr_result = result.get("expression_result")
    guide_result = result.get("guidance_result")

    eval_dict = (
        _format_evaluation(evaluation, coverage_labels, expr_result)
        if evaluation
        else {
            "coverage_status": {},
            "coverage_labels": coverage_labels,
            "coverage_rate": 0.0,
            "overall_score": 0.0,
            "expression_analysis": {"clarity": 0, "professionalism": 0, "persuasiveness": 0},
            "suggestions": [],
        }
    )

    guidance_dict = _format_guidance(guide_result)

    return SendMessageResponse(
        session_id=session_id,
        turn=turn,
        ai_response=ai_response or "",
        evaluation=eval_dict,
        guidance=guidance_dict,
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
            coverage_labels={},
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
            coverage_labels={},
            coverage_rate=0.0,
            overall_score=0.0,
            expression_analysis={"clarity": 0, "professionalism": 0, "persuasiveness": 0},
        )

    customer = _build_customer_profile(session.customer_profile)
    product = _build_product_info(session.product_info)

    semantic_points = _build_semantic_points_from_product(product)

    workflow_state: WorkflowState = {
        "session_id": session_id,
        "sales_message": last_user_msg.content,
        "customer_profile": customer,
        "product_info": product,
        "semantic_points": semantic_points,
        "messages": [],
    }

    result = _get_workflow().invoke(workflow_state)
    evaluation = result.get("evaluation_result")

    if not evaluation:
        return EvaluationResponse(
            session_id=session_id,
            coverage_status={},
            coverage_rate=0.0,
            overall_score=0.0,
            expression_analysis={"clarity": 0, "professionalism": 0, "persuasiveness": 0},
        )

    coverage_labels = {sp.point_id: sp.description for sp in semantic_points}

    return EvaluationResponse(
        session_id=session_id,
        coverage_status=evaluation.coverage_status,
        coverage_labels=coverage_labels,
        coverage_rate=evaluation.coverage_rate,
        overall_score=evaluation.overall_score,
        expression_analysis={
            "clarity": evaluation.expression_analysis.clarity,
            "professionalism": evaluation.expression_analysis.professionalism,
            "persuasiveness": evaluation.expression_analysis.persuasiveness,
        },
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    hard: bool = Query(False, description="是否执行硬删除（物理删除，不可恢复）"),
) -> None:
    """删除会话。

    支持软删除和硬删除两种模式：
    - 软删除（默认）：标记为已删除状态，数据保留在数据库中
    - 硬删除（hard=true）：物理删除会话及其所有关联数据

    Args:
        session_id: 会话ID
        hard: 是否执行硬删除

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

    if hard:
        db.hard_delete_session(session_id)
    else:
        db.soft_delete_session(session_id)


@router.delete("/sessions", status_code=status.HTTP_204_NO_CONTENT)
def delete_all_sessions(
    hard: bool = Query(True, description="是否执行硬删除（默认真删除）"),
) -> None:
    """清空所有会话。

    物理删除全部会话、消息和覆盖记录。

    Args:
        hard: 是否执行硬删除（默认True）
    """
    db: DatabaseService = get_db_service()
    if hard:
        db.hard_delete_all_sessions()
    else:
        sessions = db.get_all_sessions()
        for s in sessions:
            db.soft_delete_session(s.id)


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


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions() -> SessionListResponse:
    """获取所有会话列表。

    返回所有非软删除的会话，按创建时间倒序排列，
    用于前端会话切换和历史记录展示。

    Returns:
        会话列表响应
    """
    db: DatabaseService = get_db_service()
    sessions = db.get_all_sessions()

    items = []
    for s in sessions:
        msgs = db.get_messages(s.id)
        items.append(
            SessionListItem(
                session_id=s.id,
                status=s.status,
                created_at=s.created_at or datetime.utcnow(),
                message_count=len(msgs),
            )
        )

    return SessionListResponse(sessions=items, total=len(items))


@router.get("/sessions/{session_id}/messages", response_model=MessagesListResponse)
def list_session_messages(session_id: str) -> MessagesListResponse:
    """获取会话消息历史。

    返回指定会话的所有消息记录，用于会话切换时恢复对话上下文。

    Args:
        session_id: 会话ID

    Returns:
        消息列表响应

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

    msg_items = [
        MessageResponse(
            id=m.id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            turn=m.turn,
            created_at=m.created_at or datetime.utcnow(),
        )
        for m in messages
    ]

    return MessagesListResponse(
        session_id=session_id,
        messages=msg_items,
        total=len(msg_items),
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
