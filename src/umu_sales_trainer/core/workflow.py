"""LangGraph StateGraph 工作流模块。

实现AI销售训练Chatbot的核心工作流，使用Pipeline模式+条件分支架构。
包含6个节点和条件边路由。
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from umu_sales_trainer.core.evaluator import SemanticEvaluator
from umu_sales_trainer.models.conversation import Message
from umu_sales_trainer.models.customer import CustomerProfile
from umu_sales_trainer.models.evaluation import EvaluationResult
from umu_sales_trainer.models.product import ProductInfo
from umu_sales_trainer.models.semantic import SemanticPoint
from umu_sales_trainer.services.embedding import EmbeddingService
from umu_sales_trainer.services.llm import create_llm

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """工作流状态。

    存储整个工作流执行过程中的所有状态数据，在节点间传递。

    Attributes:
        session_id: 会话唯一标识
        sales_message: 销售人员的发言内容
        customer_profile: 客户画像信息
        product_info: 产品信息
        conversation_history: 对话历史消息列表
        semantic_points: 语义点列表（评估标准）
        analysis_result: 分析结果（可选）
        evaluation_result: 评估结果（可选）
        guidance: 引导内容（可选）
        ai_response: AI客户回复（可选）
        next_node: 下一个节点名称（用于条件路由）
    """

    session_id: str
    sales_message: str
    customer_profile: CustomerProfile
    product_info: ProductInfo
    conversation_history: list[Message]
    semantic_points: list[SemanticPoint]
    analysis_result: Optional[dict]
    evaluation_result: Optional[EvaluationResult]
    guidance: Optional[str]
    ai_response: Optional[str]
    next_node: str


def create_workflow() -> "CompiledStateGraph":
    """工厂方法：创建并编译工作流图。

    构建包含6个节点和条件边路由的StateGraph，并返回编译后的图实例。

    Returns:
        编译后的 StateGraph 实例，可通过 invoke() 方法执行
    """
    workflow = StateGraph(WorkflowState)

    workflow.add_node("start", _node_start)
    workflow.add_node("analyze", _node_analyze)
    workflow.add_node("evaluate", _node_evaluate)
    workflow.add_node("guidance", _node_guidance)
    workflow.add_node("simulate", _node_simulate)
    workflow.add_node("end", _node_end)

    workflow.set_entry_point("start")

    workflow.add_conditional_edges(
        "start",
        _route_from_start,
        {
            "analyze": "analyze",
            "end": "end",
        },
    )

    workflow.add_edge("analyze", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        _route_from_evaluate,
        {
            "guidance": "guidance",
            "simulate": "simulate",
        },
    )

    workflow.add_edge("guidance", "simulate")
    workflow.add_edge("simulate", "end")
    workflow.add_edge("end", END)

    return workflow.compile()


def _validate_input(state: WorkflowState) -> bool:
    """验证输入状态是否有效。

    Args:
        state: 工作流状态

    Returns:
        输入是否有效
    """
    return bool(
        state.get("session_id")
        and state.get("sales_message")
        and state.get("customer_profile")
        and state.get("product_info")
    )


def _node_start(state: WorkflowState) -> WorkflowState:
    """入口节点：验证输入。

    检查输入状态的有效性，如果无效则直接结束工作流。
    有效则将控制权传递给 analyze 节点。

    Args:
        state: 当前工作流状态

    Returns:
        更新后的工作流状态
    """
    if _validate_input(state):
        state["next_node"] = "analyze"
    else:
        state["next_node"] = "end"
    return state


def _node_analyze(state: WorkflowState) -> WorkflowState:
    """分析节点：分析销售发言。

    对销售人员的发言进行分析，提取关键信息和上下文。
    这里为简化实现，直接设置 analysis_result。

    Args:
        state: 当前工作流状态

    Returns:
        更新后的工作流状态，包含分析结果
    """
    sales_msg = state.get("sales_message", "")
    customer = state.get("customer_profile", CustomerProfile("", ""))
    state["analysis_result"] = {
        "message_length": len(sales_msg),
        "customer_industry": customer.industry,
        "contains_objection": any(
            keyword in sales_msg for keyword in ["贵", "价格", "不需要", "考虑", "比较"]
        ),
        "analyzed": True,
    }
    state["next_node"] = "evaluate"
    return state


def _node_evaluate(state: WorkflowState) -> WorkflowState:
    """评估节点：使用三层检测机制评估语义点覆盖。

    调用 SemanticEvaluator 执行完整的三层语义检测：
    - 第一层：关键词匹配（权重 0.2）
    - 第二层：Embedding 相似度（权重 0.3）
    - 第三层：LLM 深度判断（权重 0.5）
    同时进行表达能力分析（AgenticRAG）。

    Args:
        state: 当前工作流状态

    Returns:
        更新后的工作流状态，包含完整评估结果
    """
    semantic_points = state.get("semantic_points", [])
    sales_message = state.get("sales_message", "")
    session_id = state.get("session_id", "unknown")

    if not semantic_points or not sales_message:
        logger.warning("Missing semantic_points or sales_message, using default evaluation")
        coverage_status = {sp.point_id: "covered" for sp in semantic_points}
        state["evaluation_result"] = EvaluationResult(
            session_id=session_id,
            coverage_status=coverage_status,
            coverage_rate=1.0 if coverage_status else 0.0,
            overall_score=100.0 if coverage_status else 0.0,
        )
        state["next_node"] = "simulate"
        return state

    try:
        embedding_service = EmbeddingService()
        llm_service = create_llm("dashscope")
        evaluator = SemanticEvaluator(embedding_service, llm_service)

        context = {"session_id": session_id}
        evaluation = evaluator.evaluate(sales_message, semantic_points, context)

        logger.info(
            "Evaluation completed: coverage_rate=%.2f, overall_score=%.1f, "
            "expression=(clarity=%d, pro=%d, pers=%d)",
            evaluation.coverage_rate,
            evaluation.overall_score,
            evaluation.expression_analysis.clarity,
            evaluation.expression_analysis.professionalism,
            evaluation.expression_analysis.persuasiveness,
        )

        state["evaluation_result"] = evaluation
    except Exception as e:
        logger.error("SemanticEvaluator failed, using fallback: %s", e)
        coverage_status = {sp.point_id: "covered" for sp in semantic_points}
        state["evaluation_result"] = EvaluationResult(
            session_id=session_id,
            coverage_status=coverage_status,
            coverage_rate=1.0,
            overall_score=100.0,
        )

    evaluation = state["evaluation_result"]
    state["next_node"] = "guidance" if evaluation.coverage_rate < 0.8 else "simulate"
    return state


def _node_guidance(state: WorkflowState) -> WorkflowState:
    """引导节点：生成销售引导。

    当语义点覆盖不足时，生成针对性的销售引导建议。
    帮助销售人员更好地覆盖关键语义点。

    Args:
        state: 当前工作流状态

    Returns:
        更新后的工作流状态，包含引导内容
    """
    evaluation = state.get("evaluation_result")
    uncovered = (
        [pid for pid, status in evaluation.coverage_status.items() if status != "covered"]
        if evaluation
        else []
    )

    state["guidance"] = (
        f"建议加强以下语义点的覆盖：{', '.join(uncovered)}"
        if uncovered
        else "当前表达已覆盖主要语义点。"
    )
    state["next_node"] = "simulate"
    return state


def _node_simulate(state: WorkflowState) -> WorkflowState:
    """模拟节点：生成AI客户回复。

    根据当前对话上下文，调用LLM或使用规则引擎生成AI客户角色的回复。
    优先使用真实LLM，失败时回退到智能规则回复。

    Args:
        state: 当前工作流状态

    Returns:
        更新后的工作流状态，包含AI客户回复
    """
    customer = state.get("customer_profile", CustomerProfile("", ""))
    sales_msg = state.get("sales_message", "")
    history = state.get("conversation_history", [])

    ai_response = _generate_ai_response(customer, sales_msg, history)

    state["ai_response"] = ai_response
    state["next_node"] = "end"
    return state


def _generate_ai_response(
    customer: CustomerProfile,
    sales_msg: str,
    history: list[Message],
) -> str:
    """生成AI客户回复。

    尝试调用LLM服务生成动态回复，如果失败则使用基于规则的智能回复。

    Args:
        customer: 客户画像
        sales_msg: 销售人员最新消息
        history: 对话历史

    Returns:
        AI客户回复文本
    """
    # 尝试使用LLM
    llm_response = _try_llm_generation(customer, sales_msg, history)
    if llm_response:
        return llm_response

    # 回退到规则引擎
    logger.info("LLM generation failed, falling back to rule-based response")
    return _generate_rule_based_response(customer, sales_msg, history)


def _try_llm_generation(
    customer: CustomerProfile,
    sales_msg: str,
    history: list[Message],
) -> Optional[str]:
    """尝试调用LLM生成回复。

    Returns:
        LLM生成的回复字符串，如果失败返回None
    """
    try:
        from umu_sales_trainer.services.llm import LLMServicesError, create_llm
        from langchain_core.messages import AIMessage

        llm = create_llm("dashscope")

        customer_name = getattr(customer, "name", None) or customer.position
        customer_hospital = getattr(customer, "hospital", "") or "某医院"

        system_prompt = f"""【角色设定】
你正在扮演 **{customer_name}**（{customer.position}，就职于{customer_hospital}）。
你是客户（医生/采购方），坐在你对面的是一位**医药销售代表**。

【你的画像】
- 姓名：{customer_name}
- 职位：{customer.position}
- 机构：{customer_hospital}
- 性格：{customer.personality or '专业审慎'}
- 核心关注点：{", ".join(customer.concerns[:3]) if customer.concerns else '疗效、安全性、价格'}

【对话规则】
1. 你是**客户**，对方是**销售代表**。不要混淆角色。
2. 以"{customer_name}"或"我"自称，称呼对方为"您"或"你们"。
3. 回应要符合{customer.position}的专业身份和性格特点。
4. 自然地提出专业问题或表达关切（围绕关注点展开）。
5. 回应长度控制在50-150字之间。
6. 使用中文回答。"""

        messages = [SystemMessage(content=system_prompt)]

        for msg in history[-6:]:
            if msg.role == "user":
                messages.append(
                    HumanMessage(content=f"[销售代表]: {msg.content}")
                )
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        if sales_msg:
            messages.append(
                HumanMessage(content=f"[销售代表]: {sales_msg}")
            )

        # 调用LLM
        response = llm.invoke(messages)
        ai_text = response.content.strip()

        if ai_text and len(ai_text) > 5:
            logger.info(f"LLM generated response: {ai_text[:50]}...")
            return ai_text

    except (ImportError, LLMServicesError) as e:
        logger.warning(f"LLM service unavailable: {e}")
    except Exception as e:
        logger.error(f"LLM generation error: {e}")

    return None


def _generate_rule_based_response(
    customer: CustomerProfile,
    sales_msg: str,
    history: list[Message],
) -> str:
    """基于规则生成智能回复。

    当LLM不可用时，根据对话上下文和客户画像生成多样化的回复。
    通过分析用户输入关键词和历史轮次来决定回复策略。

    Args:
        customer: 客户画像
        sales_msg: 销售人员消息
        history: 对话历史

    Returns:
        规则生成的回复文本
    """
    turn_count = len([m for m in history if m.role == "user"])

    # 定义不同阶段的回复模板库
    opening_responses = [
        f"您好，我是{customer.position}。听说您有一款新产品要介绍？请说说看。",
        "你好，欢迎来访。我这边时间有限，请直接说明您的来意吧。",
        "您好。我们科室最近在评估新的治疗方案，如果您有好的产品，可以简单介绍一下。",
    ]

    interest_responses = [
        f"嗯，听起来不错。作为{customer.position}，我比较关心产品的{customer.concerns[0] if customer.concerns else '临床数据'}，这方面能详细说说吗？",
        f"有点意思。不过我们医院对这类产品要求比较严格，特别是{customer.concerns[0] if customer.concerns else '安全性'}方面，你们做得怎么样？",
        f"了解了基本情况。我想知道这款产品相比市面上其他方案有什么优势？特别是在{customer.concerns[0] if len(customer.concerns) > 0 else '疗效'}方面。",
    ]

    objection_responses = [
        "这个价格确实需要考虑一下。您知道我们科室的预算情况，能不能申请一些优惠或者分期方案？",
        f"证据方面，除了厂商提供的数据，有没有第三方临床试验的结果？作为{customer.position}，我需要看到更权威的数据支撑。",
        "我理解您的观点，但我们这里之前用过类似的产品，效果并不理想。您能保证这次会有所不同吗？",
    ]

    follow_up_responses = [
        "好的，我记下了。还有其他方面的信息吗？比如用药方案、副作用这些我也比较关注。",
        "明白了。那如果我们在临床上遇到一些特殊情况，比如患者依从性不好，你们有什么支持措施吗？",
        "嗯...让我想想。其实我还想了解一下，这款产品的医保覆盖情况怎么样？这对我们推广很重要。",
    ]

    closing_responses = [
        "好的，今天的信息量不少，我需要时间消化一下。您可以把产品资料留下吗？我会和团队讨论后再联系您。",
        "感谢您的介绍。整体来说印象还不错，但我还需要再对比一下其他方案。我们可以保持联系。",
        "今天先聊到这里吧。如果后续有新的临床数据或者优惠政策，欢迎随时告诉我。",
    ]

    # 分析用户消息关键词
    msg_lower = sales_msg.lower() if sales_msg else ""

    # 判断回复阶段和类型
    if turn_count <= 1:
        # 开场阶段
        if not sales_msg or sales_msg in ["请开始对话", ""]:
            return random.choice(opening_responses)
        return random.choice(interest_responses)

    # 检测关键词决定回复策略
    has_price_keyword = any(kw in msg_lower for kw in ["价格", "贵", "便宜", "预算", "成本"])
    has_evidence_keyword = any(
        kw in msg_lower for kw in ["证据", "研究", "试验", "数据", "文献", "论文"]
    )
    has_safety_keyword = any(kw in msg_lower for kw in ["安全", "副作用", "不良反应", "耐受"])
    has_efficacy_keyword = any(kw in msg_lower for kw in ["效果", "疗效", "控制率", "降低", "改善"])

    if has_price_keyword or any(kw in msg_lower for kw in customer.objection_tendencies):
        return random.choice(objection_responses)

    if has_safety_keyword or has_evidence_keyword:
        return random.choice(
            [
                f"{customer.concerns[0] if customer.concerns else '安全性'}确实是我们最关注的。您提到的这一点很重要，能展开说说具体数据吗？",
                f"关于{'安全性' if has_safety_keyword else '证据支持'}，我想了解更多细节。有没有相关的头对头研究结果？",
                f"这点很关键。在我们科，{'安全性' if has_safety_keyword else '证据充分性'}是首要考虑因素。",
            ]
        )

    if has_efficacy_keyword:
        return random.choice(
            [
                "疗效数据看起来不错。但实际临床应用中，患者的依从性怎么样？这点我很关心。",
                "降糖效果是基础，但我们更看重长期的安全性 profile。这方面的数据呢？",
                "控制率数据令人印象深刻。不过我想知道，对于特殊人群比如老年人，效果如何？",
            ]
        )

    # 根据轮次选择回复
    if turn_count >= 5:
        return random.choice(closing_responses)
    elif turn_count >= 3:
        return random.choice(follow_up_responses)
    else:
        # 中间轮次，混合使用兴趣和追问
        all_mid = interest_responses + follow_up_responses
        return random.choice(all_mid)


def _node_end(state: WorkflowState) -> WorkflowState:
    """结束节点：处理结束逻辑。

    执行工作流结束时的清理和收尾工作。

    Args:
        state: 当前工作流状态

    Returns:
        最终工作流状态
    """
    state["next_node"] = END
    return state


def _route_from_start(state: WorkflowState) -> Literal["analyze", "end"]:
    """从 start 节点的条件路由。

    根据验证结果决定下一步路由。

    Args:
        state: 当前工作流状态

    Returns:
        下一节点名称
    """
    return "analyze" if state.get("next_node") == "analyze" else "end"


def _route_from_evaluate(state: WorkflowState) -> Literal["guidance", "simulate"]:
    """从 evaluate 节点的条件路由。

    根据评估结果决定是否需要引导或直接进入模拟。

    Args:
        state: 当前工作流状态

    Returns:
        下一节点名称
    """
    next_node = state.get("next_node", "simulate")
    if next_node in ("guidance", "simulate"):
        return next_node
    return "simulate"


_workflow_instance: Optional["CompiledStateGraph"] = None


def get_workflow() -> "CompiledStateGraph":
    """获取工作流单例实例。

    Returns:
        编译后的 StateGraph 实例
    """
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = create_workflow()
    return _workflow_instance


def invoke(state: WorkflowState) -> WorkflowState:
    """执行工作流。

    接收初始状态，执行完整的工作流流程，返回最终状态。

    Args:
        state: 初始工作流状态

    Returns:
        最终工作流状态
    """
    workflow = get_workflow()
    result = workflow.invoke(state)
    return dict(result)
