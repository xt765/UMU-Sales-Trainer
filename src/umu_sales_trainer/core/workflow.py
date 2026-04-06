"""Agentic RAG 工作流模块。

基于 LangGraph StateGraph 实现多 Agent 协作的销售训练评估流水线。
7 个节点，语义评估与表达评估并行执行以缩短响应时间：

    start → semantic_eval ═══╗
          ════════════════╝
          expression_eval  ═══╗→ synthesize → [guidance] → simulate → end
                           ═══╝

各节点与 Agent 的映射关系：
- semantic_eval: SemanticCoverageExpert（语义覆盖专家）
- expression_eval: ExpressionCoach（表达教练）
- guidance: GuidanceMentor（引导导师）
- simulate: CustomerSimulator（客户模拟器，保持不变）
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from umu_sales_trainer.core.evaluator import (
    CoverageResult,
    ExpressionCoach,
    ExpressionResult,
    SemanticCoverageExpert,
    calculate_overall_score,
)
from umu_sales_trainer.core.guidance import GuidanceMentor, GuidanceResult
from umu_sales_trainer.models.conversation import Message
from umu_sales_trainer.models.customer import CustomerProfile
from umu_sales_trainer.models.evaluation import EvaluationResult
from umu_sales_trainer.models.product import ProductInfo
from umu_sales_trainer.models.semantic import SemanticPoint
from umu_sales_trainer.services.embedding import EmbeddingService
from umu_sales_trainer.services.llm import LLMService, create_llm

logger = logging.getLogger(__name__)


class WorkflowState(dict):
    """工作流状态字典。

    贯穿所有节点的共享状态，包含输入、中间结果和最终输出。

    Attributes:
        session_id: 会话唯一标识
        sales_message: 销售人员最新发言（单条）
        current_message: 当前轮次消息（同 sales_message，用于兼容）
        customer_profile: 客户画像
        product_info: 产品信息
        semantic_points: 语义点列表
        messages: 对话历史消息列表
        evaluation_result: 最终聚合的评估结果
        coverage_result: 语义覆盖检测结果（中间态）
        expression_result: 表达能力评估结果（中间态）
        guidance_result: 智能引导结果（中间态）
        ai_response: AI 生成的客户回复
        error: 错误信息
    """

    session_id: str = ""
    sales_message: str = ""
    current_message: str = ""
    customer_profile: Optional[CustomerProfile] = None
    product_info: Optional[ProductInfo] = None
    semantic_points: list[SemanticPoint] = []
    messages: list[Message] = []
    evaluation_result: Optional[EvaluationResult] = None
    coverage_result: Optional[CoverageResult] = None
    expression_result: Optional[ExpressionResult] = None
    guidance_result: Optional[GuidanceResult] = None
    ai_response: str = ""
    error: str = ""


def create_workflow(
    embedding_service: EmbeddingService,
    llm_service: LLMService,
) -> StateGraph:
    """创建 Agentic RAG 工作流图。

    初始化 3 个 Agent 并构建 7 节点有向图。
    semantic_eval 和 expression_eval 并行执行。

    Args:
        embedding_service: 向量嵌入服务实例
        llm_service: LLM 服务实例

    Returns:
        编译好的 StateGraph 工作流
    """
    semantic_expert = SemanticCoverageExpert(embedding_service, llm_service)
    expression_coach = ExpressionCoach(llm_service)
    guidance_mentor = GuidanceMentor(llm_service)

    graph = StateGraph(WorkflowState)

    graph.add_node("start", _node_start)
    graph.add_node("semantic_eval", _make_node_semantic_eval(semantic_expert))
    graph.add_node("expression_eval", _make_node_expression_eval(expression_coach))
    graph.add_node("synthesize", _node_synthesize)
    graph.add_node("guidance", _make_node_guidance(guidance_mentor))
    graph.add_node("simulate", _node_simulate)
    graph.add_node("end", _node_end)

    graph.set_entry_point("start")
    # 并行扇出：start 同时触发两个独立评估节点
    graph.add_edge("start", "semantic_eval")
    graph.add_edge("start", "expression_eval")
    # 汇合：两个并行节点都完成后进入 synthesize
    graph.add_edge("semantic_eval", "synthesize")
    graph.add_edge("expression_eval", "synthesize")
    # 后续串行链路不变
    graph.add_conditional_edges(
        "synthesize",
        _should_generate_guidance,
        {"yes": "guidance", "no": "simulate"},
    )
    graph.add_edge("guidance", "simulate")
    graph.add_edge("simulate", "end")
    graph.add_edge("end", END)

    return graph.compile()


def _node_start(state: WorkflowState) -> dict[str, Any]:
    """起始节点：验证和初始化输入。

    确保必要字段存在并做基础校验。

    Args:
        state: 当前工作流状态

    Returns:
        更新的状态片段
    """
    logger.info("[start] Initializing workflow for session %s", state.get("session_id", "?"))

    if not state.get("sales_message"):
        return {"error": "No sales message provided"}

    if not state.get("semantic_points"):
        logger.warning("[start] No semantic points provided")

    return {}


def _make_node_semantic_eval(expert: SemanticCoverageExpert):
    """创建语义评估节点的工厂函数。

    Args:
        expert: SemanticCoverageExpert 实例

    Returns:
        节点函数
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        """语义评估节点：调用 SemanticCoverageExpert 检测语义点覆盖。

        执行三层检测（关键词+Embedding+LLM）判断每个语义点的覆盖情况。
        与 expression_eval 并行执行，互不依赖。

        Args:
            state: 工作流状态

        Returns:
            包含 coverage_result 的状态更新
        """
        logger.info("[semantic_eval] Evaluating semantic point coverage")

        result = expert.evaluate_coverage(
            sales_message=state["sales_message"],
            semantic_points=state.get("semantic_points", []),
            context={"session_id": state.get("session_id")},
        )

        logger.info(
            "[semantic_eval] rate=%.2f, uncovered=%s",
            result.coverage_rate,
            result.uncovered_points,
        )

        return {"coverage_result": result}

    return _node


def _make_node_expression_eval(coach: ExpressionCoach):
    """创建表达评估节点的工厂函数。

    Args:
        coach: ExpressionCoach 实例

    Returns:
        节点函数
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        """表达评估节点：调用 ExpressionCoach 评估表达能力。

        三维度评分（清晰度/专业性/说服力）+ 改进建议生成。
        与 semantic_eval 并行执行，互不依赖。

        Args:
            state: 工作流状态

        Returns:
            包含 expression_result 的状态更新
        """
        logger.info("[expression_eval] Evaluating expression quality")

        result = coach.evaluate(
            message=state["sales_message"],
            context={
                "session_id": state.get("session_id"),
                "customer_position": (state.get("customer_profile") or CustomerProfile()).position,
            },
        )

        expr = result.analysis
        logger.info(
            "[expression_eval] clarity=%d, pro=%d, pers=%d, suggestions=%d",
            expr.clarity,
            expr.professionalism,
            expr.persuasiveness,
            len(result.suggestions),
        )

        return {"expression_result": result}

    return _node


def _node_synthesize(state: WorkflowState) -> dict[str, Any]:
    """综合节点：合并两个 Agent 的结果为统一 EvaluationResult。

    将 coverage_result 和 expression_result 聚合为前端可用的统一评估对象。

    Args:
        state: 工作流状态

    Returns:
        包含 evaluation_result 的状态更新
    """
    logger.info("[synthesize] Merging agent results into unified EvaluationResult")

    coverage = state.get("coverage_result") or CoverageResult()
    expression = state.get("expression_result") or ExpressionResult()

    overall = calculate_overall_score(coverage, expression)

    eval_result = EvaluationResult(
        session_id=state.get("session_id", ""),
        coverage_status=coverage.coverage_status,
        coverage_rate=coverage.coverage_rate,
        overall_score=overall,
        expression_analysis=expression.analysis,
    )

    logger.info(
        "[synthesize] overall=%.2f, coverage_rate=%.2f",
        overall,
        coverage.coverage_rate,
    )

    return {"evaluation_result": eval_result}


def _should_generate_guidance(state: WorkflowState) -> str:
    """条件路由：判断是否需要生成引导建议。

    当覆盖率低于 80% 时走 guidance 节点，
    否则直接跳到 simulate 节点。

    Args:
        state: 工作流状态

    Returns:
        "yes" 或 "no"
    """
    coverage = state.get("coverage_result")
    if coverage and coverage.coverage_rate < 0.8:
        return "yes"
    return "no"


def _make_node_guidance(mentor: GuidanceMentor):
    """创建引导节点的工厂函数。

    Args:
        mentor: GuidanceMentor 实例

    Returns:
        节点函数
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        """引导节点：调用 GuidanceMentor 生成结构化引导建议。

        综合两个 Agent 结果，按紧急度排序改进项。

        Args:
            state: 工作流状态

        Returns:
            包含 guidance_result 的状态更新
        """
        logger.info("[guidance] Generating structured guidance")

        result = mentor.generate_guidance(
            coverage_result=state.get("coverage_result") or CoverageResult(),
            expression_result=state.get("expression_result") or ExpressionResult(),
            semantic_points=state.get("semantic_points", []),
            conversation_analysis=None,
            customer_profile=state.get("customer_profile"),
        )

        logger.info(
            "[guidance] actionable=%s, items=%d",
            result.is_actionable,
            len(result.priority_list),
        )

        return {"guidance_result": result}

    return _node


def _node_simulate(state: WorkflowState) -> dict[str, Any]:
    """客户模拟节点：生成 AI 客户回复。

    基于客户画像、产品信息和完整评估结果，
    使用 LLM 生成符合角色设定的客户回复。

    Args:
        state: 工作流状态

    Returns:
        包含 ai_response 的状态更新
    """
    logger.info("[simulate] Generating AI customer response")

    try:
        llm = create_llm("dashscope")
        response_text = _generate_ai_response(state, llm)

        if not response_text or len(response_text) < 5:
            response_text = _generate_fallback_response(state)

        return {"ai_response": response_text}

    except Exception as e:
        logger.error("[simulate] Failed to generate AI response: %s", e)
        return {
            "ai_response": _generate_fallback_response(state),
            "error": f"AI generation failed: {e}",
        }


def _generate_ai_response(state: WorkflowState, llm: LLMService) -> str:
    """使用 LLM 生成 AI 客户回复。

    构建完整的 System Prompt 和历史消息上下文，
    让 AI 以客户身份进行角色扮演式回复。

    Args:
        state: 工作流状态
        llm: LLM 服务实例

    Returns:
        AI 生成的客户回复文本
    """
    customer = state.get("customer_profile") or CustomerProfile()
    product = state.get("product_info") or ProductInfo()
    messages = state.get("messages", [])
    sales_msg = state.get("sales_message", "")
    eval_result = state.get("evaluation_result")

    customer_name = customer.name or "医生"
    customer_hospital = customer.hospital or "某医院"
    customer_position = customer.position or "科室主任"

    system_prompt = _build_customer_system_prompt(
        customer_name, customer_hospital, customer_position, customer, product
    )

    langchain_messages: list[Any] = [SystemMessage(content=system_prompt)]

    for msg in messages[-6:]:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=f"[销售代表]: {msg.content}"))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))

    if sales_msg:
        context_suffix = ""
        if eval_result:
            context_suffix += f"\n（您的表现评分：{eval_result.overall_score:.0f}/100分）"
        langchain_messages.append(HumanMessage(content=f"[销售代表]: {sales_msg}{context_suffix}"))

    response = llm.invoke(langchain_messages)
    return response.content.strip() if hasattr(response, "content") else str(response)


def _build_customer_system_prompt(
    name: str,
    hospital: str,
    position: str,
    customer: CustomerProfile,
    product: ProductInfo,
) -> str:
    """构建客户角色扮演的系统提示词。

    明确定义 AI 应该扮演的角色、性格特征和对话规则，
    避免角色混淆问题。

    Args:
        name: 客户姓名
        hospital: 客户机构
        position: 客户职位
        customer: 完整客户画像
        product: 产品信息

    Returns:
        格式化的系统提示词
    """
    concerns_text = ", ".join(customer.concerns[:3]) if customer.concerns else "疗效、安全性、价格"
    personality_text = customer.personality or "专业审慎，注重数据和循证医学证据"

    return f"""【角色设定】
你正在扮演 **{name}**（{position}，就职于{hospital}）。
你是**客户**（医生/采购方），坐在你对面的是一位**医药销售代表**。

【你的画像】
- 姓名：{name}
- 职位：{position}
- 机构：{hospital}
- 性格：{personality_text}
- 核心关注点：{concerns_text}
- 产品认知度：对「{product.name or "该产品"}」有一定了解但持谨慎态度

【对话规则】
1. 你是**客户**，对方是**销售代表**。不要混淆角色。
2. 以"{name}"或"我"自称，称呼对方为"您"或"你们"。
3. 回复长度控制在50-150字之间，像真实对话一样自然。
4. 根据你的性格特点回应：
   - 如果对方提到数据，你会追问来源和样本量
   - 如果对方只说优点，你会主动询问副作用或局限性
   - 如果对方说得太笼统，你会要求具体说明
5. 可以适当提出反对意见或顾虑（这是真实的销售场景）。
6. 不要总是同意对方——真实的客户会有疑虑。"""


def _generate_fallback_response(state: WorkflowState) -> str:
    """降级方案：当 LLM 不可用时生成默认回复。

    使用通用模板回复，不再依赖 conversation_analysis 阶段信息。

    Args:
        state: 工作流状态

    Returns:
        默认回复文本
    """
    customer = state.get("customer_profile") or CustomerProfile()
    name = customer.name or "张主任"

    fallback_responses = [
        f"{name}：嗯，我明白了。您能再详细说说这个产品的具体优势吗？",
        f"{name}：听起来不错。不过我还是想了解一下有没有相关的临床数据支撑？",
        f"{name}：好的，这一点我记下了。那关于安全性方面呢？有什么数据可以参考？",
    ]

    import random

    return random.choice(fallback_responses)


def _node_end(state: WorkflowState) -> dict[str, Any]:
    """结束节点：清理和日志记录。

    记录最终状态摘要用于调试追踪。

    Args:
        state: 工作流状态

    Returns:
        空状态更新（无修改）
    """
    logger.info(
        "[end] Workflow complete - session=%s, score=%.2f, ai_len=%d",
        state.get("session_id"),
        (state.get("evaluation_result") or EvaluationResult()).overall_score,
        len(state.get("ai_response", "")),
    )
    return {}
