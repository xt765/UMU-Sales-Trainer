"""Agentic RAG 工作流模块。

基于 LangGraph StateGraph 实现多 Agent 协作的销售训练评估流水线。
8 个节点，每个节点对应一个明确的 Agent 或数据处理步骤：

    start → conversation_analyze → semantic_eval → expression_eval
      → synthesize → [guidance] → simulate → end

各节点与 Agent 的映射关系：
- conversation_analyze: ConversationAnalyst（对话分析师）
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

from umu_sales_trainer.core.analyzer import (
    ConversationAnalysis,
    ConversationAnalyst,
)
from umu_sales_trainer.core.evaluator import (
    CoverageResult,
    ExpressionCoach,
    ExpressionResult,
    SemanticCoverageExpert,
    calculate_overall_score,
)
from umu_sales_trainer.core.guidance import GuidanceMentor, GuidanceResult
from umu_sales_trainer.core.response_predictor import ResponsePredictor
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
        conversation_analysis: 对话分析结果（中间态）
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
    conversation_analysis: Optional[ConversationAnalysis] = None
    guidance_result: Optional[GuidanceResult] = None
    ai_response: str = ""
    predicted_responses: list[dict] = []
    error: str = ""


def create_workflow(
    embedding_service: EmbeddingService,
    llm_service: LLMService,
) -> StateGraph:
    """创建 Agentic RAG 工作流图。

    初始化 4 个 Agent 并构建 8 节点有向图。

    Args:
        embedding_service: 向量嵌入服务实例
        llm_service: LLM 服务实例

    Returns:
        编译好的 StateGraph 工作流
    """
    conversation_analyst = ConversationAnalyst(llm_service)
    semantic_expert = SemanticCoverageExpert(embedding_service, llm_service)
    expression_coach = ExpressionCoach(llm_service)
    guidance_mentor = GuidanceMentor(llm_service)

    graph = StateGraph(WorkflowState)

    graph.add_node("start", _node_start)
    graph.add_node("parallel_fanout", _node_parallel_fanout)
    graph.add_node("conversation_analyze", _make_node_conversation_analyze(conversation_analyst))
    graph.add_node("semantic_eval", _make_node_semantic_eval(semantic_expert))
    graph.add_node("expression_eval", _make_node_expression_eval(expression_coach))
    graph.add_node("synthesize", _node_synthesize)
    graph.add_node("guidance", _make_node_guidance(guidance_mentor))
    graph.add_node("simulate", _node_simulate)
    graph.add_node("end", _node_end)

    graph.set_entry_point("start")
    graph.add_edge("start", "parallel_fanout")

    # Fan-out: 三个独立分析 Agent 并行执行
    graph.add_edge("parallel_fanout", "conversation_analyze")
    graph.add_edge("parallel_fanout", "semantic_eval")
    graph.add_edge("parallel_fanout", "expression_eval")

    # Fan-in: 汇聚到综合节点（LangGraph 自动等待所有入边完成）
    graph.add_edge("conversation_analyze", "synthesize")
    graph.add_edge("semantic_eval", "synthesize")
    graph.add_edge("expression_eval", "synthesize")
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


def _node_parallel_fanout(state: WorkflowState) -> dict[str, Any]:
    """并行分发节点：将工作流拆分为三个并行分析分支。

    此节点本身不做任何计算，仅作为 LangGraph fan-out 的路由点，
    将执行流同时分发给 conversation_analyze / semantic_eval / expression_eval。

    Args:
        state: 当前工作流状态

    Returns:
        空状态更新（纯路由节点）
    """
    logger.info("[parallel_fanout] Dispatching to 3 parallel analysis agents")
    return {}


def _make_node_conversation_analyze(analyst: ConversationAnalyst):
    """创建对话分析节点的工厂函数。

    Args:
        analyst: ConversationAnalyst 实例

    Returns:
        节点函数
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        """对话分析节点：调用 ConversationAnalyst 分析销售发言。

        识别销售阶段、意图、异议信号和情感倾向。

        Args:
            state: 工作流状态

        Returns:
            包含 conversation_analysis 的状态更新
        """
        logger.info("[conversation_analyze] Analyzing message intent and stage")

        analysis = analyst.analyze(
            sales_message=state["sales_message"],
            customer_profile=state.get("customer_profile"),
            conversation_history=state.get("messages"),
        )

        logger.info(
            "[conversation_analyze] stage=%s, intent=%s, objections=%s",
            analysis.stage,
            analysis.intent,
            analysis.objections,
        )

        return {"conversation_analysis": analysis}

    return _node


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
    """综合节点：合并三个 Agent 的结果为统一 EvaluationResult。

    将 coverage_result、expression_result 和 conversation_analysis
    聚合为前端可用的统一评估对象。

    Args:
        state: 工作流状态

    Returns:
        包含 evaluation_result 的状态更新
    """
    logger.info("[synthesize] Merging agent results into unified EvaluationResult")

    coverage = state.get("coverage_result") or CoverageResult()
    expression = state.get("expression_result") or ExpressionResult()

    messages = state.get("messages", [])
    turn = sum(1 for m in messages if m.role == "user") + 1

    overall = calculate_overall_score(coverage, expression, turn=turn)

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

    以下任一条件满足时走 guidance 节点：
    - 覆盖率低于 80%（语义点未覆盖完整）
    - 综合评分低于 70 分（表达能力需改进）

    仅当覆盖率 ≥ 80% 且评分 ≥ 70 时才跳过引导（真正优秀）。

    Args:
        state: 工作流状态

    Returns:
        "yes" 或 "no"
    """
    coverage = state.get("coverage_result")
    eval_result = state.get("evaluation_result")
    overall_score = eval_result.overall_score if eval_result else 0.0

    if coverage and coverage.coverage_rate < 0.8:
        return "yes"
    if overall_score < 70:
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

        综合三个 Agent 结果，按紧急度排序改进项。

        Args:
            state: 工作流状态

        Returns:
            包含 guidance_result 的状态更新
        """
        eval_result = state.get("evaluation_result")
        overall_score = eval_result.overall_score if eval_result else 0.0

        logger.info(
            "[guidance] Generating structured guidance | coverage=%.2f overall_score=%s",
            (state.get("coverage_result") or CoverageResult()).coverage_rate,
            overall_score,
        )

        result = mentor.generate_guidance(
            coverage_result=state.get("coverage_result") or CoverageResult(),
            expression_result=state.get("expression_result") or ExpressionResult(),
            conversation_analysis=state.get("conversation_analysis"),
            semantic_points=state.get("semantic_points", []),
            customer_profile=state.get("customer_profile"),
            overall_score=overall_score,
        )

        logger.info(
            "[guidance] actionable=%s, items=%d",
            result.is_actionable,
            len(result.priority_list),
        )

        return {"guidance_result": result}

    return _node


def _node_simulate(state: WorkflowState) -> dict[str, Any]:
    """客户模拟节点：生成 AI 客户回复 + 三策略预测回复。

    基于客户画像、产品信息和完整评估结果，
    使用 LLM 生成符合角色设定的客户回复，
    同时通过 ResponsePredictor 生成 3 个不同策略的预测回复选项。

    Args:
        state: 工作流状态

    Returns:
        包含 ai_response 和 predicted_responses 的状态更新
    """
    logger.info("[simulate] Generating AI customer response + predictions")

    try:
        llm = create_llm("dashscope")
        response_text = _generate_ai_response(state, llm)

        if not response_text or len(response_text) < 5:
            response_text = _generate_fallback_response(state)

        predictor = ResponsePredictor(llm_service=llm)
        predictions = predictor.predict(
            sales_message=state["sales_message"],
            last_ai_response=response_text,
            context={
                "conversation_analysis": state.get("conversation_analysis"),
                "customer_profile": state.get("customer_profile"),
                "coverage_result": state.get("coverage_result"),
            },
        )
        predicted_dicts = [
            {
                "strategy": p.strategy,
                "strategy_label": p.strategy_label,
                "content": p.content,
                "confidence": p.confidence,
                "source_hints": p.source_hints,
            }
            for p in predictions
        ]

        logger.info(
            "[simulate] Generated %d predictions with avg confidence %.2f",
            len(predicted_dicts),
            sum(p["confidence"] for p in predicted_dicts) / max(len(predicted_dicts), 1),
        )

        return {
            "ai_response": response_text,
            "predicted_responses": predicted_dicts,
        }

    except Exception as e:
        logger.error("[simulate] Failed to generate AI response: %s", e)
        return {
            "ai_response": _generate_fallback_response(state),
            "predicted_responses": [],
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
    conv_analysis = state.get("conversation_analysis")
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
        if conv_analysis:
            stage_label = conv_analysis.stage.replace("_", "-")
            context_suffix += f"\n（当前销售处于{stage_label}阶段）"
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

    根据对话分析的阶段信息选择合适的模板回复。

    Args:
        state: 工作流状态

    Returns:
        默认回复文本
    """
    customer = state.get("customer_profile") or CustomerProfile()
    name = customer.name or "张主任"
    conv = state.get("conversation_analysis")

    stage_responses = {
        "opening": f"{name}：您好，请简要介绍一下您今天想聊什么？我时间比较紧。",
        "needs_discovery": f"{name}：嗯，我想了解更多细节。您能具体说说这个产品的优势在哪里吗？",
        "presentation": f"{name}：听起来不错，但我需要看到更多的临床数据支撑。有没有头对头的研究？",
        "objection_handling": f"{name}：我理解您的说法，但这一点我还是有些顾虑...",
        "closing": f"{name}：好的，让我再考虑一下。您可以先发一份详细资料给我。",
    }

    stage = conv.stage if conv else "opening"
    return stage_responses.get(stage, f"{name}：我明白了，请继续。")


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
