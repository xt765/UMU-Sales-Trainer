"""LangGraph StateGraph 工作流模块。

实现AI销售训练Chatbot的核心工作流，使用Pipeline模式+条件分支架构。
包含6个节点和条件边路由。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from umu_sales_trainer.models.conversation import Message
from umu_sales_trainer.models.customer import CustomerProfile
from umu_sales_trainer.models.evaluation import EvaluationResult
from umu_sales_trainer.models.product import ProductInfo
from umu_sales_trainer.models.semantic import SemanticPoint


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
            keyword in sales_msg
            for keyword in ["贵", "价格", "不需要", "考虑", "比较"]
        ),
        "analyzed": True,
    }
    state["next_node"] = "evaluate"
    return state


def _node_evaluate(state: WorkflowState) -> WorkflowState:
    """评估节点：评估语义点覆盖。

    根据分析结果评估语义点覆盖情况。
    如果有语义点未覆盖或覆盖率低于阈值，触发引导节点。

    Args:
        state: 当前工作流状态

    Returns:
        更新后的工作流状态，包含评估结果
    """
    semantic_points = state.get("semantic_points", [])
    coverage_status = {sp.point_id: "covered" for sp in semantic_points}

    coverage_rate = (
        sum(1 for v in coverage_status.values() if v == "covered") / len(semantic_points)
        if semantic_points
        else 1.0
    )

    state["evaluation_result"] = EvaluationResult(
        session_id=state.get("session_id", ""),
        coverage_status=coverage_status,
        coverage_rate=coverage_rate,
        overall_score=coverage_rate * 100,
    )

    state["next_node"] = "guidance" if coverage_rate < 0.8 else "simulate"
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

    根据当前对话上下文，生成AI客户角色的回复。
    用于销售训练场景中的客户模拟。

    Args:
        state: 当前工作流状态

    Returns:
        更新后的工作流状态，包含AI客户回复
    """
    customer = state.get("customer_profile", CustomerProfile("", ""))
    sales_msg = state.get("sales_message", "")

    state["ai_response"] = (
        f"作为{customer.position}，我对您的方案很感兴趣。"
        + f"请问{customer.concerns[0] if customer.concerns else '相关细节'}如何？"
        if sales_msg
        else "您好，请问有什么可以帮您的？"
    )
    state["next_node"] = "end"
    return state


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
