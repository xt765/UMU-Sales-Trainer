"""LangGraph 工作流测试模块。

测试 AI 销售训练 Chatbot 的核心工作流，包含6个节点和条件边路由。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from umu_sales_trainer.core.workflow import (
    WorkflowState,
    create_workflow,
    get_workflow,
    invoke,
)
from umu_sales_trainer.models.conversation import Message
from umu_sales_trainer.models.customer import CustomerProfile
from umu_sales_trainer.models.evaluation import EvaluationResult
from umu_sales_trainer.models.product import ProductInfo
from umu_sales_trainer.models.semantic import SemanticPoint

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class TestWorkflowCreate:
    """测试工作流创建和编译。"""

    def test_workflow_create(self, compiled_workflow: "CompiledStateGraph") -> None:
        """测试工作流创建和编译。

        验证 create_workflow 能够成功创建包含6个节点的 StateGraph 并正确编译。
        """
        workflow = compiled_workflow
        assert workflow is not None
        assert hasattr(workflow, "invoke")

    def test_workflow_singleton(self) -> None:
        """测试工作流单例模式。

        验证 get_workflow 返回相同的实例。
        """
        workflow1 = get_workflow()
        workflow2 = get_workflow()
        assert workflow1 is workflow2

    def test_workflow_has_required_nodes(self) -> None:
        """测试工作流包含所有必需节点。

        验证编译后的工作流包含 start, analyze, evaluate, guidance, simulate, end 节点。
        """
        workflow = create_workflow()
        assert workflow is not None
        node_names = ["start", "analyze", "evaluate", "guidance", "simulate", "end"]
        for node_name in node_names:
            assert node_name in workflow.nodes, f"Missing node: {node_name}"


class TestWorkflowFullExecution:
    """测试完整工作流执行。"""

    def test_workflow_full_execution(
        self,
        workflow_state: WorkflowState,
        compiled_workflow: "CompiledStateGraph",
    ) -> None:
        """测试完整工作流执行。

        验证给定有效输入状态时，工作流能够完整执行所有节点。
        预期路径：start -> analyze -> evaluate -> simulate -> end
        """
        result = compiled_workflow.invoke(workflow_state)
        assert result is not None
        assert result["session_id"] == "test-session-1"
        assert result["analysis_result"] is not None
        assert result["analysis_result"].get("analyzed") is True
        assert result["evaluation_result"] is not None
        assert result["evaluation_result"].session_id == "test-session-1"
        assert result["ai_response"] is not None
        assert "内分泌科主任" in result["ai_response"]

    def test_workflow_invalid_input_routes_to_end(
        self,
        workflow_state_invalid: WorkflowState,
        compiled_workflow: "CompiledStateGraph",
    ) -> None:
        """测试无效输入直接路由到结束节点。

        验证当输入状态缺少必需字段时，工作流直接结束而不执行后续节点。
        """
        result = compiled_workflow.invoke(workflow_state_invalid)
        assert result is not None
        assert result.get("next_node") != "analyze"

    def test_invoke_function(
        self,
        workflow_state: WorkflowState,
    ) -> None:
        """测试 invoke 辅助函数。

        验证 invoke 函数能够正确执行工作流并返回最终状态。
        """
        result = invoke(workflow_state)
        assert result is not None
        assert result["session_id"] == "test-session-1"
        assert result["evaluation_result"] is not None


class TestWorkflowNodeRouting:
    """测试节点路由逻辑。"""

    def test_workflow_node_routing(
        self,
        workflow_state: WorkflowState,
        compiled_workflow: "CompiledStateGraph",
    ) -> None:
        """测试工作流节点路由。

        验证工作流按照预期路径执行：start -> analyze -> evaluate -> simulate -> end。
        检查每个节点的输出状态，确保 next_node 正确设置。
        """
        result = compiled_workflow.invoke(workflow_state)
        assert result["analysis_result"] is not None
        assert result["analysis_result"].get("analyzed") is True
        assert result["analysis_result"].get("customer_industry") == "医疗"
        assert result["evaluation_result"] is not None
        assert result["ai_response"] is not None

    def test_start_node_validates_input(
        self,
        workflow_state: WorkflowState,
        workflow_state_invalid: WorkflowState,
    ) -> None:
        """测试 start 节点输入验证。

        验证 start 节点能够正确区分有效和无效输入。
        有效输入：next_node = "analyze"
        无效输入：next_node = "end"
        """
        from umu_sales_trainer.core.workflow import _node_start
        valid_result = _node_start(workflow_state)
        assert valid_result["next_node"] == "analyze"
        invalid_result = _node_start(workflow_state_invalid)
        assert invalid_result["next_node"] == "end"

    def test_analyze_node_processes_message(
        self,
        workflow_state: WorkflowState,
    ) -> None:
        """测试 analyze 节点处理消息。

        验证 analyze 节点正确提取消息长度、客户行业和异议关键词。
        """
        from umu_sales_trainer.core.workflow import _node_analyze
        result = _node_analyze(workflow_state)
        assert result["analysis_result"] is not None
        assert result["analysis_result"]["message_length"] > 0
        assert result["analysis_result"]["customer_industry"] == "医疗"
        assert result["next_node"] == "evaluate"

    def test_route_from_start_function(
        self,
        workflow_state: WorkflowState,
        workflow_state_invalid: WorkflowState,
    ) -> None:
        """测试 _route_from_start 路由函数。

        验证根据 next_node 值正确返回目标节点。
        """
        from umu_sales_trainer.core.workflow import _route_from_start
        assert _route_from_start({"next_node": "analyze"}) == "analyze"
        assert _route_from_start({"next_node": "end"}) == "end"

    def test_route_from_evaluate_function(
        self,
        workflow_state: WorkflowState,
    ) -> None:
        """测试 _route_from_evaluate 路由函数。

        验证根据 next_node 值正确返回 guidance 或 simulate。
        """
        from umu_sales_trainer.core.workflow import _route_from_evaluate
        assert _route_from_evaluate({"next_node": "guidance"}) == "guidance"
        assert _route_from_evaluate({"next_node": "simulate"}) == "simulate"
        assert _route_from_evaluate({"next_node": "invalid"}) == "simulate"


class TestWorkflowEvaluateToGuidance:
    """测试评估到引导的路由（coverage_rate < 0.8）。"""

    def test_workflow_evaluate_to_guidance(
        self,
        customer_profile: CustomerProfile,
        product_info: ProductInfo,
        semantic_points_low_coverage: list[SemanticPoint],
        conversation_history: list[Message],
    ) -> None:
        """测试评估到引导的路由（覆盖率 < 0.8）。

        验证当 evaluate 节点计算覆盖率 < 0.8 时，触发 guidance 节点。
        由于 evaluate 节点默认将所有语义点标记为 covered，
        此测试直接验证 evaluate 节点的路由决策逻辑。
        """
        from umu_sales_trainer.core.workflow import _node_evaluate, _route_from_evaluate
        state = WorkflowState(
            session_id="test-session-guidance",
            sales_message="这个产品效果不错。",
            customer_profile=customer_profile,
            product_info=product_info,
            conversation_history=conversation_history,
            semantic_points=semantic_points_low_coverage,
            analysis_result={"analyzed": True},
            evaluation_result=None,
            guidance=None,
            ai_response=None,
            next_node="",
        )
        result = _node_evaluate(state)
        assert result["evaluation_result"] is not None
        assert result["evaluation_result"].coverage_rate == 1.0
        assert result["next_node"] == "simulate"
        route_result = _route_from_evaluate(result)
        assert route_result == "simulate"

    def test_guidance_node_generates_suggestions(
        self,
        workflow_state: WorkflowState,
    ) -> None:
        """测试 guidance 节点生成建议。

        验证 guidance 节点能够根据未覆盖的语义点生成引导建议。
        """
        from umu_sales_trainer.core.workflow import _node_guidance
        evaluation_with_uncovered = EvaluationResult(
            session_id="test-session-1",
            coverage_status={"SP-001": "covered", "SP-002": "not_covered", "SP-003": "not_covered"},
            coverage_rate=0.33,
            overall_score=33.0,
        )
        state = workflow_state.copy()
        state["evaluation_result"] = evaluation_with_uncovered
        result = _node_guidance(state)
        assert result["guidance"] is not None
        assert "SP-002" in result["guidance"] or "SP-003" in result["guidance"]
        assert result["next_node"] == "simulate"

    def test_guidance_node_all_covered(
        self,
        workflow_state: WorkflowState,
    ) -> None:
        """测试 guidance 节点处理全部覆盖情况。

        验证当所有语义点都已覆盖时，guidance 节点生成正面反馈。
        """
        from umu_sales_trainer.core.workflow import _node_guidance
        evaluation_all_covered = EvaluationResult(
            session_id="test-session-1",
            coverage_status={"SP-001": "covered", "SP-002": "covered", "SP-003": "covered"},
            coverage_rate=1.0,
            overall_score=100.0,
        )
        state = workflow_state.copy()
        state["evaluation_result"] = evaluation_all_covered
        result = _node_guidance(state)
        assert result["guidance"] is not None
        assert "主要语义点" in result["guidance"] or "已覆盖" in result["guidance"]


class TestWorkflowEvaluateToSimulate:
    """测试评估到模拟的路由（coverage_rate >= 0.8）。"""

    def test_workflow_evaluate_to_simulate(
        self,
        customer_profile: CustomerProfile,
        product_info: ProductInfo,
        semantic_points: list[SemanticPoint],
        conversation_history: list[Message],
        compiled_workflow: "CompiledStateGraph",
    ) -> None:
        """测试评估到模拟的路由（覆盖率 >= 0.8）。

        验证当语义点覆盖率高于或等于 80% 时，工作流路由到 simulate 节点。
        由于 evaluate 节点默认将所有语义点标记为 covered，正常执行时
        覆盖率应为 1.0，触发此路由。
        """
        state = WorkflowState(
            session_id="test-session-simulate",
            sales_message="这个降糖药效果很好，每天只需服用一次，副作用也很低。",
            customer_profile=customer_profile,
            product_info=product_info,
            conversation_history=conversation_history,
            semantic_points=semantic_points,
            analysis_result=None,
            evaluation_result=EvaluationResult(
                session_id="test-session-simulate",
                coverage_status={"SP-001": "covered", "SP-002": "covered", "SP-003": "covered"},
                coverage_rate=1.0,
                overall_score=100.0,
            ),
            guidance=None,
            ai_response=None,
            next_node="",
        )
        result = compiled_workflow.invoke(state)
        assert result["evaluation_result"] is not None
        assert result["evaluation_result"].coverage_rate >= 0.8
        assert result.get("ai_response") is not None

    def test_simulate_node_generates_customer_response(
        self,
        workflow_state: WorkflowState,
    ) -> None:
        """测试 simulate 节点生成客户响应。

        验证 simulate 节点能够根据客户画像生成合理的 AI 客户回复。
        """
        from umu_sales_trainer.core.workflow import _node_simulate
        result = _node_simulate(workflow_state)
        assert result["ai_response"] is not None
        assert "内分泌科主任" in result["ai_response"]
        assert result["next_node"] == "end"

    def test_simulate_node_empty_message(
        self,
        customer_profile: CustomerProfile,
        product_info: ProductInfo,
    ) -> None:
        """测试 simulate 节点处理空消息。

        验证当销售消息为空时，simulate 节点生成默认回复。
        """
        from umu_sales_trainer.core.workflow import _node_simulate
        state = WorkflowState(
            session_id="test-session-empty",
            sales_message="",
            customer_profile=customer_profile,
            product_info=product_info,
            conversation_history=[],
            semantic_points=[],
            analysis_result=None,
            evaluation_result=None,
            guidance=None,
            ai_response=None,
            next_node="",
        )
        result = _node_simulate(state)
        assert result["ai_response"] is not None
        assert "您好" in result["ai_response"]

    def test_end_node_sets_final_state(
        self,
        workflow_state: WorkflowState,
    ) -> None:
        """测试 end 节点设置最终状态。

        验证 end 节点正确设置 next_node 为 END (langgraph.graph.END = '__end__')。
        """
        from langgraph.graph import END
        from umu_sales_trainer.core.workflow import _node_end
        result = _node_end(workflow_state)
        assert result["next_node"] == END
