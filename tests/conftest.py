"""Pytest fixtures for tests.

提供测试所需的共享 fixtures，包括 EmbeddingService、LLMService、SemanticEvaluator 的 mock 对象
以及 LangGraph 工作流测试所需的 fixtures。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from umu_sales_trainer.core.evaluator import SemanticEvaluator
from umu_sales_trainer.core.guidance import GuidanceGenerator
from umu_sales_trainer.core.workflow import WorkflowState, create_workflow
from umu_sales_trainer.models.conversation import Message
from umu_sales_trainer.models.customer import CustomerProfile
from umu_sales_trainer.models.evaluation import ExpressionAnalysis
from umu_sales_trainer.models.product import ProductInfo, SellingPoint
from umu_sales_trainer.models.semantic import SemanticPoint
from umu_sales_trainer.services.embedding import EmbeddingService
from umu_sales_trainer.services.llm import LLMService

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """创建 Mock EmbeddingService。

    Returns:
        模拟的 EmbeddingService 实例
    """
    mock = MagicMock(spec=EmbeddingService)
    mock.encode_query.return_value = [0.1] * 384
    return mock


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """创建 Mock LLMService。

    Returns:
        模拟的 LLMService 实例
    """
    mock = MagicMock(spec=LLMService)
    mock.invoke.return_value = MagicMock(content="1")
    return mock


@pytest.fixture
def semantic_evaluator(
    mock_embedding_service: MagicMock, mock_llm_service: MagicMock
) -> SemanticEvaluator:
    """创建 SemanticEvaluator 实例。

    Args:
        mock_embedding_service: Mock 向量嵌入服务
        mock_llm_service: Mock LLM 服务

    Returns:
        SemanticEvaluator 实例
    """
    return SemanticEvaluator(
        embedding_service=mock_embedding_service,
        llm_service=mock_llm_service,
    )


@pytest.fixture
def sample_semantic_points() -> list[SemanticPoint]:
    """创建示例语义点列表。

    Returns:
        包含测试用语义点的列表
    """
    return [
        SemanticPoint(
            point_id="SP-001",
            description="介绍产品的降糖效果",
            keywords=["降糖", "血糖", "效果"],
        ),
        SemanticPoint(
            point_id="SP-002",
            description="说明药品的安全性",
            keywords=["安全", "副作用", "耐受"],
        ),
    ]


@pytest.fixture
def sample_context() -> dict[str, str]:
    """创建示例评估上下文。

    Returns:
        包含 session_id 的上下文字典
    """
    return {"session_id": "test-session-001"}


@pytest.fixture
def expression_analysis() -> ExpressionAnalysis:
    """创建示例表达分析结果。

    Returns:
        预设的 ExpressionAnalysis 实例
    """
    return ExpressionAnalysis(
        clarity=8,
        professionalism=7,
        persuasiveness=9,
    )


@pytest.fixture
def customer_profile() -> CustomerProfile:
    """创建测试用客户画像。

    Returns:
        包含医疗行业内分泌科主任的客户画像
    """
    return CustomerProfile(
        industry="医疗",
        position="内分泌科主任",
        concerns=["价格", "疗效", "副作用"],
        personality="谨慎型",
        objection_tendencies=["价格异议", "效果疑虑"],
    )


@pytest.fixture
def product_info() -> ProductInfo:
    """创建测试用产品信息。

    Returns:
        包含降糖药的产品信息
    """
    selling_points = {
        "SP-001": SellingPoint(
            point_id="SP-001",
            description="降糖效果好",
            keywords=["降糖", "血糖", "效果好"],
            sample_phrases=["您的血糖控制得很好"],
        ),
        "SP-002": SellingPoint(
            point_id="SP-002",
            description="副作用低",
            keywords=["副作用", "安全", "耐受"],
            sample_phrases=["这款药物安全性高"],
        ),
        "SP-003": SellingPoint(
            point_id="SP-003",
            description="服用方便",
            keywords=["方便", "简单", "依从"],
            sample_phrases=["每天只需服用一次"],
        ),
    }
    return ProductInfo(
        name="降糖药A",
        description="一种新型降糖药物",
        core_benefits=["降糖效果好", "副作用低", "服用方便"],
        key_selling_points=selling_points,
    )


@pytest.fixture
def semantic_points() -> list[SemanticPoint]:
    """创建测试用语义点列表（高覆盖率场景）。

    Returns:
        包含3个语义点的列表
    """
    return [
        SemanticPoint(
            point_id="SP-001",
            description="降糖效果好",
            keywords=["降糖", "血糖", "效果好"],
            weight=1.0,
        ),
        SemanticPoint(
            point_id="SP-002",
            description="副作用低",
            keywords=["副作用", "安全", "耐受"],
            weight=1.0,
        ),
        SemanticPoint(
            point_id="SP-003",
            description="服用方便",
            keywords=["方便", "简单", "依从"],
            weight=1.0,
        ),
    ]


@pytest.fixture
def semantic_points_low_coverage() -> list[SemanticPoint]:
    """创建测试用语义点列表（低覆盖率场景）。

    用于测试评估到引导的路由（coverage_rate < 0.8）。

    Returns:
        包含3个语义点的列表
    """
    return [
        SemanticPoint(
            point_id="SP-001",
            description="降糖效果好",
            keywords=["降糖", "血糖", "效果好"],
            weight=1.0,
        ),
        SemanticPoint(
            point_id="SP-002",
            description="副作用低",
            keywords=["副作用", "安全", "耐受"],
            weight=1.0,
        ),
        SemanticPoint(
            point_id="SP-003",
            description="服用方便",
            keywords=["方便", "简单", "依从"],
            weight=1.0,
        ),
    ]


@pytest.fixture
def conversation_history() -> list[Message]:
    """创建测试用对话历史。

    Returns:
        包含3条消息的对话历史
    """
    return [
        Message(session_id="test-session-1", role="user", content="您好，我想了解一下这个产品", turn=1),
        Message(session_id="test-session-1", role="assistant", content="您好，这款降糖药效果很好", turn=2),
        Message(session_id="test-session-1", role="user", content="价格是多少？", turn=3),
    ]


@pytest.fixture
def workflow_state(
    customer_profile: CustomerProfile,
    product_info: ProductInfo,
    semantic_points: list[SemanticPoint],
    conversation_history: list[Message],
) -> WorkflowState:
    """创建测试用工作流状态（有效输入）。

    Args:
        customer_profile: 客户画像
        product_info: 产品信息
        semantic_points: 语义点列表
        conversation_history: 对话历史

    Returns:
        包含完整有效数据的工作流状态
    """
    return WorkflowState(
        session_id="test-session-1",
        sales_message="这个降糖药效果很好，每天只需服用一次，副作用也很低。",
        customer_profile=customer_profile,
        product_info=product_info,
        conversation_history=conversation_history,
        semantic_points=semantic_points,
        analysis_result=None,
        evaluation_result=None,
        guidance=None,
        ai_response=None,
        next_node="",
    )


@pytest.fixture
def workflow_state_invalid() -> WorkflowState:
    """创建测试用工作流状态（无效输入）。

    用于测试输入验证失败时直接路由到结束节点。

    Returns:
        缺少必需字段的工作流状态
    """
    return WorkflowState(
        session_id="",
        sales_message="",
        customer_profile=CustomerProfile("", ""),
        product_info=ProductInfo(name=""),
        conversation_history=[],
        semantic_points=[],
        analysis_result=None,
        evaluation_result=None,
        guidance=None,
        ai_response=None,
        next_node="",
    )


@pytest.fixture
def guidance_generator(mock_llm_service: MagicMock) -> GuidanceGenerator:
    """创建 GuidanceGenerator 实例。

    Args:
        mock_llm_service: Mock LLM 服务

    Returns:
        GuidanceGenerator 实例
    """
    return GuidanceGenerator(llm=mock_llm_service)


@pytest.fixture
def compiled_workflow() -> "CompiledStateGraph[object, object]":
    """创建并返回编译后的工作流图。

    Returns:
        编译后的 LangGraph StateGraph 实例
    """
    return create_workflow()
