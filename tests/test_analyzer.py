"""SalesAnalyzer 模块测试。

测试销售发言分析器的各项功能，包括基本分析、上下文分析、
表达质量提取和语义点覆盖检测。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from umu_sales_trainer.core.analyzer import SalesAnalyzer
from umu_sales_trainer.models.conversation import Message
from umu_sales_trainer.models.customer import CustomerProfile
from umu_sales_trainer.models.evaluation import ExpressionAnalysis
from umu_sales_trainer.models.product import ProductInfo
from umu_sales_trainer.models.semantic import SemanticPoint
from umu_sales_trainer.services.llm import LLMService


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """创建模拟的 LLM 服务。

    Returns:
        包含 invoke 方法的 MagicMock 实例
    """
    mock = MagicMock(spec=LLMService)
    mock.invoke.return_value = MagicMock(
        content='{"key_information_points":["信息点1","信息点2"],"expression_analysis":{"clarity":8,"professionalism":7,"persuasiveness":9},"coverage_status":{"SP-001":"covered"}}'
    )
    return mock


@pytest.fixture
def analyzer(mock_llm_service: MagicMock) -> SalesAnalyzer:
    """创建 SalesAnalyzer 实例。

    Args:
        mock_llm_service: 模拟的 LLM 服务

    Returns:
        SalesAnalyzer 实例
    """
    return SalesAnalyzer(llm_service=mock_llm_service)


@pytest.fixture
def sample_customer_profile() -> CustomerProfile:
    """创建示例客户画像。

    Returns:
        包含医疗行业内分泌科主任信息的客户画像
    """
    return CustomerProfile(
        industry="医疗",
        position="内分泌科主任",
        concerns=["药品疗效", "安全性", "价格"],
        personality="谨慎型",
    )


@pytest.fixture
def sample_product_info() -> ProductInfo:
    """创建示例产品信息。

    Returns:
        包含降糖药信息的产品详情
    """
    return ProductInfo(
        name="降糖药物X",
        description="新一代降糖药物，每日一次服用",
        core_benefits=["降糖效果好", "副作用低", "使用方便"],
    )


@pytest.fixture
def sample_semantic_points() -> list[SemanticPoint]:
    """创建示例语义点列表。

    Returns:
        包含两个语义点的列表
    """
    return [
        SemanticPoint(
            point_id="SP-001",
            description="强调产品降糖效果",
            keywords=["降糖", "效果", "显著"],
            weight=1.0,
        ),
        SemanticPoint(
            point_id="SP-002",
            description="说明产品安全性",
            keywords=["安全", "副作用", "低风险"],
            weight=1.0,
        ),
    ]


@pytest.fixture
def sample_conversation_history() -> list[Message]:
    """创建示例对话历史。

    Returns:
        包含两条消息的对话历史
    """
    return [
        Message(session_id="test-session", role="user", content="这个药效果怎么样？", turn=1),
        Message(session_id="test-session", role="assistant", content="我们的药效果非常好。", turn=2),
    ]


@pytest.mark.asyncio
async def test_analyze_basic(analyzer: SalesAnalyzer) -> None:
    """测试基本分析功能。

    验证 analyze 方法能够正确处理简单输入并返回预期的结果结构。
    """
    result = analyzer.analyze(sales_message="我们的产品降糖效果非常好。", context={})

    assert "key_information_points" in result
    assert "expression_analysis" in result
    assert isinstance(result["expression_analysis"], ExpressionAnalysis)
    assert 0 <= result["expression_analysis"].clarity <= 10
    assert 0 <= result["expression_analysis"].professionalism <= 10
    assert 0 <= result["expression_analysis"].persuasiveness <= 10


@pytest.mark.asyncio
async def test_analyze_with_context(
    analyzer: SalesAnalyzer,
    sample_customer_profile: CustomerProfile,
    sample_product_info: ProductInfo,
    sample_conversation_history: list[Message],
) -> None:
    """测试带上下文的分析功能。

    验证当提供客户画像、产品信息和对话历史时，分析器能够正确处理上下文。
    """
    context: dict[str, Any] = {
        "customer_profile": sample_customer_profile,
        "product_info": sample_product_info,
        "conversation_history": sample_conversation_history,
    }

    result = analyzer.analyze(
        sales_message="我们的产品采用先进的技术，降糖效果显著。",
        context=context,
    )

    assert "key_information_points" in result
    assert len(result["key_information_points"]) > 0
    assert "expression_analysis" in result


@pytest.mark.asyncio
async def test_analyze_quality_extraction(analyzer: SalesAnalyzer) -> None:
    """测试表达质量提取功能。

    验证 expression_analysis 包含清晰度、专业性和说服力三个维度的评分。
    """
    result = analyzer.analyze(
        sales_message="我们的产品采用最新科技，降糖效果显著，副作用极低。",
        context={},
    )

    expr_analysis = result["expression_analysis"]
    assert hasattr(expr_analysis, "clarity")
    assert hasattr(expr_analysis, "professionalism")
    assert hasattr(expr_analysis, "persuasiveness")
    assert isinstance(expr_analysis.clarity, int)
    assert isinstance(expr_analysis.professionalism, int)
    assert isinstance(expr_analysis.persuasiveness, int)


@pytest.mark.asyncio
async def test_analyze_semantic_coverage(
    analyzer: SalesAnalyzer,
    sample_semantic_points: list[SemanticPoint],
) -> None:
    """测试语义点覆盖检测功能。

    验证 coverage_status 和 coverage_rate 能够正确反映语义点覆盖情况。
    """
    context: dict[str, Any] = {
        "semantic_points": sample_semantic_points,
    }

    result = analyzer.analyze(
        sales_message="我们的产品降糖效果非常好，安全性也很高。",
        context=context,
    )

    assert "coverage_status" in result
    assert "coverage_rate" in result
    assert isinstance(result["coverage_status"], dict)
    assert isinstance(result["coverage_rate"], float)
    assert 0.0 <= result["coverage_rate"] <= 1.0
    for sp in sample_semantic_points:
        assert sp.point_id in result["coverage_status"]
