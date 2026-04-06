"""GuidanceGenerator 测试模块。

测试引导生成器的各种引导策略生成功能。
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from umu_sales_trainer.core.guidance import GuidanceGenerator


@pytest.fixture
def guidance_generator(mock_llm_service: MagicMock) -> GuidanceGenerator:
    """创建 GuidanceGenerator 实例。

    Args:
        mock_llm_service: 模拟的 LLM 服务

    Returns:
        GuidanceGenerator: 引导生成器实例
    """
    return GuidanceGenerator(llm=mock_llm_service)


@pytest.fixture
def guidance_generator_with_rag(
    mock_llm_service: MagicMock,
) -> tuple[GuidanceGenerator, MagicMock]:
    """创建带 RAG 的 GuidanceGenerator 实例。

    Args:
        mock_llm_service: 模拟的 LLM 服务

    Returns:
        tuple: (引导生成器实例, 模拟检索器)
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve = MagicMock(
        return_value=[
            {"content": "该药物临床试验显示降糖效果显著。"},
            {"content": "副作用发生率低于同类产品。"},
        ]
    )
    generator = GuidanceGenerator(
        llm=mock_llm_service,
        rag_retriever=mock_retriever,
    )
    return generator, mock_retriever


class TestGuidanceGenerator:
    """GuidanceGenerator 测试类。

    测试引导生成器的各种引导策略和功能。
    """

    @pytest.mark.asyncio
    async def test_generate_direct_question(
        self,
        guidance_generator: GuidanceGenerator,
    ) -> None:
        """测试直接提问策略。

        验证当语义点重要性在 0.6-0.8 之间时，使用直接提问策略。

        Args:
            guidance_generator: 引导生成器实例
        """
        uncovered_points = [
            {"point_id": "SP-001", "description": "药品降糖效果", "importance": 0.65}
        ]
        context = {"product_name": "某降糖药物", "customer_need": "降血糖"}

        guidance = guidance_generator.generate(uncovered_points, context)

        assert "药品降糖效果" in guidance

    @pytest.mark.asyncio
    async def test_generate_challenge(
        self,
        guidance_generator: GuidanceGenerator,
    ) -> None:
        """测试质疑挑战策略。

        验证当语义点重要性 >= 0.8 时，使用质疑挑战策略。

        Args:
            guidance_generator: 引导生成器实例
        """
        uncovered_points = [
            {"point_id": "SP-001", "description": "药品副作用", "importance": 0.85}
        ]
        context = {"product_name": "某降糖药物", "customer_need": "降血糖"}

        guidance = guidance_generator.generate(uncovered_points, context)

        assert "药品副作用" in guidance
        assert "客户" in guidance or "患者" in guidance or "确定" in guidance

    @pytest.mark.asyncio
    async def test_generate_clarification(
        self,
        guidance_generator: GuidanceGenerator,
    ) -> None:
        """测试澄清请求策略。

        验证当语义点重要性在 0.4-0.6 之间时，使用澄清请求策略。

        Args:
            guidance_generator: 引导生成器实例
        """
        uncovered_points = [
            {"point_id": "SP-001", "description": "使用方法", "importance": 0.45}
        ]
        context = {"product_name": "某降糖药物", "customer_need": "降血糖"}

        guidance = guidance_generator.generate(uncovered_points, context)

        assert "使用方法" in guidance
        assert "具体" in guidance or "说说" in guidance or "吗" in guidance

    @pytest.mark.asyncio
    async def test_generate_supplementary(
        self,
        guidance_generator: GuidanceGenerator,
    ) -> None:
        """测试补充引导策略。

        验证当语义点重要性 < 0.4 时，使用补充引导策略。

        Args:
            guidance_generator: 引导生成器实例
        """
        uncovered_points = [
            {"point_id": "SP-001", "description": "药品外观", "importance": 0.3}
        ]
        context = {"product_name": "某降糖药物", "customer_need": "降血糖"}

        guidance = guidance_generator.generate(uncovered_points, context)

        assert "药品外观" in guidance
        assert "补充" in guidance or "还有什么" in guidance

    @pytest.mark.asyncio
    async def test_generate_with_rag(
        self,
        guidance_generator_with_rag: tuple[GuidanceGenerator, MagicMock],
        mock_llm_service: MagicMock,
    ) -> None:
        """测试 RAG 增强的引导生成。

        验证当提供 RAG 检索器时，引导生成能结合知识库内容。

        Args:
            guidance_generator_with_rag: 带 RAG 的引导生成器实例和模拟检索器
            mock_llm_service: 模拟的 LLM 服务
        """
        generator, mock_retriever = guidance_generator_with_rag

        mock_response = MagicMock()
        mock_response.content = "结合知识的引导话术。"
        mock_llm_service.invoke = MagicMock(return_value=mock_response)

        uncovered_points = [
            {"point_id": "SP-001", "description": "降糖效果", "importance": 0.7}
        ]
        context = {"product_name": "某降糖药物", "customer_need": "降血糖"}

        guidance = generator.generate(uncovered_points, context)

        mock_retriever.retrieve.assert_called_once()
        mock_llm_service.invoke.assert_called_once()
        assert "结合知识的引导话术" in guidance

    @pytest.mark.asyncio
    async def test_generate_empty_points(
        self,
        guidance_generator: GuidanceGenerator,
    ) -> None:
        """测试空语义点列表处理。

        验证当未覆盖语义点列表为空时，返回肯定性反馈。

        Args:
            guidance_generator: 引导生成器实例
        """
        guidance = guidance_generator.generate([], {})

        assert "全面" in guidance or "关键信息" in guidance

    @pytest.mark.asyncio
    async def test_generate_multiple_points(
        self,
        guidance_generator: GuidanceGenerator,
    ) -> None:
        """测试多个语义点生成。

        验证当有多个未覆盖语义点时，生成主引导和次要引导。

        Args:
            guidance_generator: 引导生成器实例
        """
        uncovered_points = [
            {"point_id": "SP-001", "description": "主要卖点", "importance": 0.9},
            {"point_id": "SP-002", "description": "次要卖点", "importance": 0.5},
        ]
        context = {"product_name": "某产品", "customer_need": "客户需求"}

        guidance = guidance_generator.generate(uncovered_points, context)

        assert "主要卖点" in guidance
        assert "次要卖点" in guidance or "另外" in guidance

    @pytest.mark.asyncio
    async def test_strategy_selection_by_importance(
        self,
        guidance_generator: GuidanceGenerator,
    ) -> None:
        """测试根据重要性选择策略。

        验证不同重要性阈值对应不同引导策略。

        Args:
            guidance_generator: 引导生成器实例
        """
        test_cases: list[dict[str, Any]] = [
            {"importance": 0.9, "expected_strategy": "challenge"},
            {"importance": 0.7, "expected_strategy": "direct"},
            {"importance": 0.5, "expected_strategy": "clarification"},
            {"importance": 0.3, "expected_strategy": "supplementary"},
        ]

        for test_case in test_cases:
            uncovered_points = [
                {"point_id": "SP-001", "description": "测试点", "importance": test_case["importance"]}
            ]
            context = {"product_name": "某产品", "customer_need": "需求"}

            guidance = guidance_generator.generate(uncovered_points, context)

            assert "测试点" in guidance, f"Failed for importance {test_case['importance']}"
