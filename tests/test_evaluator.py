"""SemanticEvaluator 测试模块。

测试三层检测机制：关键词检测、Embedding相似度检测和LLM检测。
"""

from unittest.mock import MagicMock

import pytest

from umu_sales_trainer.core.evaluator import SemanticEvaluator
from umu_sales_trainer.models.evaluation import EvaluationResult
from umu_sales_trainer.models.semantic import SemanticPoint


class TestKeywordDetection:
    """关键词检测（第一层）测试类。"""

    def test_evaluate_keyword_detection(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_embedding_service: MagicMock,
        mock_llm_service: MagicMock,
    ) -> None:
        """测试关键词检测功能。

        验证当消息包含语义点的关键词时，关键词检测返回正确的匹配分数。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_embedding_service: Mock 向量嵌入服务
            mock_llm_service: Mock LLM 服务
        """
        mock_embedding_service.encode_query.return_value = [0.1] * 384
        mock_llm_service.invoke.return_value = MagicMock(content="1")

        point = SemanticPoint(
            point_id="SP-001",
            description="介绍产品的降糖效果",
            keywords=["降糖", "血糖", "效果"],
        )

        message = "这个产品降糖效果很好，能有效控制血糖"

        score = semantic_evaluator._keyword_detection(message, point)

        assert score == 1.0, "包含所有关键词时应返回1.0"

    def test_keyword_detection_partial_match(
        self,
        semantic_evaluator: SemanticEvaluator,
    ) -> None:
        """测试关键词部分匹配。

        验证当消息只包含部分关键词时，返回按比例计算的分数。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
        """
        point = SemanticPoint(
            point_id="SP-002",
            description="说明药品的安全性",
            keywords=["安全", "副作用", "耐受"],
        )

        message = "这个药很安全"

        score = semantic_evaluator._keyword_detection(message, point)

        assert score == pytest.approx(1 / 3, rel=0.01), "只包含1个关键词时应返回1/3"

    def test_keyword_detection_no_match(
        self,
        semantic_evaluator: SemanticEvaluator,
    ) -> None:
        """测试关键词无匹配。

        验证当消息不包含任何关键词时，返回0分。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
        """
        point = SemanticPoint(
            point_id="SP-003",
            description="介绍产品价格",
            keywords=["价格", "优惠", "折扣"],
        )

        message = "这个产品质量很好"

        score = semantic_evaluator._keyword_detection(message, point)

        assert score == 0.0, "不包含关键词时应返回0.0"

    def test_keyword_detection_empty_keywords(
        self,
        semantic_evaluator: SemanticEvaluator,
    ) -> None:
        """测试空关键词列表。

        验证当语义点没有定义关键词时，返回默认值0.5。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
        """
        point = SemanticPoint(
            point_id="SP-004",
            description="介绍产品",
            keywords=[],
        )

        message = "这是一个产品介绍"

        score = semantic_evaluator._keyword_detection(message, point)

        assert score == 0.5, "空关键词列表时应返回默认值0.5"


class TestEmbeddingSimilarity:
    """Embedding 相似度（第二层）测试类。"""

    def test_evaluate_embedding_similarity(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_embedding_service: MagicMock,
    ) -> None:
        """测试 Embedding 相似度检测功能。

        验证向量嵌入服务正确计算消息与语义点描述的语义相似度。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_embedding_service: Mock 向量嵌入服务
        """
        mock_embedding_service.encode_query.side_effect = [
            [0.1] * 384,
            [0.1] * 384,
        ]

        point = SemanticPoint(
            point_id="SP-001",
            description="介绍产品的降糖效果",
            keywords=["降糖"],
        )

        message = "这个产品降糖效果很好"

        score = semantic_evaluator._embedding_similarity(message, point)

        assert mock_embedding_service.encode_query.call_count == 2
        assert score >= 0, "相似度得分应大于等于0"

    def test_embedding_similarity_high_similarity(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_embedding_service: MagicMock,
    ) -> None:
        """测试高相似度情况。

        验证当相似度超过阈值时，返回1.0。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_embedding_service: Mock 向量嵌入服务
        """
        mock_embedding_service.encode_query.side_effect = [
            [1.0] * 384,
            [1.0] * 384,
        ]

        point = SemanticPoint(
            point_id="SP-001",
            description="介绍产品的降糖效果",
            keywords=["降糖"],
        )

        message = "产品降糖效果"

        score = semantic_evaluator._embedding_similarity(message, point)

        assert score == 1.0, "高相似度时应返回1.0"

    def test_embedding_similarity_below_threshold(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_embedding_service: MagicMock,
    ) -> None:
        """测试低于阈值情况。

        验证当相似度低于阈值时，返回归一化后的分数。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_embedding_service: Mock 向量嵌入服务
        """
        mock_embedding_service.encode_query.side_effect = [
            [0.1] * 384,
            [0.9] * 384,
        ]

        point = SemanticPoint(
            point_id="SP-001",
            description="介绍产品的降糖效果",
            keywords=["降糖"],
        )

        message = "完全不相关的内容"

        score = semantic_evaluator._embedding_similarity(message, point)

        assert 0 <= score <= 1.0, "应返回0-1之间的分数"


class TestLLMJudgment:
    """LLM 判断（第三层）测试类。"""

    def test_evaluate_llm_judgment(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_llm_service: MagicMock,
    ) -> None:
        """测试 LLM 判断功能。

        验证 LLM 服务正确判断消息是否覆盖语义点。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_llm_service: Mock LLM 服务
        """
        mock_llm_service.invoke.return_value = MagicMock(content="1")

        point = SemanticPoint(
            point_id="SP-001",
            description="介绍产品的降糖效果",
            keywords=["降糖"],
        )

        message = "这个产品降糖效果很好"

        score = semantic_evaluator._llm_judgment(message, point)

        assert score == 1.0, "LLM判断为覆盖时应返回1.0"
        mock_llm_service.invoke.assert_called_once()

    def test_llm_judgment_not_covered(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_llm_service: MagicMock,
    ) -> None:
        """测试 LLM 判断为未覆盖。

        验证当 LLM 返回 0 时，表示消息未覆盖语义点。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_llm_service: Mock LLM 服务
        """
        mock_llm_service.invoke.return_value = MagicMock(content="0")

        point = SemanticPoint(
            point_id="SP-001",
            description="介绍产品的降糖效果",
            keywords=["降糖"],
        )

        message = "天气真好"

        score = semantic_evaluator._llm_judgment(message, point)

        assert score == 0.0, "LLM判断为未覆盖时应返回0.0"

    def test_llm_judgment_ambiguous_response(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_llm_service: MagicMock,
    ) -> None:
        """测试 LLM 返回不明确响应。

        验证当 LLM 返回非0/1响应时，返回默认值0.5。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_llm_service: Mock LLM 服务
        """
        mock_llm_service.invoke.return_value = MagicMock(content="maybe")

        point = SemanticPoint(
            point_id="SP-001",
            description="介绍产品的降糖效果",
            keywords=["降糖"],
        )

        message = "这个产品"

        score = semantic_evaluator._llm_judgment(message, point)

        assert score == 0.5, "不明确响应时应返回默认值0.5"


class TestCombinedResults:
    """三层综合结果测试类。"""

    def test_evaluate_combined_results(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_embedding_service: MagicMock,
        mock_llm_service: MagicMock,
        sample_semantic_points: list[SemanticPoint],
        sample_context: dict,
    ) -> None:
        """测试三层综合评估结果。

        验证 evaluate 方法正确综合三层检测结果并返回完整评估结果。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_embedding_service: Mock 向量嵌入服务
            mock_llm_service: Mock LLM 服务
            sample_semantic_points: 示例语义点列表
            sample_context: 示例上下文
        """
        mock_embedding_service.encode_query.return_value = [0.1] * 384
        mock_llm_service.invoke.return_value = MagicMock(content="1")

        message = "这个产品降糖效果好，安全可靠"

        result = semantic_evaluator.evaluate(message, sample_semantic_points, sample_context)

        assert isinstance(result, EvaluationResult)
        assert result.session_id == "test-session-001"
        assert result.coverage_rate >= 0.0
        assert 0 <= result.overall_score <= 100.0

    def test_combined_all_covered(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_embedding_service: MagicMock,
        mock_llm_service: MagicMock,
    ) -> None:
        """测试所有语义点都被覆盖的情况。

        验证当所有语义点都被覆盖时，覆盖率为1.0。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_embedding_service: Mock 向量嵌入服务
            mock_llm_service: Mock LLM 服务
        """
        mock_embedding_service.encode_query.return_value = [1.0] * 384
        mock_llm_service.invoke.return_value = MagicMock(content="1")

        point = SemanticPoint(
            point_id="SP-001",
            description="介绍产品的降糖效果",
            keywords=["降糖", "血糖", "效果"],
        )

        message = "这个产品降糖效果很好，能有效控制血糖"

        result = semantic_evaluator.evaluate(message, [point], {"session_id": "test"})

        assert result.coverage_status["SP-001"] == "covered"
        assert result.coverage_rate == 1.0

    def test_combined_none_covered(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_embedding_service: MagicMock,
        mock_llm_service: MagicMock,
    ) -> None:
        """测试所有语义点都未被覆盖的情况。

        验证当所有语义点都未被覆盖时，覆盖率为0.0。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_embedding_service: Mock 向量嵌入服务
            mock_llm_service: Mock LLM 服务
        """
        mock_embedding_service.encode_query.return_value = [0.1] * 384
        mock_llm_service.invoke.return_value = MagicMock(content="0")

        point = SemanticPoint(
            point_id="SP-001",
            description="介绍产品的降糖效果",
            keywords=["降糖"],
        )

        message = "今天天气不错"

        result = semantic_evaluator.evaluate(message, [point], {"session_id": "test"})

        assert result.coverage_status["SP-001"] == "not_covered"
        assert result.coverage_rate == 0.0


class TestExpressionQuality:
    """表达能力评估测试类。"""

    def test_evaluate_expression_quality(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_llm_service: MagicMock,
    ) -> None:
        """测试表达能力评估功能。

        验证 evaluate 方法正确分析销售人员的表达能力。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_llm_service: Mock LLM 服务
        """
        mock_llm_service.invoke.return_value = MagicMock(
            content="清晰度:8, 专业性:7, 说服力:9"
        )

        message = "这个产品降糖效果显著，是您的最佳选择"

        expression = semantic_evaluator._analyze_expression(message, {})

        assert expression.clarity == 8
        assert expression.professionalism == 7
        assert expression.persuasiveness == 9

    def test_expression_quality_default_values(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_llm_service: MagicMock,
    ) -> None:
        """测试表达能力默认值。

        验证当 LLM 返回无法解析的响应时，使用默认值。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_llm_service: Mock LLM 服务
        """
        mock_llm_service.invoke.return_value = MagicMock(content="invalid response")

        message = "产品介绍"

        expression = semantic_evaluator._analyze_expression(message, {})

        assert expression.clarity == 5
        assert expression.professionalism == 5
        assert expression.persuasiveness == 5

    def test_expression_quality_with_english_labels(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_llm_service: MagicMock,
    ) -> None:
        """测试英文标签解析。

        验证表达能力分析能正确解析英文标签。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_llm_service: Mock LLM 服务
        """
        mock_llm_service.invoke.return_value = MagicMock(
            content="clarity:8, professionalism:9, persuasiveness:7"
        )

        message = "This product has excellent quality"

        expression = semantic_evaluator._analyze_expression(message, {})

        assert expression.clarity == 8
        assert expression.professionalism == 9
        assert expression.persuasiveness == 7

    def test_expression_quality_score_bounded(
        self,
        semantic_evaluator: SemanticEvaluator,
        mock_llm_service: MagicMock,
    ) -> None:
        """测试表达能力分数边界。

        验证表达能力分数被限制在1-10范围内。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            mock_llm_service: Mock LLM 服务
        """
        mock_llm_service.invoke.return_value = MagicMock(
            content="清晰度:15, 专业性:0, 说服力:50"
        )

        message = "测试消息"

        expression = semantic_evaluator._analyze_expression(message, {})

        assert expression.clarity == 10, "超过10的值应被限制为10"
        assert expression.professionalism == 1, "小于1的值应被限制为1"
        assert expression.persuasiveness == 10, "超过10的值应被限制为10"


class TestCoverageRate:
    """覆盖率计算测试类。"""

    def test_calculate_coverage_rate_empty(self, semantic_evaluator: SemanticEvaluator) -> None:
        """测试空覆盖状态的覆盖率。

        验证当 coverage_status 为空时，返回 0.0。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
        """
        rate = semantic_evaluator._calculate_coverage_rate({})

        assert rate == 0.0, "空覆盖状态时应返回0.0"

    def test_calculate_coverage_rate_partial(
        self,
        semantic_evaluator: SemanticEvaluator,
    ) -> None:
        """测试部分覆盖率计算。

        验证当只有部分语义点被覆盖时，返回正确的覆盖率。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
        """
        coverage_status = {
            "SP-001": "covered",
            "SP-002": "not_covered",
            "SP-003": "covered",
        }

        rate = semantic_evaluator._calculate_coverage_rate(coverage_status)

        assert rate == pytest.approx(2 / 3, rel=0.01), "2/3覆盖时应返回约0.667"


class TestCosineSimilarity:
    """余弦相似度计算测试类。"""

    def test_cosine_similarity_identical(self, semantic_evaluator: SemanticEvaluator) -> None:
        """测试相同向量的相似度。

        验证相同向量的余弦相似度为1.0。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
        """
        vec = [0.5, 0.5, 0.5, 0.5]

        similarity = semantic_evaluator._cosine_similarity(vec, vec)

        assert similarity == 1.0, "相同向量的相似度应为1.0"

    def test_cosine_similarity_orthogonal(self, semantic_evaluator: SemanticEvaluator) -> None:
        """测试正交向量的相似度。

        验证正交向量的余弦相似度为0.0。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
        """
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        similarity = semantic_evaluator._cosine_similarity(vec1, vec2)

        assert similarity == 0.0, "正交向量的相似度应为0.0"

    def test_cosine_similarity_opposite(self, semantic_evaluator: SemanticEvaluator) -> None:
        """测试相反向量的相似度。

        验证相反向量的余弦相似度为-1.0。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
        """
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]

        similarity = semantic_evaluator._cosine_similarity(vec1, vec2)

        assert similarity == -1.0, "相反向量的相似度应为-1.0"


class TestOverallScore:
    """综合评分测试类。"""

    def test_calculate_overall_score_full_coverage(
        self,
        semantic_evaluator: SemanticEvaluator,
        expression_analysis,
    ) -> None:
        """测试全覆盖时的综合评分。

        验证全覆盖且高表达能力时，综合评分接近100。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            expression_analysis: 预设的表达分析结果
        """
        coverage_status = {"SP-001": "covered"}

        score = semantic_evaluator._calculate_overall_score(
            coverage_rate=1.0,
            expression=expression_analysis,
            coverage_status=coverage_status,
        )

        assert 80 <= score <= 100, "全覆盖高表达时应得到高分"

    def test_calculate_overall_score_no_coverage(
        self,
        semantic_evaluator: SemanticEvaluator,
        expression_analysis,
    ) -> None:
        """测试无覆盖时的综合评分。

        验证无覆盖但有表达能力时，分数取决于表达能力部分。

        Args:
            semantic_evaluator: SemanticEvaluator 实例
            expression_analysis: 预设的表达分析结果
        """
        coverage_status = {"SP-001": "not_covered"}

        score = semantic_evaluator._calculate_overall_score(
            coverage_rate=0.0,
            expression=expression_analysis,
            coverage_status=coverage_status,
        )

        expected_expression_score = (8 + 7 + 9) / 30 * 50
        expected_total = expected_expression_score
        assert abs(score - expected_total) < 0.1, "无覆盖时分数应等于表达能力分数"
