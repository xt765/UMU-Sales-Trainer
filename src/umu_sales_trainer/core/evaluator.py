"""语义点评估器模块。

实现三层检测机制评估语义点覆盖情况：关键词检测、Embedding相似度检测和LLM检测。
"""

from typing import List

from langchain_core.messages import HumanMessage

from umu_sales_trainer.models.evaluation import (
    EvaluationResult,
    ExpressionAnalysis,
)
from umu_sales_trainer.models.semantic import SemanticPoint
from umu_sales_trainer.services.embedding import EmbeddingService
from umu_sales_trainer.services.llm import LLMService


class SemanticEvaluator:
    """语义点评估器。

    使用三层检测机制评估销售对话中语义点的覆盖情况：
    1. 关键词检测：快速筛选，检查消息中是否包含预定义关键词
    2. Embedding相似度：语义匹配，计算消息与语义点描述的向量相似度
    3. LLM判断：深度理解，使用大语言模型判断语义是否被正确表达

    Attributes:
        embedding_service: 向量嵌入服务，用于计算文本相似度
        llm_service: LLM服务，用于深度语义理解判断
        _keyword_weight: 关键词检测在最终判定中的权重
        _embedding_weight: Embedding相似度在最终判定中的权重
        _llm_weight: LLM判断在最终判定中的权重
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ) -> None:
        """初始化语义点评估器。

        Args:
            embedding_service: 向量嵌入服务实例
            llm_service: LLM服务实例
        """
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self._keyword_weight = 0.2
        self._embedding_weight = 0.3
        self._llm_weight = 0.5

    def evaluate(
        self,
        sales_message: str,
        semantic_points: List[SemanticPoint],
        context: dict,
    ) -> EvaluationResult:
        """评估销售消息对语义点的覆盖情况。

        Args:
            sales_message: 销售人员发送的消息
            semantic_points: 需要评估的语义点列表
            context: 评估上下文，包含session_id等信息

        Returns:
            评估结果，包含覆盖状态和表达能力分析
        """
        session_id = context.get("session_id", "unknown")
        coverage_status: dict[str, str] = {}

        for point in semantic_points:
            result = self._evaluate_single_point(sales_message, point)
            coverage_status[point.point_id] = result

        coverage_rate = self._calculate_coverage_rate(coverage_status)
        expression = self._analyze_expression(sales_message, context)
        overall_score = self._calculate_overall_score(
            coverage_rate, expression, coverage_status
        )

        return EvaluationResult(
            session_id=session_id,
            coverage_status=coverage_status,
            expression_analysis=expression,
            coverage_rate=coverage_rate,
            overall_score=overall_score,
        )

    def _evaluate_single_point(
        self,
        message: str,
        point: SemanticPoint,
    ) -> str:
        """评估单个语义点的覆盖情况。

        Args:
            message: 销售消息
            point: 语义点

        Returns:
            "covered" 或 "not_covered"
        """
        keyword_score = self._keyword_detection(message, point)
        embedding_score = self._embedding_similarity(message, point)
        llm_score = self._llm_judgment(message, point)

        final_score = (
            keyword_score * self._keyword_weight
            + embedding_score * self._embedding_weight
            + llm_score * self._llm_weight
        )

        return "covered" if final_score >= 0.5 else "not_covered"

    def _keyword_detection(self, message: str, point: SemanticPoint) -> float:
        """第一层检测：关键词匹配。

        Args:
            message: 销售消息
            point: 语义点

        Returns:
            关键词匹配得分，0.0-1.0
        """
        if not point.keywords:
            return 0.5

        message_lower = message.lower()
        matched = sum(1 for kw in point.keywords if kw.lower() in message_lower)
        return matched / len(point.keywords)

    def _embedding_similarity(self, message: str, point: SemanticPoint) -> float:
        """第二层检测：Embedding相似度。

        Args:
            message: 销售消息
            point: 语义点

        Returns:
            语义相似度得分，0.0-1.0
        """
        threshold = getattr(point, "threshold", 0.7)
        query_emb = self.embedding_service.encode_query(message)
        point_emb = self.embedding_service.encode_query(point.description)

        similarity = self._cosine_similarity(query_emb, point_emb)
        return 1.0 if similarity >= threshold else similarity / threshold

    def _llm_judgment(self, message: str, point: SemanticPoint) -> float:
        """第三层检测：LLM判断。

        Args:
            message: 销售消息
            point: 语义点

        Returns:
            LLM判断得分，0.0-1.0
        """
        prompt = (
            f"判断以下销售话术是否覆盖了指定的语义点。\n"
            f"语义点描述：{point.description}\n"
            f"销售话术：{message}\n"
            f"如果话术充分覆盖了该语义点，返回1；如果只是略微提及或未覆盖，返回0。"
        )

        response = self.llm_service.invoke([HumanMessage(content=prompt)])
        content = response.content.lower().strip()

        if content.startswith("1"):
            return 1.0
        elif content.startswith("0"):
            return 0.0
        return 0.5

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度。

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度值，-1.0到1.0
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return dot_product

    def _calculate_coverage_rate(self, coverage_status: dict[str, str]) -> float:
        """计算语义点覆盖率。

        Args:
            coverage_status: 覆盖状态字典

        Returns:
            覆盖率，0.0-1.0
        """
        if not coverage_status:
            return 0.0
        covered = sum(1 for v in coverage_status.values() if v == "covered")
        return covered / len(coverage_status)

    def _analyze_expression(
        self,
        message: str,
        context: dict,
    ) -> ExpressionAnalysis:
        """分析销售人员的表达能力。

        Args:
            message: 销售消息
            context: 评估上下文

        Returns:
            表达能力分析结果
        """
        prompt = (
            f"分析以下销售话术的表达质量，从清晰度、专业性、说服力三个维度评分。\n"
            f"话术：{message}\n"
            f"每个维度1-10分，回复格式：清晰度:X, 专业性:Y, 说服力:Z"
        )

        response = self.llm_service.invoke([HumanMessage(content=prompt)])
        return self._parse_expression_response(response.content)

    def _parse_expression_response(self, content: str) -> ExpressionAnalysis:
        """解析LLM表达分析响应。

        Args:
            content: LLM响应内容

        Returns:
            表达能力分析结果
        """
        analysis = ExpressionAnalysis(clarity=5, professionalism=5, persuasiveness=5)
        content_lower = content.lower()

        for part in content_lower.split(","):
            if "清晰度" in part or "clarity" in part:
                try:
                    score = int("".join(filter(str.isdigit, part)))
                    analysis.clarity = min(max(score, 1), 10)
                except ValueError:
                    pass
            elif "专业性" in part or "professionalism" in part:
                try:
                    score = int("".join(filter(str.isdigit, part)))
                    analysis.professionalism = min(max(score, 1), 10)
                except ValueError:
                    pass
            elif "说服力" in part or "persuasiveness" in part:
                try:
                    score = int("".join(filter(str.isdigit, part)))
                    analysis.persuasiveness = min(max(score, 1), 10)
                except ValueError:
                    pass

        return analysis

    def _calculate_overall_score(
        self,
        coverage_rate: float,
        expression: ExpressionAnalysis,
        coverage_status: dict[str, str],
    ) -> float:
        """计算综合评分。

        Args:
            coverage_rate: 语义点覆盖率
            expression: 表达能力分析
            coverage_status: 覆盖状态

        Returns:
            综合评分，0.0-100.0
        """
        coverage_score = coverage_rate * 50
        expression_score = (
            expression.clarity + expression.professionalism + expression.persuasiveness
        ) / 30 * 50
        return round(coverage_score + expression_score, 2)
