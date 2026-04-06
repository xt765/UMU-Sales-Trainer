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
        overall_score = self._calculate_overall_score(coverage_rate, expression, coverage_status)

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
        try:
            prompt = (
                "你是一位严格的销售话术质量评估专家。请对以下销售话术从三个维度严格评分，"
                "不要宽容，要客观反映真实水平。\n\n"
                f"【待评估话术】\n{message}\n\n"
                "【评分标准（严格执行）】\n"
                "清晰度（Clarity）：\n"
                "  1-3分：语句不通顺、无标点或标点混乱、逻辑跳跃、难以理解\n"
                "  4-5分：基本通顺但有语病、标点缺失、结构松散\n"
                "  6-7分：通顺完整、有适当标点、结构合理但不够精炼\n"
                "  8-9分：表达流畅、层次分明、用词准确、易于理解\n"
                "  10分：精炼有力、环环相扣、一气呵成、极具感染力\n\n"
                "专业性（Professionalism）：\n"
                "  1-3分：无专业术语、口语化严重、无数据支撑\n"
                "  4-5分：有少量术语但使用不当、数据模糊\n"
                "  6-7分：术语基本准确、有具体数据但引用不规范\n"
                "  8-9分：专业术语准确、数据引用规范、体现行业认知\n"
                "  10分：专家级表达、数据详实、深度专业洞察\n\n"
                "说服力（Persuasiveness）：\n"
                "  1-3分：无论证逻辑、纯陈述性表达、无行动引导\n"
                "  4-5分：有简单对比但缺乏力度、无明确利益点\n"
                "  6-7分：有一定论证、有数据支撑但缺乏情感共鸣\n"
                "  8-9分：论证充分、数据+对比+痛点结合、有感染力\n"
                "  10分：无可辩服的逻辑链、直击痛点、强烈行动号召\n\n"
                "请严格按照以上标准评分。回复格式（仅数字和逗号）：\n"
                "清晰度:X, 专业性:Y, 说服力:Z"
            )

            response = self.llm_service.invoke([HumanMessage(content=prompt)])
            return self._parse_expression_response(response.content)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                "LLM expression analysis failed, using rule-based fallback: %s", e
            )
            return self._rule_based_expression_analysis(message)

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

    def _rule_based_expression_analysis(self, message: str) -> ExpressionAnalysis:
        """基于规则的表达能力分析（LLM不可用时的降级方案）。

        通过文本特征分析话术质量：
        - 清晰度：句子长度、标点使用、结构完整性
        - 专业性：专业术语密度、数据引用
        - 说服力：数据支撑、对比论证、行动号召

        Args:
            message: 销售消息

        Returns:
            表达能力分析结果
        """
        import re

        clarity = 5
        professionalism = 5
        persuasiveness = 5

        # 清晰度评估 — 多维度严格打分
        sentences = re.split(r"[。！？!?.]", message)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

        if valid_sentences:
            avg_len = sum(len(s) for s in valid_sentences) / len(valid_sentences)

            if 15 <= avg_len <= 50:
                clarity += 1
            elif 10 <= avg_len < 15 or 50 < avg_len <= 70:
                pass
            elif avg_len < 10 or avg_len > 70:
                clarity -= 1

            has_comma = "，" in message or "," in message
            has_pause = "、" in message
            if has_comma and has_pause:
                clarity += 1
            elif has_comma or has_pause:
                clarity += 0

            long_sentences = sum(1 for s in valid_sentences if len(s) > 80)
            short_sentences = sum(1 for s in valid_sentences if len(s) < 8)
            ratio_long = long_sentences / len(valid_sentences)
            ratio_short = short_sentences / len(valid_sentences)

            if ratio_long > 0.4:
                clarity -= 1
            if ratio_short > 0.5:
                clarity -= 1

            words = re.findall(r"[\u4e00-\u9fff]+", message)
            unique_words = len(set(words))
            total_words = len(words)
            if total_words > 20 and unique_words / total_words < 0.6:
                clarity -= 1

        else:
            clarity = 2

        clarity = max(1, min(10, clarity))

        # 专业性评估
        professional_terms = [
            "临床",
            "数据",
            "研究",
            "试验",
            "HbA1c",
            "%",
            "患者",
            "治疗",
            "疗效",
            "安全性",
            "副作用",
            "依从性",
            "发生率",
            "mg",
            "ml",
            "剂量",
            "方案",
        ]
        term_count = sum(1 for term in professional_terms if term in message)
        professionalism = min(10, 5 + term_count)

        # 说服力评估
        persuasion_signals = [
            (r"\d+[.]?\d*%", "数据百分比"),
            (r"相比|远低于|优于|超过|降低|提高", "对比论证"),
            (r"可以考虑|建议|推荐|作为.*方案", "行动号召"),
            (r"更重要的是|而且|此外|同时", "递进论述"),
        ]
        signal_score = 0
        for pattern, _ in persuasion_signals:
            if re.search(pattern, message):
                signal_score += 1
        persuasiveness = min(10, 4 + signal_score)

        return ExpressionAnalysis(
            clarity=clarity,
            professionalism=professionalism,
            persuasiveness=persuasiveness,
        )

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
            (expression.clarity + expression.professionalism + expression.persuasiveness) / 30 * 50
        )
        return round(coverage_score + expression_score, 2)
