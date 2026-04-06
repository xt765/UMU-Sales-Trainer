"""语义评估模块。

实现两个独立的 Agent：
- SemanticCoverageExpert（语义覆盖专家）：三层语义点覆盖检测
- ExpressionCoach（表达教练）：表达能力评估 + 改进建议

替代原有的单一 SemanticEvaluator 类，遵循单一职责原则。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import HumanMessage

from umu_sales_trainer.models.evaluation import (
    EvaluationResult,
    ExpressionAnalysis,
)
from umu_sales_trainer.models.semantic import SemanticPoint
from umu_sales_trainer.services.embedding import EmbeddingService
from umu_sales_trainer.services.llm import LLMService

logger = logging.getLogger(__name__)


@dataclass
class Suggestion:
    """表达改进建议。

    Attributes:
        dimension: 评分维度名称（clarity / professionalism / persuasiveness）
        current_score: 当前分数（1-10）
        advice: 具体改进建议
        example: 参考话术范例
    """

    dimension: str = ""
    current_score: int = 0
    advice: str = ""
    example: str = ""


@dataclass
class CoverageResult:
    """语义覆盖检测结果。

    Attributes:
        coverage_status: 各语义点覆盖状态字典
        coverage_rate: 覆盖率（0-1）
        uncovered_points: 未覆盖的语义点 ID 列表
    """

    coverage_status: dict[str, str] = field(default_factory=dict)
    coverage_rate: float = 0.0
    uncovered_points: list[str] = field(default_factory=list)


@dataclass
class ExpressionResult:
    """表达能力评估结果。

    Attributes:
        analysis: 表达能力三维度分析
        suggestions: 改进建议列表
    """

    analysis: ExpressionAnalysis = field(default_factory=lambda: ExpressionAnalysis())
    suggestions: list[Suggestion] = field(default_factory=list)


class SemanticCoverageExpert:
    """语义覆盖专家 Agent。

    使用三层检测机制评估销售对话中语义点的覆盖情况：
    1. 关键词检测（权重 0.2）：快速筛选预定义关键词
    2. Embedding 相似度（权重 0.3）：语义匹配向量计算
    3. LLM 判断（权重 0.5）：大语言模型深度语义理解

    这是 Agentic RAG 工作流的第二个分析节点，
    仅负责语义点覆盖判定，不涉及表达质量评估。

    Attributes:
        embedding_service: 向量嵌入服务，用于文本相似度计算
        llm_service: LLM 服务，用于深度语义判断
        _keyword_weight: 关键词层权重
        _embedding_weight: Embedding 层权重
        _llm_weight: LLM 层权重
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ) -> None:
        """初始化语义覆盖专家。

        Args:
            embedding_service: 向量嵌入服务实例
            llm_service: LLM 服务实例
        """
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self._keyword_weight = 0.2
        self._embedding_weight = 0.3
        self._llm_weight = 0.5

    def evaluate_coverage(
        self,
        sales_message: str,
        semantic_points: List[SemanticPoint],
        context: Optional[dict] = None,
    ) -> CoverageResult:
        """评估销售消息对语义点的覆盖情况。

        对每个语义点执行三层检测并加权综合判断。

        Args:
            sales_message: 销售人员发送的消息
            semantic_points: 需要评估的语义点列表
            context: 评估上下文（可选）

        Returns:
            CoverageResult 覆盖检测结果
        """
        context = context or {}
        session_id = context.get("session_id", "unknown")
        coverage_status: dict[str, str] = {}

        for point in semantic_points:
            result = self._evaluate_single_point(sales_message, point)
            coverage_status[point.point_id] = result

        uncovered = [
            pid for pid, status in coverage_status.items()
            if status != "covered"
        ]
        coverage_rate = self._calculate_coverage_rate(coverage_status)

        logger.info(
            "SemanticCoverageExpert: rate=%.2f, covered=%d/%d",
            coverage_rate,
            len(semantic_points) - len(uncovered),
            len(semantic_points),
        )

        return CoverageResult(
            coverage_status=coverage_status,
            coverage_rate=coverage_rate,
            uncovered_points=uncovered,
        )

    def _evaluate_single_point(
        self, message: str, point: SemanticPoint
    ) -> str:
        """评估单个语义点的覆盖情况。

        通过三层检测加权综合判断是否覆盖。

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

    @staticmethod
    def _keyword_detection(message: str, point: SemanticPoint) -> float:
        """第一层检测：关键词匹配。

        检查消息中是否包含语义点的预定义关键词。

        Args:
            message: 销售消息
            point: 语义点

        Returns:
            关键词匹配得分（0-1）
        """
        if not point.keywords:
            return 0.5

        message_lower = message.lower()
        matched = sum(1 for kw in point.keywords if kw.lower() in message_lower)
        return matched / len(point.keywords)

    def _embedding_similarity(self, message: str, point: SemanticPoint) -> float:
        """第二层检测：Embedding 相似度。

        计算消息与语义点描述之间的向量余弦相似度。

        Args:
            message: 销售消息
            point: 语义点

        Returns:
            归一化后的相似度得分（0-1）
        """
        threshold = getattr(point, "threshold", 0.7)
        query_emb = self.embedding_service.encode_query(message)
        point_emb = self.embedding_service.encode_query(point.description)

        similarity = self._cosine_similarity(query_emb, point_emb)
        return 1.0 if similarity >= threshold else similarity / threshold

    def _llm_judgment(self, message: str, point: SemanticPoint) -> float:
        """第三层检测：LLM 深度判断。

        使用大语言模型判断销售话术是否充分覆盖了指定语义点。

        Args:
            message: 销售话术
            point: 语义点

        Returns:
            LLM 判断得分（0 或 1，中间值 0.5 表示不确定）
        """
        prompt = (
            f"判断以下销售话术是否充分覆盖了指定的语义点。\n"
            f"语义点描述：{point.description}\n"
            f"销售话术：{message}\n"
            f"如果话术充分且具体地覆盖了该语义点，返回1；"
            f"如果只是略微提及或未覆盖，返回0。只回复数字。"
        )

        try:
            response = self.llm_service.invoke([HumanMessage(content=prompt)])
            content = response.content.lower().strip()

            if content.startswith("1"):
                return 1.0
            elif content.startswith("0"):
                return 0.0
            return 0.5
        except Exception as e:
            logger.warning("LLM judgment failed for %s: %s", point.point_id, e)
            return 0.5

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度。

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度值（-1 到 1）
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return dot_product

    @staticmethod
    def _calculate_coverage_rate(coverage_status: dict[str, str]) -> float:
        """计算语义点覆盖率。

        Args:
            coverage_status: 覆盖状态字典

        Returns:
            覆盖率（0-1）
        """
        if not coverage_status:
            return 0.0
        covered = sum(1 for v in coverage_status.values() if v == "covered")
        return covered / len(coverage_status)


class ExpressionCoach:
    """表达教练 Agent。

    专门负责评估销售人员的话术表达能力（清晰度/专业性/说服力），
    并针对低分维度生成具体的改进建议和参考话术范例。

    这是 Agentic RAG 工作流的第三个分析节点，
    与 SemanticCoverageExpert 独立运行，职责不重叠。

    Attributes:
        llm_service: LLM 服务实例，用于评分和建议生成
    """

    def __init__(self, llm_service: LLMService) -> None:
        """初始化表达教练。

        Args:
            llm_service: LLM 服务实例
        """
        self.llm_service = llm_service

    def evaluate(self, message: str, context: Optional[dict] = None) -> ExpressionResult:
        """评估销售人员的表达能力并生成改进建议。

        先通过 LLM 进行三维度打分和建议生成，
        失败时回退到基于规则的分析方案。

        Args:
            message: 销售消息
            context: 评估上下文（可选）

        Returns:
            ExpressionResult 包含分析和建议
        """
        context = context or {}

        try:
            analysis = self._llm_evaluate(message)
            suggestions = self._generate_suggestions(message, analysis)
        except Exception as e:
            logger.warning("LLM expression evaluation failed, using fallback: %s", e)
            analysis = self._rule_based_expression_analysis(message)
            suggestions = self._generate_suggestions_from_rules(analysis)

        return ExpressionResult(analysis=analysis, suggestions=suggestions)

    def _llm_evaluate(self, message: str) -> ExpressionAnalysis:
        """通过 LLM 评估表达能力三维度。

        使用严格的五级评分标准确保区分度。

        Args:
            message: 待评估的销售话术

        Returns:
            ExpressionAnalysis 三维度评分结果
        """
        prompt = (
            "你是一位严格的销售话术质量评估专家兼表达教练。请对以下销售话术从三个维度严格评分，"
            "不要宽容，要客观反映真实水平。\n\n"
            f"【待评估话术】\n{message}\n\n"
            "【评分标准（严格执行）】\n"
            "清晰度（Clarity）：\n"
            "  1-3分：语句不通顺、无标点或标点混乱、逻辑跳跃、难以理解\n"
            "  4-5分：基本通顺但有语病、标点缺失、结构松散\n"
            "  6-7分：通顺完整、有适当标点、结构合理但不够精炼\n"
            "  8-9分：表达流畅、层次分明、用词准确、易于理解\n"
            " 10分：精炼有力、环环相扣、一气呵成、极具感染力\n\n"
            "专业性（Professionalism）：\n"
            "  1-3分：无专业术语、口语化严重、无数据支撑\n"
            "  4-5分：有少量术语但使用不当、数据模糊\n"
            "  6-7分：术语基本准确、有具体数据但引用不规范\n"
            "  8-9分：专业术语准确、数据引用规范、体现行业认知\n"
            " 10分：专家级表达、数据详实、深度专业洞察\n\n"
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

    def _generate_suggestions(
        self, message: str, analysis: ExpressionAnalysis
    ) -> list[Suggestion]:
        """基于评分结果生成改进建议。

        对低于 7 分的维度生成具体改进建议和参考话术。

        Args:
            message: 原始销售话术
            analysis: 已完成的三维度评分

        Returns:
            Suggestion 建议列表
        """
        suggestions = []
        dimensions = [
            ("clarity", "清晰度", analysis.clarity),
            ("professionalism", "专业性", analysis.professionalism),
            ("persuasiveness", "说服力", analysis.persuasiveness),
        ]

        for dim_key, dim_name, score in dimensions:
            if score < 7:
                suggestion = self._build_suggestion_for_dimension(
                    dim_key, dim_name, score, message
                )
                suggestions.append(suggestion)

        return suggestions

    def _build_suggestion_for_dimension(
        self, dim_key: str, dim_name: str, score: int, message: str
    ) -> Suggestion:
        """为单个低分维度构建改进建议。

        根据维度特征选择针对性的建议模板。

        Args:
            dim_key: 维度标识
            dim_name: 维度中文名
            score: 当前分数
            message: 原始话术

        Returns:
            Suggestion 改进建议对象
        """
        advice_map = {
            "clarity": {
                "advice": "语句结构需要优化，建议使用'总-分-总'结构：先说核心观点，再展开细节，最后总结。",
                "example": "核心观点是XX，具体来说：（数据支撑1）+（数据支撑2），所以XX。",
            },
            "professionalism": {
                "advice": "缺少专业术语和数据支撑，建议引用临床试验数据或权威指南推荐。",
                "example": "根据XX研究显示（n=XXX），患者使用后HbA1c平均降低X%，p<0.05。",
            },
            "persuasiveness": {
                "advice": "论证逻辑不够有力，建议采用'痛点-方案-证据-行动'四步法增强说服力。",
                "example": "您提到的XX问题确实存在（痛点），我们的方案是XX（方案），临床数据证明XX（证据），建议您可以先试用（行动）。",
            },
        }

        template = advice_map.get(dim_key, {
            "advice": f"{dim_name}有待提升，建议加强相关训练。",
            "example": "",
        })

        return Suggestion(
            dimension=dim_key,
            current_score=score,
            advice=template["advice"],
            example=template["example"],
        )

    def _parse_expression_response(self, content: str) -> ExpressionAnalysis:
        """解析 LLM 表达分析响应。

        从 LLM 返回文本中提取三个维度的数值评分。

        Args:
            content: LLM 响应原始文本

        Returns:
            ExpressionAnalysis 解析结果
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

    @staticmethod
    def _rule_based_expression_analysis(message: str) -> ExpressionAnalysis:
        """基于规则的表达能力分析降级方案。

        通过多维度文本特征分析话术质量：
        - 清晰度：句子长度分布、标点使用、长/短句比例、词汇多样性
        - 专业性：专业术语密度、数据引用频率
        - 说服力：数据百分比、对比论证、行动号召信号

        Args:
            message: 销售消息

        Returns:
            ExpressionAnalysis 规则分析结果
        """
        clarity = 5
        professionalism = 5
        persuasiveness = 5

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

        professional_terms = [
            "临床", "数据", "研究", "试验", "HbA1c",
            "%", "患者", "治疗", "疗效", "安全性",
            "副作用", "依从性", "发生率", "mg", "ml", "剂量", "方案",
        ]
        term_count = sum(1 for term in professional_terms if term in message)
        professionalism = min(10, 5 + term_count)

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

    @staticmethod
    def _generate_suggestions_from_rules(analysis: ExpressionAnalysis) -> list[Suggestion]:
        """基于规则分析结果生成改进建议。

        当 LLM 不可用时，根据各维度分数自动生成通用建议。

        Args:
            analysis: 规则分析结果

        Returns:
            Suggestion 建议列表
        """
        suggestions = []
        dims = [
            ("clarity", "清晰度", analysis.clarity),
            ("professionalism", "专业性", analysis.professionalism),
            ("persuasiveness", "说服力", analysis.persuasiveness),
        ]
        for key, name, score in dims:
            if score < 7:
                suggestions.append(Suggestion(
                    dimension=key,
                    current_score=score,
                    advice=f"{name}得分偏低({score}/10)，建议针对性提升此维度表达能力。",
                    example="",
                ))
        return suggestions


def calculate_overall_score(
    coverage_result: CoverageResult,
    expression_result: ExpressionResult,
) -> float:
    """综合计算最终得分。

    将语义覆盖率和表达能力按 50:50 权重合并为百分制总分。

    Args:
        coverage_result: 语义覆盖检测结果
        expression_result: 表达能力评估结果

    Returns:
        综合评分（0-100）
    """
    expr = expression_result.analysis
    coverage_score = coverage_result.coverage_rate * 50
    expression_score = (
        (expr.clarity + expr.professionalism + expr.persuasiveness) / 30 * 50
    )
    return round(coverage_score + expression_score, 2)
