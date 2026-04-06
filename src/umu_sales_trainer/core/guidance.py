"""智能引导模块。

实现 GuidanceMentor（引导导师）Agent，综合语义覆盖、表达质量和对话分析结果，
生成结构化、可操作的销售培训引导建议。

这是 Agentic RAG 工作流中唯一面向用户输出的 Agent，
其产出直接渲染为前端「智能引导」面板。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from umu_sales_trainer.services.llm import LLMService

if TYPE_CHECKING:
    from umu_sales_trainer.core.analyzer import ConversationAnalysis
    from umu_sales_trainer.core.evaluator import CoverageResult, ExpressionResult
    from umu_sales_trainer.models.customer import CustomerProfile
    from umu_sales_trainer.models.semantic import SemanticPoint

logger = logging.getLogger(__name__)


@dataclass
class GuidanceItem:
    """单条引导建议项。

    Attributes:
        gap: 缺失或不足的方面描述
        urgency: 紧急程度（high / medium / low）
        suggestion: 具体改进建议
        talking_point: 参考话术范例
        expected_effect: 预期效果说明
    """

    gap: str = ""
    urgency: str = "medium"
    suggestion: str = ""
    talking_point: str = ""
    expected_effect: str = ""


@dataclass
class GuidanceResult:
    """智能引导结果。

    Attributes:
        priority_list: 按紧急度排序的引导建议列表
        summary: 一句话总结
        is_actionable: 是否需要立即行动
    """

    priority_list: list[GuidanceItem] = field(default_factory=list)
    summary: str = ""
    is_actionable: bool = False


URGENCY_THRESHOLDS = {"high": 0.5, "medium": 0.8, "low": 1.0}


class GuidanceMentor:
    """引导导师 Agent。

    综合来自 ConversationAnalyst、SemanticCoverageExpert 和 ExpressionCoach 的评估结果，
    生成结构化的、按优先级排序的销售培训引导建议。

    与旧版模板化 guidance 不同，本 Agent：
    - 根据覆盖率动态决定是否需要引导（≥80% 自动折叠）
    - 按紧急度排序改进项（高/中/低三级）
    - 每条建议包含：问题描述 + 具体做法 + 参考话术范例
    - 引导语气鼓励性而非批评性

    Attributes:
        llm_service: LLM 服务实例，用于生成个性化引导内容
    """

    def __init__(self, llm_service: LLMService) -> None:
        """初始化引导导师。

        Args:
            llm_service: LLM 服务实例
        """
        self._llm = llm_service

    def generate_guidance(
        self,
        coverage_result: CoverageResult,
        expression_result: ExpressionResult,
        semantic_points: list[SemanticPoint],
        conversation_analysis=None,
        customer_profile: CustomerProfile | None = None,
    ) -> GuidanceResult:
        """综合评估结果生成结构化引导。

        当覆盖率达到 80% 以上时，返回空引导（is_actionable=False），
        前端面板将自动折叠。

        Args:
            coverage_result: 语义覆盖检测结果
            expression_result: 表达能力评估结果
            conversation_analysis: 对话分析结果（可选）
            semantic_points: 完整语义点列表（用于获取未覆盖点的描述）
            customer_profile: 客户画像（可选）

        Returns:
            GuidanceResult 结构化引导结果
        """
        if coverage_result.coverage_rate >= URGENCY_THRESHOLDS["low"]:
            return GuidanceResult(
                summary="表现优秀，继续保持！",
                is_actionable=False,
            )

        items = self._build_priority_items(
            coverage_result, expression_result, conversation_analysis, semantic_points
        )

        if not items:
            return GuidanceResult(
                summary="整体表现良好。",
                is_actionable=False,
            )

        items.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.urgency, 1))
        summary = self._generate_summary(items, coverage_result)

        return GuidanceResult(
            priority_list=items,
            summary=summary,
            is_actionable=True,
        )

    def _build_priority_items(
        self,
        coverage_result: CoverageResult,
        expression_result: ExpressionResult,
        conversation_analysis: ConversationAnalysis | None,
        semantic_points: list[SemanticPoint],
    ) -> list[GuidanceItem]:
        """构建优先级引导项列表。

        从三个维度收集需要改进的点并分配紧急度：
        - 未覆盖的语义点 → high urgency（核心缺失）
        - 表达低分维度（<6分）→ high urgency（明显短板）
        - 表达中低分维度（6-7分）→ medium urgency（有提升空间）
        - 异议信号检测 → medium urgency（需关注）

        Args:
            coverage_result: 覆盖检测结果
            expression_result: 表达评估结果
            conversation_analysis: 对话分析结果
            semantic_points: 语义点列表

        Returns:
            GuidanceItem 列表
        """
        items: list[GuidanceItem] = []

        point_map = {p.point_id: p for p in semantic_points}

        for pid in coverage_result.uncovered_points:
            sp = point_map.get(pid)
            description = sp.description if sp else pid
            item = GuidanceItem(
                gap=f"未充分覆盖：{description}",
                urgency="high",
                suggestion=f"在下次发言中主动提及{description}相关的内容",
                talking_point=self._generate_talking_point(description),
                expected_effect=f"提升语义点覆盖率，当前 {coverage_result.coverage_rate:.0%} → 目标 100%",
            )
            items.append(item)

        expr = expression_result.analysis
        dim_config = {
            "clarity": {
                "name": "清晰度",
                "advice": "优化语句结构，使用'总-分-总'模式组织表达",
                "example": f"先说核心观点（关于{self._get_product_context()}），再展开具体数据支撑，最后总结要点。",
            },
            "professionalism": {
                "name": "专业性",
                "advice": "引用临床试验数据、权威指南或真实案例增强说服力",
                "example": "根据XX研究（n=XXX），患者HbA1c平均降低X%，且安全性数据优于对照组。",
            },
            "persuasiveness": {
                "name": "说服力",
                "advice": "采用'痛点-方案-证据-行动'四步法构建论证逻辑链",
                "example": "您提到的XX问题确实存在（痛点），我们的方案是XX（方案），临床证明XX（证据），建议先试用（行动）。",
            },
        }

        for dim_key, config in dim_config.items():
            score = getattr(expr, dim_key, 5)
            if score < 6:
                urgency = "high"
            elif score < 7:
                urgency = "medium"
            else:
                continue

            items.append(
                GuidanceItem(
                    gap=f"{config['name']}偏低（{score}/10分）",
                    urgency=urgency,
                    suggestion=config["advice"],
                    talking_point=config["example"],
                    expected_effect=f"提升{config['name']}至7分以上",
                )
            )

        if conversation_analysis and conversation_analysis.objections:
            for obj in conversation_analysis.objections[:2]:
                items.append(
                    GuidanceItem(
                        gap=f"检测到异议信号：{obj}",
                        urgency="medium",
                        suggestion=f"准备{obj}相关的应对策略和证据材料",
                        talking_point="我理解您的顾虑，这一点确实很重要。让我从XX角度为您详细说明...",
                        expected_effect="提前化解潜在异议，推进对话进程",
                    )
                )

        return items

    @staticmethod
    def _get_product_context() -> str:
        """获取产品上下文信息用于话术模板。

        Returns:
            产品名称占位符
        """
        return "产品疗效与安全性"

    @staticmethod
    def _generate_talking_point(point_description: str) -> str:
        """根据语义点描述生成参考话术。

        Args:
            point_description: 语义点中文描述

        Returns:
            参考话术文本
        """
        return f"关于{point_description}，我想特别强调的是：我们的产品在这方面具有显著优势..."

    @staticmethod
    def _generate_summary(items: list[GuidanceItem], coverage_result: CoverageResult) -> str:
        """生成一句话总结。

        Args:
            items: 引导项列表
            coverage_result: 覆盖检测结果

        Returns:
            总结文本
        """
        high_count = sum(1 for i in items if i.urgency == "high")
        total_count = len(items)

        if high_count > 0:
            return f"有{high_count}项急需改进（共{total_count}项），建议优先处理标红项目。"

        return f"有{total_count}项可优化，继续加油！"

    def generate_guidance_with_llm(
        self,
        coverage_result: CoverageResult,
        expression_result: ExpressionResult,
        semantic_points: list[SemanticPoint],
        conversation_analysis=None,
        customer_profile: CustomerProfile | None = None,
    ) -> GuidanceResult:
        """通过 LLM 生成更个性化的引导内容。

        对基础引导结果的每条建议进行 LLM 增强，
        使话术范例更加贴合实际场景。

        Args:
            coverage_result: 覆盖检测结果
            expression_result: 表达评估结果
            conversation_analysis: 对话分析结果
            semantic_points: 语义点列表
            customer_profile: 客户画像

        Returns:
            GuidanceResult LLM 增强后的引导结果
        """
        base_result = self.generate_guidance(
            coverage_result,
            expression_result,
            conversation_analysis,
            semantic_points,
            customer_profile,
        )

        if not base_result.is_actionable or not base_result.priority_list:
            return base_result

        try:
            enhanced_items = []
            for item in base_result.priority_list:
                enhanced = self._enhance_item_with_llm(item, customer_profile)
                enhanced_items.append(enhanced)
            base_result.priority_list = enhanced_items
        except Exception as e:
            logger.warning("LLM guidance enhancement failed: %s", e)

        return base_result

    def _enhance_item_with_llm(
        self, item: GuidanceItem, customer_profile: CustomerProfile | None
    ) -> GuidanceItem:
        """用 LLM 增强单条引导建议的话术范例。

        Args:
            item: 原始引导项
            customer_profile: 客户画像（用于定制话术）

        Returns:
            增强后的引导项
        """
        context = ""
        if customer_profile and customer_profile.position:
            context = f"客户职位：{customer_profile.position}"

        prompt = (
            f"你是一位销售培训导师。请针对以下改进建议，生成一句简短的参考话术。\n\n"
            f"{context}\n"
            f"待改进项：{item.gap}\n"
            f"建议方向：{item.suggestion}\n\n"
            f"请只输出参考话术（不超过50字），不要其他内容。"
        )

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            item.talking_point = response.content.strip()
        except Exception as e:
            logger.warning("LLM talking point generation failed: %s", e)

        return item
