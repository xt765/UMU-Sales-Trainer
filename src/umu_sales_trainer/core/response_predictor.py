"""推荐话术生成器（Quick Reply Generator）模块。

基于动态加权 RAG 知识库召回结果和对话上下文（含客户最新回复），
为销售代表生成 3 个不同策略的可点击推荐话术选项。
销售代表可直接点击发送，无需手动打字。

三种策略：
- data_evidence（数据导向）：引用临床数据、研究样本量、循证等级，专业严谨
- empathy_relation（共情切入）：理解医生/患者痛点，拉近距离，温和自然
- benefit_action（利益推动）：强调患者获益、使用场景、行动号召
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from umu_sales_trainer.services.llm import LLMService, create_llm

if TYPE_CHECKING:
    from umu_sales_trainer.core.analyzer import ConversationAnalysis
    from umu_sales_trainer.core.evaluator import CoverageResult
    from umu_sales_trainer.core.hybrid_search import HybridSearchEngine
    from umu_sales_trainer.models.customer import CustomerProfile

logger = logging.getLogger(__name__)


@dataclass
class SuggestedReply:
    """单条推荐话术（Quick Reply）。

    销售代表可直接点击发送的预设话术选项，
    由 RAG 知识库召回 + LLM 生成。

    Attributes:
        strategy: 策略标识符
        strategy_label: 策略中文标签
        content: 推荐话术文本（可直接发送，50-100字）
        confidence: 推荐置信度 0-1（基于 RAG 命中质量）
        source_hints: RAG 来源提示列表
    """

    strategy: str = ""
    strategy_label: str = ""
    content: str = ""
    confidence: float = 0.0
    source_hints: list[str] = field(default_factory=list)


STRATEGY_CONFIG: dict[str, dict[str, str]] = {
    "data_evidence": {
        "label": "数据导向",
        "icon": "bar-chart-3",
        "color": "#1a4332",
        "system_prompt": (
            "你是一位资深医药销售培训师。根据当前对话上下文和产品知识库信息，"
            "为销售代表推荐一条以**数据和证据**为核心的销售话术。\n\n"
            "要求：\n"
            "- 话术应引用具体临床数据、研究样本量、循证等级\n"
            "- 专业严谨有说服力，适合在'产品呈现'阶段使用\n"
            "- 50-100字，简洁有力\n"
            "- **直接输出话术文本本身**，不要加任何前缀说明或引号包裹"
        ),
    },
    "empathy_relation": {
        "label": "共情切入",
        "icon": "heart",
        "color": "#c4791a",
        "system_prompt": (
            "你是一位资深医药销售培训师。根据当前对话上下文和产品知识库信息，"
            "为销售代表推荐一条以**建立信任和共情**为核心的销售话术。\n\n"
            "要求：\n"
            "- 话术应体现对医生/患者实际痛点的理解\n"
            "- 温和自然、拉近距离，适合在'需求探查'或'异议处理'阶段使用\n"
            "- 50-100字\n"
            "- **直接输出话术文本本身**，不要加任何前缀说明或引号包裹"
        ),
    },
    "benefit_action": {
        "label": "利益推动",
        "icon": "target",
        "color": "#7c3aed",
        "system_prompt": (
            "你是一位资深医药销售培训师。根据当前对话上下文和产品知识库信息，"
            "为销售代表推荐一条以**患者获益和行动引导**为核心的销售话术。\n\n"
            "要求：\n"
            "- 话术应强调具体使用场景、患者获益点和行动号召\n"
            "- 务实有推动力，适合在'缔结成交'阶段使用\n"
            "- 50-100字\n"
            "- **直接输出话术文本本身**，不要加任何前缀说明或引号包裹"
        ),
    },
}

STAGE_WEIGHT_MAP: dict[str, dict[str, float]] = {
    "opening": {"excellent_samples": 1.5},
    "needs_discovery": {"product_knowledge": 1.2},
    "presentation": {"product_knowledge": 1.5},
    "objection_handling": {"objection_handling": 1.5},
    "closing": {"excellent_samples": 1.4},
}

CONCERN_WEIGHT_MAP: dict[str, tuple[str, float]] = {
    "HbA1c": ("product_knowledge", 0.2),
    "低血糖": ("product_knowledge", 0.2),
    "安全性": ("objection_handling", 0.2),
    "依从性": ("excellent_samples", 0.2),
}


class ResponsePredictor:
    """推荐话术生成器（Quick Reply Generator）。

    利用混合搜索引擎（HybridSearchEngine）从知识库中召回相关内容，
    结合对话上下文动态加权，通过 LLM 生成 3 个不同策略的销售话术建议。
    销售代表可直接点击发送，无需手动打字。

    Attributes:
        llm_service: LLM 服务实例
        search_engine: 混合搜索引擎实例（可选）
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        search_engine: HybridSearchEngine | None = None,
    ) -> None:
        """初始化推荐话术生成器。

        Args:
            llm_service: LLM 服务实例，未传入时自动创建默认实例
            search_engine: 混合搜索引擎实例，可选；无搜索引擎时降级为纯 LLM 模式
        """
        self.llm_service = llm_service or create_llm("dashscope")
        self.search_engine = search_engine

    def predict(
        self,
        sales_message: str,
        last_ai_response: str,
        context: dict[str, Any],
    ) -> list[SuggestedReply]:
        """生成 3 个策略的推荐话术（Quick Reply）。

        根据销售发言、客户（张主任）最新回复和对话上下文，
        分别从数据导向、共情切入、利益推动三种策略角度
        为销售代表生成可点击发送的话术建议。

        Args:
            sales_message: 销售人员发送的最新消息
            last_ai_response: 客户（张主任/AI模拟）的最新回复或提问
            context: 对话上下文字典，包含：
                - conversation_analysis: 对话分析结果
                - customer_profile: 客户画像
                - coverage_result: 语义覆盖检测结果

        Returns:
            包含 3 个 SuggestedReply 的列表，按策略固定顺序排列
        """
        conv_analysis: ConversationAnalysis | None = context.get("conversation_analysis")
        customer: CustomerProfile | None = context.get("customer_profile")
        _coverage_result: CoverageResult | None = context.get("coverage_result")

        stage = conv_analysis.stage if conv_analysis else "opening"
        objections = conv_analysis.objections if conv_analysis else []
        concerns = customer.concerns if customer else []

        weights = self._compute_dynamic_weights(stage, objections, concerns)

        rag_context = self._retrieve_rag_context(sales_message, weights)

        suggestions: list[SuggestedReply] = []
        for strategy_id, config in STRATEGY_CONFIG.items():
            try:
                suggestion = self._generate_single_prediction(
                    strategy_id=strategy_id,
                    config=config,
                    sales_message=sales_message,
                    rag_context=rag_context,
                    customer=customer,
                    weights=weights,
                    last_ai_response=last_ai_response,
                )
                suggestions.append(suggestion)
            except Exception as e:
                logger.warning(
                    "Failed to generate suggestion for strategy %s: %s",
                    strategy_id,
                    e,
                )
                suggestions.append(
                    SuggestedReply(
                        strategy=strategy_id,
                        strategy_label=config["label"],
                        content="话术生成失败，请重试。",
                        confidence=0.0,
                        source_hints=[],
                    )
                )

        return suggestions

    def _compute_dynamic_weights(
        self,
        stage: str,
        objections: list[str],
        concerns: list[str],
    ) -> dict[str, float]:
        """根据对话上下文计算各知识库 Collection 的动态权重。

        权重受三个因素影响：
        1. 销售阶段：不同阶段偏重不同类型的知识
        2. 异议信号：检测到异议时提升异议处理库权重
        3. 客户关注点：匹配关键词时微调对应领域权重

        Args:
            stage: 当前销售阶段标识
            objections: 检测到的异议信号列表
            concerns: 客户关注点列表

        Returns:
            各 Collection 名称到权重系数的映射字典
        """
        base: dict[str, float] = {
            "product_knowledge": 1.0,
            "objection_handling": 1.0,
            "excellent_samples": 1.0,
        }

        stage_override = STAGE_WEIGHT_MAP.get(stage)
        if stage_override:
            for collection, boost in stage_override.items():
                base[collection] = boost

        if objections:
            base["objection_handling"] = base.get("objection_handling", 1.0) + 0.3

        for concern in concerns:
            for keyword, (collection, boost) in CONCERN_WEIGHT_MAP.items():
                if keyword in concern:
                    base[collection] = base.get(collection, 1.0) + boost

        return base

    def _retrieve_rag_context(
        self,
        query: str,
        weights: dict[str, float],
    ) -> str:
        """从知识库检索相关信息并拼接为上下文文本。

        使用动态权重调用混合搜索引擎，将召回结果格式化为
        可注入 LLM prompt 的参考文本。

        Args:
            query: 检索查询文本（通常为销售发言）
            weights: 各 Collection 的动态权重

        Returns:
            格式化后的 RAG 知识文本，无搜索引擎时返回空字符串
        """
        if self.search_engine is None:
            return ""

        try:
            results = asyncio.get_event_loop().run_until_complete(
                self.search_engine.search(
                    query=query,
                    collections={},
                    weights=weights,
                )
            )
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)
            return ""

        if not results:
            return ""

        top_results = results[:5]
        context_parts: list[str] = []
        seen_sources: set[str] = set()

        for item in top_results:
            content = item.get("content", "")
            source = item.get("collection", "未知来源")
            score = item.get("final_score", item.get("rrf_score", 0))

            if content and len(content) > 10:
                context_parts.append(f"[{source} 相关度:{score:.2f}] {content}")
                seen_sources.add(source)

        return "\n\n".join(context_parts) if context_parts else ""

    def _generate_single_prediction(
        self,
        strategy_id: str,
        config: dict[str, str],
        sales_message: str,
        rag_context: str,
        customer: CustomerProfile | None,
        weights: dict[str, float],
        last_ai_response: str = "",
    ) -> SuggestedReply:
        """为单个策略生成一条推荐话术。

        构建差异化 prompt（包含策略倾向的系统提示 + RAG 知识 + 客户最新回复 + 上下文），
        调用 LLM 生成销售话术并计算置信度。

        Args:
            strategy_id: 策略标识符
            config: 策略配置（含 label / icon / color / system_prompt）
            sales_message: 销售人员发言
            rag_context: RAG 召回的知识上下文
            customer: 客户画像
            weights: 动态权重（用于置信度计算）
            last_ai_response: 客户（张主任）的最新回复或提问

        Returns:
            单条 SuggestedReply
        """
        system_prompt = config["system_prompt"]

        user_content_parts: list[str] = []

        if last_ai_response:
            user_content_parts.append(f"【客户（张主任）最新回复/提问】\n{last_ai_response}")

        user_content_parts.append(f"【销售代表刚发送的内容】\n{sales_message}")

        if rag_context:
            user_content_parts.append(
                f"\n【产品知识库参考信息】（可从中提取数据/案例融入话术）\n{rag_context}"
            )

        if customer and customer.concerns:
            concerns_text = "、".join(customer.concerns[:3])
            user_content_parts.append(f"\n【客户关注点】{concerns_text}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(user_content_parts)),
        ]

        response = self.llm_service.invoke(messages)
        content = response.content.strip() if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join(str(item) for item in content).strip()

        confidence = self._calculate_confidence(rag_context, weights)
        source_hints = self._extract_source_hints(rag_context)

        return SuggestedReply(
            strategy=strategy_id,
            strategy_label=config["label"],
            content=content,
            confidence=confidence,
            source_hints=source_hints,
        )

    @staticmethod
    def _calculate_confidence(
        rag_context: str,
        weights: dict[str, float],
    ) -> float:
        """基于 RAG 命中质量和权重计算推荐话术的置信度。

        置信度反映推荐话术的知识支撑强度：
        - 有 RAG 内容命中 → 基础分 0.55 + 内容长度加成
        - 无 RAG 内容 → 基础分 0.35（纯 LLM 生成，可靠性较低）
        - 最高不超过 0.95

        Args:
            rag_context: RAG 召回的上下文文本
            weights: 动态权重字典

        Returns:
            0-1 之间的置信度分数
        """
        if not rag_context:
            return 0.35

        base = 0.55
        length_bonus = min(len(rag_context) / 500, 0.25)
        weight_avg = sum(weights.values()) / max(len(weights), 1)
        weight_bonus = min((weight_avg - 1.0) * 0.15, 0.15)

        raw = base + length_bonus + weight_bonus
        return round(min(max(raw, 0.0), 0.95), 2)

    @staticmethod
    def _extract_source_hints(rag_context: str) -> list[str]:
        """从 RAG 上下文中提取来源提示。

        解析 [source_name] 格式的来源标记，用于前端展示
        数据来源信息。

        Args:
            rag_context: RAG 召回上下文文本

        Returns:
            来源名称列表（去重）
        """
        import re

        pattern = r"\[(.+?)\s+相关度"
        matches = re.findall(pattern, rag_context)
        source_map = {
            "product_knowledge": "产品知识库",
            "objection_handling": "异议处理策略",
            "excellent_samples": "优秀话术示例",
        }

        hints: list[str] = []
        seen: set[str] = set()
        for m in matches:
            display = source_map.get(m, m)
            if display not in seen:
                hints.append(display)
                seen.add(display)

        return hints[:3]
