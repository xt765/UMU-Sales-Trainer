"""对话分析模块。

实现 ConversationAnalyst（对话分析师）Agent，负责分析销售发言的意图、
判断对话阶段、检测异议信号，为后续评估和引导提供上下文。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from umu_sales_trainer.services.llm import LLMService

if TYPE_CHECKING:
    from umu_sales_trainer.models.conversation import Message
    from umu_sales_trainer.models.customer import CustomerProfile

logger = logging.getLogger(__name__)


@dataclass
class ConversationAnalysis:
    """对话分析结果。

    Attributes:
        stage: 当前销售阶段标识
        intent: 销售意图描述
        objections: 检测到的异议信号列表
        sentiment: 情感倾向
        confidence: 分析置信度（0-1）
    """

    stage: str = "opening"
    intent: str = ""
    objections: list[str] = field(default_factory=list)
    sentiment: str = "neutral"
    confidence: float = 0.5


STAGE_DEFINITIONS = """【销售阶段定义】
- opening（开场破冰）：问候、自我介绍、建立初步联系、引起兴趣
- needs_discovery（需求探查）：询问客户痛点、了解现状、挖掘深层需求
- presentation（产品呈现）：介绍产品特点、展示疗效数据、说明安全性
- objection_handling（异议处理）：回应客户反对意见、化解疑虑、提供证据
- closing（缔结成交）：推动下一步行动、确认合作意向、安排跟进"""

OBJECTION_KEYWORDS = {
    "价格": ["贵", "太贵", "预算", "成本", "便宜", "降价", "折扣"],
    "安全性": ["副作用", "不良反应", "肝肾", "毒性", "风险", "安全吗", "禁忌症", "相互作用"],
    "证据": ["证据", "研究", "试验", "文献", "论文", "数据来源", "第三方"],
    "竞品": ["其他产品", "同类药", "竞品", "对比", "别的品牌", "之前用过"],
    "时机": ["再考虑", "不急", "等等", "下次再说", "暂时不需要"],
    "用法便利性": ["一天几次", "用法复杂", "不方便", "麻烦", "依从性", "服用方式"],
    "医保报销": ["医保", "报销", "自费", "进医保", "报销比例", "患者负担"],
    "处方限制": ["处方", "限制", "适应症", "开药", "非处方"],
    "疗效疑虑": ["效果不明显", "没效果", "起效慢", "疗效差", "多久见效", "有效率"],
}


class ConversationAnalyst:
    """对话分析师 Agent。

    通过 LLM 分析销售人员的发言，识别当前所处的销售阶段、
    意图方向和潜在异议信号。这是 Agentic RAG 工作流的第一个分析节点，
    为后续的语义覆盖评估和表达质量评估提供关键上下文。

    Attributes:
        llm_service: LLM 服务实例，用于调用大语言模型进行分析
    """

    def __init__(self, llm_service: LLMService) -> None:
        """初始化对话分析师。

        Args:
            llm_service: LLM 服务实例，用于生成分析结果
        """
        self._llm = llm_service

    def analyze(
        self,
        sales_message: str,
        customer_profile: CustomerProfile | None,
        conversation_history: list[Message] | None = None,
    ) -> ConversationAnalysis:
        """分析销售发言。

        调用 LLM 识别销售阶段、意图和潜在异议信号，
        并通过规则引擎补充关键词级别的异议检测。

        Args:
            sales_message: 销售人员最新发言内容
            customer_profile: 客户画像信息（用于调整分析视角）
            conversation_history: 对话历史消息列表（用于判断阶段演进）

        Returns:
            ConversationAnalysis 结构化分析结果
        """
        prompt = self._build_prompt(sales_message, customer_profile, conversation_history)

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            result = self._parse_response(response.content)
        except Exception as e:
            logger.warning("LLM conversation analysis failed, using rule-based fallback: %s", e)
            result = self._rule_based_analysis(sales_message)

        rule_objections = self._detect_objections_by_keywords(sales_message)
        result.objections = list(set(result.objections + rule_objections))

        return result

    def _build_prompt(
        self,
        sales_message: str,
        customer_profile: CustomerProfile | None,
        conversation_history: list[Message] | None,
    ) -> str:
        """构建对话分析提示词。

        将客户画像和对话历史作为上下文注入 prompt，
        让 LLM 能够结合完整场景进行准确判断。

        Args:
            sales_message: 销售发言内容
            customer_profile: 客户画像
            conversation_history: 对话历史

        Returns:
            格式化的提示词字符串
        """
        customer_context = ""
        if customer_profile:
            customer_context = f"""
客户背景：
- 姓名：{customer_profile.name or "未知"}
- 职位：{customer_profile.position or "未知"}
- 关注点：{", ".join(customer_profile.concerns[:3]) if customer_profile.concerns else "未知"}
- 性格倾向：{customer_profile.personality or "未知"}
"""

        history_context = ""
        if conversation_history:
            recent_msgs = conversation_history[-4:]
            history_lines = []
            for msg in recent_msgs:
                role_label = "销售" if msg.role == "user" else "客户"
                history_lines.append(f"[{role_label}]: {msg.content}")
            history_context = "\n近期对话历史：\n" + "\n".join(history_lines)

        return f"""你是一位拥有15年医药销售培训经验的资深对话分析师。
请仔细分析以下销售发言，判断其处于哪个销售阶段。

{STAGE_DEFINITIONS}

{customer_context}
{history_context}

【待分析的最新销售发言】
{sales_message}

请严格以 JSON 格式返回（不要有其他内容）：
{{
    "stage": "opening|needs_discovery|presentation|objection_handling|closing",
    "intent": "一句话描述这位销售代表想达成什么目标",
    "objections": ["如果发言中隐含了客户的潜在反对或顾虑，列出关键词；否则为空数组。常见类别包括：价格/安全性（副作用、禁忌症）/证据/竞品/时机/用法便利性/医保报销/处方限制/疗效疑虑"],
    "sentiment": "positive(积极自信)|neutral(平稳客观)|cautious(谨慎保守)"
}}"""

    def _parse_response(self, content: str) -> ConversationAnalysis:
        """解析 LLM 返回的 JSON 响应。

        支持纯 JSON 和 markdown 代码块包裹两种格式。

        Args:
            content: LLM 原始响应文本

        Returns:
            解析后的 ConversationAnalysis 对象
        """
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```", 1)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.split("```")[0].strip()

        try:
            data = json.loads(text)
            return ConversationAnalysis(
                stage=data.get("stage", "opening"),
                intent=data.get("intent", ""),
                objections=data.get("objections", []),
                sentiment=data.get("sentiment", "neutral"),
                confidence=0.8,
            )
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse conversation analysis JSON: %s", e)
            return ConversationAnalysis(confidence=0.2)

    @staticmethod
    def _rule_based_analysis(message: str) -> ConversationAnalysis:
        """基于规则的分析降级方案。

        当 LLM 不可用时，通过关键词匹配做基础判断。

        Args:
            message: 销售发言内容

        Returns:
            规则分析结果
        """
        stage = "opening"
        intent = ""

        opening_words = ["您好", "你好", "感谢", "很高兴", "介绍", "我是"]
        discovery_words = ["您觉得", "您目前", "请问", "了解到", "情况如何"]
        presentation_words = ["我们的产品", "这款", "临床", "疗效", "数据显示", "HbA1c", "安全性"]
        objection_words = ["关于您的顾虑", "您提到的", "确实", "理解您的担心", "这个问题的答案是"]
        closing_words = ["那么", "接下来", "我们可以", "建议", "安排", "跟进"]

        message_lower = message.lower()
        if any(w in message_lower for w in objection_words):
            stage = "objection_handling"
            intent = "回应客户异议或疑虑"
        elif any(w in message_lower for w in closing_words):
            stage = "closing"
            intent = "推动下一步行动"
        elif any(w in message_lower for w in presentation_words):
            stage = "presentation"
            intent = "呈现产品特点和疗效"
        elif any(w in message_lower for w in discovery_words):
            stage = "needs_discovery"
            intent = "探查客户需求和痛点"
        elif any(w in message_lower for w in opening_words):
            stage = "opening"
            intent = "开场破冰并建立关系"

        sentiment = "positive"
        cautious_words = ["可能", "也许", "不太确定", "大概", "还需要确认"]
        if any(w in message_lower for w in cautious_words):
            sentiment = "cautious"

        return ConversationAnalysis(
            stage=stage,
            intent=intent,
            objections=[],
            sentiment=sentiment,
            confidence=0.5,
        )

    @staticmethod
    def _detect_objections_by_keywords(message: str) -> list[str]:
        """通过关键词检测发言中隐含的异议信号。

        扫描预定义的异议关键词库，匹配到即认为存在对应类型的异议。

        Args:
            message: 销售发言内容

        Returns:
            匹配到的异议类型名称列表
        """
        detected = []
        message_lower = message.lower()
        for category, keywords in OBJECTION_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                detected.append(category)
        return detected
