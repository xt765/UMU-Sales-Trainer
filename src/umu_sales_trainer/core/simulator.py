"""客户模拟器模块。

实现 AI 客户模拟器，用于销售训练场景。AI 扮演客户（内分泌科主任），
与销售人员进行多轮对话，帮助提升销售技能。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from umu_sales_trainer.models.conversation import Message
from umu_sales_trainer.services.llm import LLMService

if TYPE_CHECKING:
    from umu_sales_trainer.core.hybrid_search import HybridSearchEngine


class CustomerSimulator:
    """AI 客户模拟器。

    在销售训练场景中，AI 扮演客户（内分泌科主任），与销售人员进行
    真实感的多轮对话。通过结合 RAG 知识库生成专业的回复内容。

    Attributes:
        llm_service: LLM 服务实例，用于生成 AI 回复
        search_engine: 混合搜索引擎实例，用于 RAG 知识检索
        system_prompt: 系统提示词，定义 AI 客户的角色和行为

    Example:
        >>> simulator = CustomerSimulator(llm_service, search_engine)
        >>> response = simulator.generate_response(
        ...     sales_message="这是我们的新产品介绍...",
        ...     context={"customer_profile": profile, "product_info": info}
        ... )
    """

    llm_service: LLMService
    search_engine: HybridSearchEngine | None
    system_prompt: str

    def __init__(
        self,
        llm_service: LLMService,
        search_engine: HybridSearchEngine | None = None,
    ) -> None:
        """初始化客户模拟器。

        Args:
            llm_service: LLM 服务实例
            search_engine: 混合搜索引擎实例，用于 RAG 检索（可选）
        """
        self.llm_service = llm_service
        self.search_engine = search_engine
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """构建系统提示词。

        定义 AI 客户的角色设定：内分泌科主任，专业、谨慎，关注患者获益、
        临床数据和价格等专业话题。对话风格专业但保持适当距离感。

        Returns:
            系统提示词字符串
        """
        return """你是张主任，内分泌科主任医师，医学博士，在三甲医院内分泌科
工作已有20年。你以专业、严谨、谨慎著称，对新事物持开放但审慎的态度。

## 专业背景
- 擅长糖尿病、甲状腺疾病、代谢综合征等内分泌代谢疾病的诊治
- 始终将患者获益放在首位，任何新药或新疗法都必须有充分的临床证据
- 对临床数据和真实世界研究结果非常关注
- 重视药物的安全性数据和不良反应信息

## 性格特点
- 专业但不刻板，愿意倾听但保持独立判断
- 谨慎务实，不会轻易被推销话术打动
- 提问直接且深入，关注核心问题
- 对夸大的疗效宣传持怀疑态度

## 对话风格
- 专业但有适度距离感，不过分热情也不冷淡
- 习惯用医学术语交流，但会适时解释
- 会主动提问以获取更多信息
- 关心患者的长期预后和生活质量

## 常用关注点
当与医药代表交流时，你通常会关注：
1. 产品的临床三期试验数据和循证医学证据
2. 患者使用后的真实获益（疗效、安全性、生活质量）
3. 药物经济学评价（性价比、医保覆盖）
4. 与现有标准治疗方案相比的优势
5. 不良反应和禁忌症信息
6. 剂量和用法是否方便患者

请始终以张主任的身份和语气回复，保持专业严谨的医学工作者形象。"""

    async def _retrieve_knowledge(self, query: str) -> str:
        """从知识库检索相关信息。

        Args:
            query: 检索查询文本

        Returns:
            检索到的相关知识文本，如果无检索引擎则返回空字符串
        """
        if self.search_engine is None:
            return ""

        results = await self.search_engine.search(
            query=query,
            collections={},
            weights={},
        )

        if not results:
            return ""

        top_results = results[:3]
        knowledge_parts = []
        for item in top_results:
            content = item.get("content", "")
            if content:
                knowledge_parts.append(content)

        return "\n\n".join(knowledge_parts)

    def _format_conversation_history(
        self,
        history: list[Message],
    ) -> str:
        """格式化对话历史为文本。

        Args:
            history: 对话历史消息列表

        Returns:
            格式化后的对话历史字符串
        """
        if not history:
            return "（暂无对话历史）"

        formatted_parts = []
        for msg in history:
            role_label = "医生" if msg.role == "user" else "张主任"
            formatted_parts.append(f"{role_label}：{msg.content}")

        return "\n".join(formatted_parts)

    def _build_user_message(
        self,
        sales_message: str,
        context: dict[str, Any],
    ) -> str:
        """构建用户消息内容。

        将销售消息、客户画像、产品信息和对话历史整合为完整的消息内容。

        Args:
            sales_message: 销售人员发送的消息
            context: 上下文信息字典

        Returns:
            完整的用户消息内容
        """
        parts = ["## 销售人员发言\n", sales_message]

        customer_profile = context.get("customer_profile")
        if customer_profile:
            parts.append("\n## 客户画像\n")
            parts.append(f"- 行业：{customer_profile.industry}")
            parts.append(f"- 职位：{customer_profile.position}")
            if customer_profile.concerns:
                parts.append(f"- 关注点：{', '.join(customer_profile.concerns)}")

        product_info = context.get("product_info")
        if product_info:
            parts.append("\n## 产品信息\n")
            parts.append(f"- 产品名称：{product_info.name}")
            if product_info.description:
                parts.append(f"- 产品描述：{product_info.description}")
            if product_info.core_benefits:
                parts.append(
                    f"- 核心优势：{', '.join(product_info.core_benefits)}"
                )

        conversation_history = context.get("conversation_history", [])
        if conversation_history:
            parts.append("\n## 对话历史\n")
            parts.append(self._format_conversation_history(conversation_history))

        return "".join(parts)

    async def generate_response(
        self,
        sales_message: str,
        context: dict[str, Any],
    ) -> str:
        """生成客户（张主任）的回复。

        结合 RAG 知识库和对话上下文，生成专业、真实的客户回复。
        支持多轮对话记忆，能够基于之前的对话内容进行连贯交流。

        Args:
            sales_message: 销售人员发送的最新消息
            context: 上下文信息字典，包含：
                - customer_profile: 客户画像（CustomerProfile）
                - product_info: 产品信息（ProductInfo）
                - conversation_history: 对话历史（List[Message]）

        Returns:
            客户（张主任）的回复文本

        Example:
            >>> simulator = CustomerSimulator(llm_service, search_engine)
            >>> response = await simulator.generate_response(
            ...     sales_message="我们产品能显著降低血糖",
            ...     context={"customer_profile": profile, "product_info": info}
            ... )
        """
        knowledge_context = await self._retrieve_knowledge(sales_message)

        user_content_parts = [
            "根据以下销售人员的话术，以张主任的身份给出专业、审慎的回应：\n\n",
            self._build_user_message(sales_message, context),
        ]

        if knowledge_context:
            user_content_parts.append("\n\n## 相关医学知识（仅供参考）\n")
            user_content_parts.append(knowledge_context)

        user_content_parts.append(
            "\n\n请以张主任的身份，根据你的专业背景和性格特点，"
            "给出一个符合内分泌科主任医师身份的回复。"
        )

        messages: Sequence[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="".join(user_content_parts)),
        ]

        response = await self.llm_service.ainvoke(messages)
        content = response.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [item if isinstance(item, str) else str(item) for item in content]
            return "".join(text_parts).strip()
        return str(content).strip()
