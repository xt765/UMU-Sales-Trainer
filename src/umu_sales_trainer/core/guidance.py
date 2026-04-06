"""引导生成器模块。

根据未覆盖的语义点生成智能销售引导话术，帮助销售完善信息传递。
支持多种引导策略：直接提问、质疑挑战、澄清请求、补充引导。
结合 RAG 知识库检索生成更精准的引导。
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage

from umu_sales_trainer.services.llm import LLMService

if TYPE_CHECKING:
    from umu_sales_trainer.core.hybrid_search import HybridSearchEngine


class GuidanceGenerator:
    """引导生成器。

    根据未覆盖的语义点列表和销售对话上下文，智能生成引导话术。
    引导策略包括：直接提问、质疑挑战、澄清请求、补充引导。
    支持结合 RAG 知识库检索生成更精准的个性化引导。

    Attributes:
        llm: LLM 服务实例，用于生成自然语言引导
        rag_retriever: 可选的 RAG 检索器，用于增强引导的准确性

    Example:
        >>> generator = GuidanceGenerator(llm_service)
        >>> context = {"product_name": "某药品", "customer_need": "降压"}
        >>> points = [{"point_id": "SP-001", "description": "药品副作用", "importance": 0.8}]
        >>> guidance = generator.generate(points, context)
    """

    def __init__(
        self,
        llm: LLMService,
        rag_retriever: HybridSearchRetriever | None = None,
    ) -> None:
        """初始化引导生成器。

        Args:
            llm: LLM 服务实例
            rag_retriever: 可选的 RAG 检索器，用于知识库增强
        """
        self._llm: LLMService = llm
        self._rag_retriever: HybridSearchRetriever | None = rag_retriever

    def generate(
        self,
        uncovered_points: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str:
        """生成引导话术。

        根据未覆盖的语义点生成智能引导，帮助销售完善信息传递。
        首先检索相关知识库内容，然后综合选择最优引导策略。

        Args:
            uncovered_points: 未覆盖的语义点列表，每项包含 point_id、
                description 和 importance
            context: 对话上下文，包含产品信息、客户需求等

        Returns:
            生成的引导话术字符串，自然融入对话
        """
        if not uncovered_points:
            return "您已经涵盖了所有关键信息，很全面！"

        sorted_points = sorted(
            uncovered_points,
            key=lambda x: x.get("importance", 0.0),
            reverse=True,
        )

        knowledge_context = self._retrieve_knowledge(sorted_points, context)

        primary_point = sorted_points[0]
        strategy = self._select_strategy(primary_point)

        guidance = self._build_guidance(strategy, primary_point, context)

        if len(sorted_points) > 1:
            secondary = self._generate_secondary_guidance(sorted_points[1], context)
            guidance = f"{guidance}\n{secondary}"

        if knowledge_context:
            guidance = self._inject_knowledge(guidance, knowledge_context)

        return guidance

    def _retrieve_knowledge(
        self,
        points: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str:
        """从 RAG 知识库检索相关信息。

        Args:
            points: 语义点列表
            context: 上下文信息

        Returns:
            检索到的相关知识内容，如无检索器则返回空字符串
        """
        if not self._rag_retriever:
            return ""

        query = f"{context.get('product_name', '')} {points[0]['description']}"
        results = self._rag_retriever.retrieve(query, top_k=3)

        if not results:
            return ""

        return "\n".join(f"- {r['content']}" for r in results[:3])

    def _select_strategy(
        self,
        point: dict[str, Any],
    ) -> Literal["direct", "challenge", "clarification", "supplementary"]:
        """根据语义点特征选择最优引导策略。

        Args:
            point: 语义点信息

        Returns:
            选择的引导策略类型
        """
        importance = point.get("importance", 0.5)

        if importance >= 0.8:
            return "challenge"
        elif importance >= 0.6:
            return "direct"
        elif importance >= 0.4:
            return "clarification"
        return "supplementary"

    def _build_guidance(
        self,
        strategy: str,
        point: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        """根据策略构建引导话术。

        Args:
            strategy: 引导策略类型
            point: 语义点信息
            context: 对话上下文

        Returns:
            生成的引导话术
        """
        description = point["description"]
        product_name = context.get("product_name", "产品")
        customer_need = context.get("customer_need", "患者需求")

        if strategy == "direct":
            return self._direct_question(description, product_name)
        elif strategy == "challenge":
            return self._challenge(description, customer_need)
        elif strategy == "clarification":
            return self._clarification(description, product_name)
        return self._supplementary(description, customer_need)

    def _direct_question(self, description: str, product_name: str) -> str:
        """直接提问策略。

        Args:
            description: 语义点描述
            product_name: 产品名称

        Returns:
            直接提问话术
        """
        templates = [
            f"您能详细说说{description}吗？这对客户选择很重要。",
            f"关于{description}，您能再展开讲讲吗？",
            f"能否介绍一下{product_name}在{description}方面的表现？",
        ]
        return random.choice(templates)

    def _challenge(self, description: str, customer_need: str) -> str:
        """质疑挑战策略。

        Args:
            description: 语义点描述
            customer_need: 客户需求

        Returns:
            质疑挑战话术
        """
        templates = [
            f"您提到{description}对{customer_need}影响不大，您确定吗？",
            f"客户通常很关注{description}这一点，您怎么看？",
            f"关于{description}，实际上很多患者都很在意，您能详细说说吗？",
        ]
        return random.choice(templates)

    def _clarification(self, description: str, product_name: str) -> str:
        """澄清请求策略。

        Args:
            description: 语义点描述
            product_name: 产品名称

        Returns:
            澄清请求话术
        """
        templates = [
            f"您提到的{description}能举个例子说明一下吗？",
            f"关于{product_name}在{description}方面的优势，能具体说说吗？",
            f"您能详细描述一下{description}的场景吗？",
        ]
        return random.choice(templates)

    def _supplementary(self, description: str, customer_need: str) -> str:
        """补充引导策略。

        Args:
            description: 语义点描述
            customer_need: 客户需求

        Returns:
            补充引导话术
        """
        templates = [
            f"除了{description}，还想了解一下对{customer_need}的其他考虑。",
            f"关于{description}这一点，您还有什么想补充的吗？",
            f"除了刚才说的，您是否还想了解{description}相关的内容？",
        ]
        return random.choice(templates)

    def _generate_secondary_guidance(
        self,
        point: dict[str, Any],
        _context: dict[str, Any],
    ) -> str:
        """生成次要引导话术。

        对次要未覆盖点生成较简短的补充引导。

        Args:
            point: 次要语义点
            _context: 对话上下文（未使用，保留参数兼容性）

        Returns:
            补充引导话术
        """
        return f"另外，关于{point['description']}也欢迎您补充说明。"

    def _inject_knowledge(self, guidance: str, knowledge: str) -> str:
        """将知识库内容融入引导话术。

        使用 LLM 将检索到的知识自然融入引导话术，使其更有说服力。

        Args:
            guidance: 原始引导话术
            knowledge: 检索到的知识内容

        Returns:
            融入知识后的引导话术
        """
        prompt = (
            f"作为销售教练，请将以下知识内容自然地融入到引导话术中。\n"
            f"引导话术：{guidance}\n\n"
            f"参考知识：\n{knowledge}\n\n"
            f"请生成融合后的引导话术，保持自然，不显得生硬："
        )

        messages: list[BaseMessage] = [HumanMessage(content=prompt)]
        response = self._llm.invoke(messages)

        content: str | list[str | dict[Any, Any]] = response.content
        if isinstance(content, list):
            first = content[0]
            content = str(first) if first else ""
        return str(content)


class HybridSearchRetriever:
    """混合搜索检索器。

    封装 HybridSearchEngine，提供简化的检索接口。

    Attributes:
        search_engine: 混合搜索引擎实例
        collections: 可搜索的 Collection 映射
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        collections: dict[str, Any],
    ) -> None:
        """初始化检索器。

        Args:
            search_engine: 混合搜索引擎实例
            collections: Collection 名称到实例的映射
        """
        self._search_engine: HybridSearchEngine = search_engine
        self._collections: dict[str, Any] = collections

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """检索相关文档。

        Args:
            query: 检索查询
            top_k: 返回结果数量

        Returns:
            相关文档列表
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results: list[dict[str, Any]] = loop.run_until_complete(
            self._search_engine.search(
                query=query,
                collections=self._collections,
                weights={name: 1.0 for name in self._collections},
            )
        )
        return results[:top_k]
