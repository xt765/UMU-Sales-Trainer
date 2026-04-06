"""混合搜索引擎模块。

实现基于 RRF（Reciprocal Rank Fusion）和动态加权融合的混合搜索功能，
支持并行查询多个 Chroma Collections 并融合结果。
"""

from typing import Any

from langchain_chroma import Chroma

from umu_sales_trainer.services.embedding import EmbeddingService


class HybridSearchEngine:
    """混合搜索引擎。

    结合 RRF（倒数排名融合）算法和动态加权策略，对多个数据源的搜索结果
    进行融合，以提供更准确、更全面的搜索体验。

    Attributes:
        embedding_service: 向量嵌入服务，用于将文本转换为向量表示
        _rrf_k: RRF 算法中的常数参数，默认为 60
    """

    def __init__(self, embedding_service: EmbeddingService) -> None:
        """初始化混合搜索引擎。

        Args:
            embedding_service: 向量嵌入服务实例，用于生成查询向量
        """
        self.embedding_service = embedding_service
        self._rrf_k = 60

    async def search(
        self,
        query: str,
        collections: dict[str, Chroma],
        weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """执行混合搜索。

        并行查询多个 Chroma Collections，然后使用 RRF 算法和动态加权
        融合搜索结果。

        Args:
            query: 搜索查询文本
            collections: Collection 名称到 Chroma 实例的映射字典
            weights: 各 Collection 的权重系数，用于动态加权

        Returns:
            融合后的搜索结果列表，每项包含文档内容、元数据、来源Collection
            和融合分数
        """
        query_embedding = await self.embedding_service.embed_query(query)
        results_per_collection: list[list[dict[str, Any]]] = []

        for name in collections:
            collection = collections[name]
            docs = collection.search(
                query_embedding=query_embedding,
                n_results=10,
            )
            results = self._format_results(docs, name)
            results_per_collection.append(results)

        fused = self._rrf_fusion(results_per_collection, k=self._rrf_k)
        return self._dynamic_weight(fused)

    def _rrf_fusion(
        self,
        results: list[list[dict[str, Any]]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """应用 RRF（倒数排名融合）算法。

        RRF 通过公式 Σ 1/(k + rank) 计算每个文档的融合分数，
        其中 rank 是该文档在各个结果列表中的排名位置（从1开始）。

        Args:
            results: 各 Collection 的搜索结果列表
            k: RRF 算法常数参数，通常设为 60
            用于平衡低排名和高排名文档的影响力

        Returns:
            按 RRF 分数排序的融合结果
        """
        scores: dict[str, dict[str, Any]] = {}

        for result_list in results:
            for rank, item in enumerate(result_list, start=1):
                doc_id = item["id"]
                rrf_score = 1 / (k + rank)

                if doc_id not in scores:
                    scores[doc_id] = item.copy()
                    scores[doc_id]["rrf_score"] = 0.0

                scores[doc_id]["rrf_score"] += rrf_score

        return sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    def _dynamic_weight(
        self,
        results: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """应用动态加权策略调整结果分数。

        根据上下文信息（如用户偏好、搜索历史等）动态调整各结果的权重，
        使搜索结果更符合当前搜索场景的需求。

        Args:
            results: 经过 RRF 融合后的结果列表
            context: 上下文信息字典，包含用户ID、会话信息等；
                目前支持 source_weight 字段用于调整特定来源的权重

        Returns:
            调整后的搜索结果列表
        """
        if context is None:
            context = {}

        source_weight = context.get("source_weight", {})

        for item in results:
            source = item.get("collection", "default")
            weight = source_weight.get(source, 1.0)
            item["final_score"] = item["rrf_score"] * weight

        return sorted(results, key=lambda x: x["final_score"], reverse=True)

    def _format_results(
        self,
        docs: dict[str, Any],
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """格式化搜索结果。

        Args:
            docs: Chroma search 返回的原始文档字典
            collection_name: Collection 名称

        Returns:
            格式化后的结果列表
        """
        formatted = []
        for i, (doc, metadata) in enumerate(zip(docs["documents"], docs["metadatas"])):
            formatted.append({
                "id": f"{collection_name}_{i}",
                "content": doc,
                "metadata": metadata,
                "collection": collection_name,
            })
        return formatted
