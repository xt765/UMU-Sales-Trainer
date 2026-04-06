"""Chroma 向量数据库服务。

提供向量存储、检索和软删除功能，支持文档的持久化存储。
"""

from typing import Any

import chromadb
from chromadb.config import Settings


class ChromaService:
    """Chroma 向量库服务类。

    封装 Chroma 向量数据库的常用操作，支持 Collection 管理、文档增删改查
    和软删除（通过 is_deleted 标记过滤）。

    Attributes:
        persist_directory: 持久化存储目录路径
        client: Chroma 客户端实例
    """

    def __init__(self, persist_directory: str = "./chroma_data") -> None:
        """初始化 Chroma 服务。

        Args:
            persist_directory: 持久化存储目录，默认为 "./chroma_data"
        """
        self.persist_directory: str = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

    def create_collection(self, name: str) -> chromadb.Collection:
        """创建或获取 Collection。

        Args:
            name: Collection 名称

        Returns:
            chromadb.Collection: 创建的 Collection 实例
        """
        return self.client.get_or_create_collection(name=name)

    def get_collection(self, name: str) -> chromadb.Collection:
        """获取指定名称的 Collection。

        Args:
            name: Collection 名称

        Returns:
            chromadb.Collection: Collection 实例
        """
        return self.client.get_collection(name=name)

    def add_document(
        self,
        collection_name: str,
        document: str,
        metadata: dict[str, Any],
        doc_id: str,
    ) -> None:
        """添加文档到指定 Collection。

        Args:
            collection_name: Collection 名称
            document: 文档内容文本
            metadata: 文档元数据，包含 is_deleted=False 标记
            doc_id: 文档唯一标识
        """
        collection = self.get_collection(collection_name)
        metadata["is_deleted"] = False
        collection.add(documents=[document], metadatas=[metadata], ids=[doc_id])

    def query(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """查询相似文档。

        只返回 is_deleted=False 的未删除文档。

        Args:
            collection_name: Collection 名称
            query_texts: 查询文本列表
            n_results: 返回结果数量，默认为 5

        Returns:
            dict: 包含 documents, distances, metadatas, ids 的查询结果
        """
        collection = self.get_collection(collection_name)
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where={"is_deleted": False},
        )
        return dict(results)

    def soft_delete(self, collection_name: str, document_id: str) -> None:
        """软删除文档。

        通过将 is_deleted 标记设为 True 实现软删除，不会真正删除数据。

        Args:
            collection_name: Collection 名称
            document_id: 要删除的文档 ID
        """
        collection = self.get_collection(collection_name)
        collection.update(
            ids=[document_id],
            metadatas=[{"is_deleted": True}],
        )

    def get_document(
        self, collection_name: str, document_id: str
    ) -> dict[str, Any] | None:
        """获取指定文档。

        Args:
            collection_name: Collection 名称
            document_id: 文档 ID

        Returns:
            包含文档内容和元数据的字典，若不存在或已删除返回 None
        """
        collection = self.get_collection(collection_name)
        result = collection.get(
            ids=[document_id],
            where={"is_deleted": False},
        )
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        ids = result.get("ids")
        if documents and metadatas and ids:
            return {
                "document": documents[0],
                "metadata": metadatas[0],
                "id": ids[0],
            }
        return None

    def list_collections(self) -> list[str]:
        """列出所有 Collection 名称。

        Returns:
            Collection 名称列表
        """
        return [col.name for col in self.client.list_collections()]

    def delete_collection(self, name: str) -> None:
        """删除 Collection（硬删除）。

        Args:
            name: Collection 名称
        """
        self.client.delete_collection(name=name)

    def close(self) -> None:
        """关闭 Chroma 客户端。

        释放资源并关闭持久化连接。
        """
        if hasattr(self.client, "close"):
            self.client.close()
        self.client = None
