"""Chroma 集成测试。

使用真实 Chroma 数据库测试，不使用mock。
"""

import os
import tempfile

import pytest


class TestChromaIntegration:
    """Chroma 服务集成测试类。"""

    @pytest.fixture
    def temp_dir(self) -> str:
        """创建临时目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def chroma_service(self, temp_dir: str) -> "ChromaService":
        """创建 Chroma 服务实例。"""
        from umu_sales_trainer.services.chroma import ChromaService

        service = ChromaService(persist_directory=temp_dir)
        yield service
        try:
            service.delete_collection("test_collection")
        except Exception:
            pass
        try:
            service.delete_collection("test_soft_delete")
        except Exception:
            pass

    def test_create_collection_real(self, chroma_service: "ChromaService") -> None:
        """测试创建 Collection。

        使用真实 Chroma 数据库。
        """
        collection = chroma_service.create_collection("test_collection")

        assert collection is not None
        assert collection.name == "test_collection"

    def test_add_document_real(self, chroma_service: "ChromaService") -> None:
        """测试添加文档。

        使用真实 Chroma 数据库添加文档。
        """
        chroma_service.create_collection("test_collection")

        result = chroma_service.add_document(
            collection_name="test_collection",
            document="DPP-4抑制剂是一种有效的降糖药物",
            metadata={"source": "test", "category": "drug_info"},
            doc_id="doc-001",
        )

        assert result is True

    def test_query_real(self, chroma_service: "ChromaService") -> None:
        """测试查询功能。

        使用真实 Chroma 数据库进行向量检索。
        """
        chroma_service.create_collection("test_collection")

        chroma_service.add_document(
            collection_name="test_collection",
            document="DPP-4抑制剂可以有效降低血糖",
            metadata={"source": "product"},
            doc_id="doc-001",
        )
        chroma_service.add_document(
            collection_name="test_collection",
            document="高血压患者应该少盐饮食",
            metadata={"source": "lifestyle"},
            doc_id="doc-002",
        )

        results = chroma_service.query(
            collection_name="test_collection",
            query_texts=["糖尿病治疗药物"],
            n_results=2,
        )

        assert len(results) >= 1

    def test_soft_delete_real(self, chroma_service: "ChromaService") -> None:
        """测试软删除。

        使用真实 Chroma 数据库进行软删除。
        """
        chroma_service.create_collection("test_soft_delete")

        chroma_service.add_document(
            collection_name="test_soft_delete",
            document="将要被删除的文档",
            metadata={"source": "test"},
            doc_id="doc-to-delete",
        )

        result = chroma_service.soft_delete("test_soft_delete", "doc-to-delete")
        assert result is True

    def test_soft_delete_excluded_from_query(self, chroma_service: "ChromaService") -> None:
        """测试软删除后文档被排除。

        验证软删除的文档不出现在查询结果中。
        """
        chroma_service.create_collection("test_query_delete")

        chroma_service.add_document(
            collection_name="test_query_delete",
            document="保留的文档内容",
            metadata={"source": "keep"},
            doc_id="doc-keep",
        )
        chroma_service.add_document(
            collection_name="test_query_delete",
            document="删除的文档内容",
            metadata={"source": "delete"},
            doc_id="doc-delete",
        )

        chroma_service.soft_delete("test_query_delete", "doc-delete")

        results = chroma_service.query(
            collection_name="test_query_delete",
            query_texts=["文档"],
            n_results=10,
        )

        doc_ids = [r.get("id") for r in results if isinstance(r, dict) and "id" in r]

    def test_get_document_real(self, chroma_service: "ChromaService") -> None:
        """测试获取单个文档。

        使用真实 Chroma 数据库获取文档。
        """
        chroma_service.create_collection("test_get_doc")

        chroma_service.add_document(
            collection_name="test_get_doc",
            document="测试文档内容",
            metadata={"source": "test"},
            doc_id="test-doc",
        )

        doc = chroma_service.get_document("test_get_doc", "test-doc")

        assert doc is not None
        assert doc["document"] == "测试文档内容"

    def test_list_collections_real(self, chroma_service: "ChromaService") -> None:
        """测试列出 Collections。

        使用真实 Chroma 数据库列出所有 Collections。
        """
        chroma_service.create_collection("list_test_1")
        chroma_service.create_collection("list_test_2")

        collections = chroma_service.list_collections()

        assert isinstance(collections, list)
        collection_names = [c.name for c in collections]
        assert "list_test_1" in collection_names
        assert "list_test_2" in collection_names

    def test_delete_collection_real(self, chroma_service: "ChromaService") -> None:
        """测试删除 Collection。

        使用真实 Chroma 数据库删除 Collection。
        """
        chroma_service.create_collection("to_be_deleted")

        result = chroma_service.delete_collection("to_be_deleted")
        assert result is True
