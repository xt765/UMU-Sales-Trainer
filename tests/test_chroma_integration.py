"""Chroma 集成测试。

使用真实 Chroma 数据库测试，不使用mock。
"""

import os
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup_env():
    """设置环境变量，从 .env 文件加载。"""
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


class TestChromaIntegration:
    """Chroma 服务集成测试类。"""

    @pytest.fixture
    def temp_dir(self) -> str:
        """创建临时目录。"""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        time.sleep(0.1)
        try:
            for root, dirs, files in os.walk(tmpdir, topdown=False):
                for name in files:
                    filepath = os.path.join(root, name)
                    try:
                        os.unlink(filepath)
                    except Exception:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception:
                        pass
            os.rmdir(tmpdir)
        except Exception:
            pass

    @pytest.fixture
    def chroma_service(self, temp_dir: str) -> "ChromaService":
        """创建 Chroma 服务实例。"""
        from umu_sales_trainer.services.chroma import ChromaService

        service = ChromaService(persist_directory=temp_dir)
        yield service
        try:
            service.close()
        except Exception:
            pass

    def test_create_collection_real(self, chroma_service: "ChromaService") -> None:
        """测试创建 Collection。"""
        collection = chroma_service.create_collection("test_collection")
        assert collection is not None
        assert collection.name == "test_collection"

    def test_add_document_real(self, chroma_service: "ChromaService") -> None:
        """测试添加文档。"""
        chroma_service.create_collection("test_collection")
        result = chroma_service.add_document(
            collection_name="test_collection",
            document="DPP-4抑制剂是一种有效的降糖药物",
            metadata={"source": "test"},
            doc_id="doc-001",
        )
        assert result is None

    def test_query_real(self, chroma_service: "ChromaService") -> None:
        """测试查询功能。"""
        chroma_service.create_collection("test_collection")
        chroma_service.add_document(
            collection_name="test_collection",
            document="DPP-4抑制剂可以有效降低血糖",
            metadata={"source": "product"},
            doc_id="doc-001",
        )
        results = chroma_service.query(
            collection_name="test_collection",
            query_texts=["糖尿病治疗药物"],
            n_results=2,
        )
        assert len(results) >= 1

    def test_soft_delete_real(self, chroma_service: "ChromaService") -> None:
        """测试软删除。"""
        chroma_service.create_collection("test_soft_delete")
        chroma_service.add_document(
            collection_name="test_soft_delete",
            document="将要被删除的文档",
            metadata={"source": "test"},
            doc_id="doc-to-delete",
        )
        result = chroma_service.soft_delete("test_soft_delete", "doc-to-delete")
        assert result is None

    def test_get_document_real(self, chroma_service: "ChromaService") -> None:
        """测试获取单个文档。"""
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
        """测试列出 Collections。"""
        chroma_service.create_collection("list_test_1")
        chroma_service.create_collection("list_test_2")
        collections = chroma_service.list_collections()
        assert isinstance(collections, list)
        assert "list_test_1" in collections
        assert "list_test_2" in collections

    def test_delete_collection_real(self, chroma_service: "ChromaService") -> None:
        """测试删除 Collection。"""
        chroma_service.create_collection("to_be_deleted")
        result = chroma_service.delete_collection("to_be_deleted")
        assert result is None
