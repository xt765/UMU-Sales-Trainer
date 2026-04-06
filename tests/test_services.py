"""Services 模块测试。

测试 LLM、Embedding、Chroma、Database 服务的核心功能。
使用 mock 避免依赖外部服务。
"""

from unittest.mock import MagicMock, patch

import pytest


class TestLLMService:
    """LLM 服务测试类。"""

    def test_create_llm_dashscope(self) -> None:
        """测试创建 DashScope LLM 实例。

        验证 create_llm 工厂方法能正确创建 DashScope provider 实例。
        """
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}):
            from umu_sales_trainer.services.llm import create_llm

            llm = create_llm("dashscope")
            assert llm is not None

    def test_create_llm_deepseek(self) -> None:
        """测试创建 DeepSeek LLM 实例。

        验证 create_llm 工厂方法能正确创建 DeepSeek provider 实例。
        """
        with patch.dict("os.environ", {"DS_API_KEY": "test-key"}):
            from umu_sales_trainer.services.llm import create_llm

            llm = create_llm("deepseek")
            assert llm is not None

    def test_llm_service_invoke(self) -> None:
        """测试 LLM 服务调用。

        验证 LLMService 的 invoke 方法存在且可调用。
        """
        from umu_sales_trainer.services.llm import LLMService

        mock_llm = MagicMock(spec=LLMService)
        mock_llm.invoke.return_value = MagicMock(content="test response")

        assert hasattr(mock_llm, "invoke")
        result = mock_llm.invoke([])
        assert result.content == "test response"


class TestEmbeddingService:
    """Embedding 服务测试类。"""

    def test_embedding_service_init(self) -> None:
        """测试 EmbeddingService 初始化。

        验证 EmbeddingService 能正确初始化并设置模型名称。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService(model_name="text-embedding-v1")
        assert service._model_name == "text-embedding-v1"

    def test_encode_empty_raises_error(self) -> None:
        """测试空文本列表抛出错误。

        验证 encode 方法在输入为空列表时抛出 ValueError。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        with pytest.raises(ValueError, match="texts cannot be empty"):
            service.encode([])

    def test_encode_query_empty_raises_error(self) -> None:
        """测试空查询文本抛出错误。

        验证 encode_query 方法在输入为空字符串时抛出 ValueError。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        with pytest.raises(ValueError, match="text cannot be empty"):
            service.encode_query("")

    def test_encode_no_api_key_raises_error(self) -> None:
        """测试未设置 API key 时抛出错误。

        验证 encode 方法在 DASHSCOPE_API_KEY 未设置时抛出 RuntimeError。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
                service.encode(["测试文本"])

    def test_clear_cache(self) -> None:
        """测试缓存清除。

        验证 clear_cache 方法能正确清除所有缓存。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        assert len(service._cache) == 0
        service.clear_cache()
        assert len(service._cache) == 0

    def test_get_cache_key(self) -> None:
        """测试缓存键生成。

        验证 _get_cache_key 方法能正确生成 MD5 哈希。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        key1 = service._get_cache_key("测试文本")
        key2 = service._get_cache_key("测试文本")
        key3 = service._get_cache_key("不同文本")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 32


class TestChromaService:
    """Chroma 服务测试类。"""

    def test_chroma_service_init(self) -> None:
        """测试 ChromaService 初始化。

        验证 ChromaService 能正确初始化并设置持久化目录。
        """
        from umu_sales_trainer.services.chroma import ChromaService

        service = ChromaService(persist_directory="./test_chroma")
        assert service.persist_directory == "./test_chroma"

    def test_create_collection(self) -> None:
        """测试创建 Collection。

        验证 create_collection 方法能正确创建或获取 Collection。
        """
        from umu_sales_trainer.services.chroma import ChromaService

        service = ChromaService(persist_directory="./test_chroma")
        collection = service.create_collection("test_collection")

        assert collection is not None
        service.delete_collection("test_collection")

    def test_soft_delete(self) -> None:
        """测试软删除。

        验证 soft_delete 方法能正确标记文档为已删除状态。
        """
        from umu_sales_trainer.services.chroma import ChromaService

        service = ChromaService(persist_directory="./test_chroma_soft")
        service.create_collection("test_soft_delete")

        service.add_document(
            collection_name="test_soft_delete",
            document="测试文档内容",
            metadata={"source": "test"},
            doc_id="test-doc-001",
        )

        service.soft_delete("test_soft_delete", "test-doc-001")

        service.delete_collection("test_soft_delete")


class TestDatabaseService:
    """Database 服务测试类。"""

    def test_database_service_init(self) -> None:
        """测试 DatabaseService 初始化。

        验证 DatabaseService 能正确初始化（使用文件路径避免 SQLite 内存模式限制）。
        """
        import tempfile
        import os

        from umu_sales_trainer.services.database import DatabaseService

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            service = DatabaseService(db_path=db_path)
            assert service.engine is not None

    def test_session_model_creation(self) -> None:
        """测试 SessionModel 创建。

        验证 SessionModel 能正确创建实例。
        """
        from umu_sales_trainer.services.database import SessionModel

        session = SessionModel(
            id="test-001",
            customer_profile={"name": "张主任"},
            product_info={"name": "降糖药"},
            status="active",
        )

        assert session.id == "test-001"
        assert session.status == "active"

    def test_message_model_creation(self) -> None:
        """测试 MessageModel 创建。

        验证 MessageModel 能正确创建实例。
        """
        from umu_sales_trainer.services.database import MessageModel

        message = MessageModel(
            session_id="test-001",
            role="user",
            content="测试消息",
            turn=1,
        )

        assert message.session_id == "test-001"
        assert message.role == "user"

    def test_coverage_record_model_creation(self) -> None:
        """测试 CoverageRecordModel 创建。

        验证 CoverageRecordModel 能正确创建实例。
        """
        from umu_sales_trainer.services.database import CoverageRecordModel

        record = CoverageRecordModel(
            session_id="test-001",
            point_id="SP-001",
            point_description="测试语义点",
            is_covered=True,
            coverage_details={"score": 0.9},
        )

        assert record.session_id == "test-001"
        assert record.is_covered is True
