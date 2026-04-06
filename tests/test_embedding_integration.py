"""Embedding 集成测试。

使用真实 DashScope API 调用测试，不使用mock。
"""

import os

import pytest


class TestEmbeddingIntegration:
    """Embedding 服务集成测试类。"""

    @pytest.fixture(autouse=True)
    def setup_api_key(self) -> None:
        """设置 API 密钥。

        从环境变量获取 DashScope API 密钥。
        """
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not set")

    def test_encode_single_text_real(self) -> None:
        """测试单个文本编码。

        使用真实 DashScope API 调用。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        result = service.encode(["今天天气很好"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) > 0
        assert all(isinstance(x, float) for x in result[0])

    def test_encode_multiple_texts_real(self) -> None:
        """测试多个文本编码。

        使用真实 DashScope API 调用。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        texts = ["糖尿病的症状", "高血压的治疗", "冠心病的预防"]
        result = service.encode(texts)

        assert len(result) == 3
        for embedding in result:
            assert isinstance(embedding, list)
            assert len(embedding) > 0

    def test_encode_query_real(self) -> None:
        """测试查询文本编码。

        使用真实 DashScope API 调用。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        result = service.encode_query("如何治疗糖尿病")

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    def test_encode_caching_real(self) -> None:
        """测试缓存机制。

        验证相同文本使用缓存，不重复调用 API。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        text = "测试缓存文本"

        result1 = service.encode([text])
        result2 = service.encode([text])

        assert result1 == result2
        assert len(service._cache) == 1

    def test_encode_empty_raises_error(self) -> None:
        """测试空输入错误处理。

        验证空列表抛出 ValueError。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()

        with pytest.raises(ValueError, match="texts cannot be empty"):
            service.encode([])

    def test_encode_query_empty_raises_error(self) -> None:
        """测试空查询错误处理。

        验证空字符串抛出 ValueError。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()

        with pytest.raises(ValueError, match="text cannot be empty"):
            service.encode_query("")

    def test_clear_cache_real(self) -> None:
        """测试缓存清除。

        验证 clear_cache 方法正确清除缓存。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        service.encode(["测试文本"])

        assert len(service._cache) > 0
        service.clear_cache()
        assert len(service._cache) == 0

    def test_embedding_consistency_real(self) -> None:
        """测试嵌入一致性。

        验证相同文本产生相同的嵌入向量。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        text = "DPP-4抑制剂是常用的降糖药物"

        result1 = service.encode([text])
        result2 = service.encode([text])

        assert result1 == result2

    def test_different_texts_produce_different_embeddings_real(self) -> None:
        """测试不同文本产生不同嵌入。

        验证语义不同的文本产生明显不同的嵌入向量。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        text1 = "降糖药物的效果"
        text2 = "高血压药物的副作用"

        result1 = service.encode([text1])[0]
        result2 = service.encode([text2])[0]

        assert result1 != result2

    def test_close_client(self) -> None:
        """测试关闭客户端。

        验证 close 方法正确关闭 HTTP 客户端。
        """
        from umu_sales_trainer.services.embedding import EmbeddingService

        service = EmbeddingService()
        service.encode(["测试"])

        service.close()
        assert service._client is None
