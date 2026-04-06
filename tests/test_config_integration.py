"""Config 集成测试。

使用真实配置加载测试，不使用mock。
"""

import os

import pytest


@pytest.fixture(autouse=True)
def clean_env():
    """清理环境变量。"""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


class TestConfigIntegration:
    """配置管理集成测试类。"""

    def test_chroma_dir_expands_user_home(self) -> None:
        """测试用户目录展开。"""
        os.environ["CHROMA_PERSIST_DIR"] = "~/test_chroma_data"
        os.environ["DASHSCOPE_API_KEY"] = "test"
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"

        from umu_sales_trainer.config import Settings

        settings = Settings()
        assert "~" not in settings.CHROMA_PERSIST_DIR
        assert "test_chroma_data" in settings.CHROMA_PERSIST_DIR

    def test_settings_default_values(self) -> None:
        """测试默认配置值。"""
        os.environ["DASHSCOPE_API_KEY"] = "test"
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"

        from umu_sales_trainer.config import Settings

        settings = Settings()
        assert settings.LLM_PROVIDER == "dashscope"
        assert settings.RATE_LIMIT_PER_MINUTE == 60
        assert "sentence-transformers" in settings.EMBEDDING_MODEL

    def test_get_llm_api_key_dashscope(self) -> None:
        """测试获取 DashScope API 密钥。"""
        os.environ["DASHSCOPE_API_KEY"] = "sk-dashscope-key"
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"

        from umu_sales_trainer.config import Settings

        settings = Settings()
        api_key, base_url = settings.get_llm_api_key()

        assert api_key == "sk-dashscope-key"
        assert "dashscope" in base_url

    def test_get_llm_api_key_deepseek(self) -> None:
        """测试获取 DeepSeek API 密钥。"""
        os.environ["DS_API_KEY"] = "sk-deepseek-key"
        os.environ["LLM_PROVIDER"] = "deepseek"
        os.environ["DASHSCOPE_API_KEY"] = "test"
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"

        from umu_sales_trainer.config import Settings

        settings = Settings()
        api_key, base_url = settings.get_llm_api_key()

        assert api_key == "sk-deepseek-key"
        assert "deepseek" in base_url
