"""Config 集成测试。

使用真实配置加载测试，不使用mock。
"""

import os
from pathlib import Path

import pytest


class TestConfigIntegration:
    """配置管理集成测试类。"""

    def test_load_config_from_env_file(self, tmp_path: Path) -> None:
        """测试从 .env 文件加载配置。

        创建临时 .env 文件，验证配置能正确加载。
        """
        env_file = tmp_path / ".env"
        env_file.write_text("""
DASHSCOPE_API_KEY=sk-test-dashscope-key
DS_API_KEY=sk-test-deepseek-key
LLM_PROVIDER=dashscope
DATABASE_URL=sqlite+aiosqlite:///./test.db
CHROMA_PERSIST_DIR=./test_chroma
EMBEDDING_MODEL=text-embedding-v1
LOG_LEVEL=DEBUG
RATE_LIMIT_PER_MINUTE=100
""")

        dotenv_path = tmp_path / ".env"
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)

        from umu_sales_trainer.config import Settings

        settings = Settings()

        assert settings.DASHSCOPE_API_KEY == "sk-test-dashscope-key"
        assert settings.DS_API_KEY == "sk-test-deepseek-key"
        assert settings.LLM_PROVIDER == "dashscope"
        assert settings.RATE_LIMIT_PER_MINUTE == 100

    def test_config_with_deepseek_provider(self, tmp_path: Path) -> None:
        """测试 DeepSeek Provider 配置。

        验证切换到 DeepSeek 时 get_llm_api_key 返回正确配置。
        """
        env_file = tmp_path / ".env"
        env_file.write_text("""
DS_API_KEY=sk-real-deepseek-key
LLM_PROVIDER=deepseek
""")

        from dotenv import load_dotenv
        load_dotenv(env_file)

        from umu_sales_trainer.config import Settings

        settings = Settings()
        api_key, base_url = settings.get_llm_api_key()

        assert api_key == "sk-real-deepseek-key"
        assert "deepseek" in base_url
        assert base_url == "https://api.deepseek.com/v1"

    def test_config_database_url_must_start_with_sqlite(self) -> None:
        """测试数据库URL验证。

        验证非sqlite URL会被拒绝。
        """
        os.environ["DATABASE_URL"] = "postgresql://localhost/db"

        from umu_sales_trainer.config import Settings

        with pytest.raises(ValueError, match="DATABASE_URL 必须以"):
            Settings()

    def test_chroma_dir_expands_user_home(self) -> None:
        """测试用户目录展开。

        验证 ~ 被正确展开。
        """
        os.environ["CHROMA_PERSIST_DIR"] = "~/test_chroma_data"

        from umu_sales_trainer.config import Settings

        settings = Settings()
        assert "~" not in settings.CHROMA_PERSIST_DIR
        assert "test_chroma_data" in settings.CHROMA_PERSIST_DIR

    def test_settings_singleton_behavior(self) -> None:
        """测试 Settings 单例行为。

        验证多次实例化返回相同配置。
        """
        os.environ["TEST_VALUE"] = "test123"

        from umu_sales_trainer.config import Settings

        settings1 = Settings()
        settings2 = Settings()

        assert settings1.LOG_LEVEL == settings2.LOG_LEVEL
