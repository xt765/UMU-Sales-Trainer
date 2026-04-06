"""配置管理模块。

使用 Pydantic Settings 管理应用配置，支持从环境变量和 .env 文件加载。
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类。

    使用 Pydantic Settings 管理配置，支持从环境变量和 .env 文件加载。
    配置验证在实例化时自动进行。

    Attributes:
        DASHSCOPE_API_KEY: DashScope API 密钥
        DS_API_KEY: DeepSeek API 密钥
        LLM_PROVIDER: 默认 LLM Provider（dashscope/deepseek）
        DATABASE_URL: SQLite 数据库路径
        CHROMA_PERSIST_DIR: Chroma 持久化目录
        EMBEDDING_MODEL: Embedding 模型名称
        LOG_LEVEL: 日志级别
        RATE_LIMIT_PER_MINUTE: 每分钟限流次数

    Example:
        >>> settings = Settings()
        >>> print(settings.LLM_PROVIDER)
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    DASHSCOPE_API_KEY: str = Field(default="", description="DashScope API 密钥")
    DS_API_KEY: str = Field(default="", description="DeepSeek API 密钥")
    LLM_PROVIDER: Literal["dashscope", "deepseek"] = Field(
        default="dashscope",
        description="默认 LLM Provider（dashscope/deepseek）",
    )
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./umu_sales.db",
        description="SQLite 数据库路径",
    )
    CHROMA_PERSIST_DIR: str = Field(
        default="./chroma_db",
        description="Chroma 持久化目录",
    )
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding 模型名称",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="日志级别",
    )
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        description="每分钟限流次数",
        ge=1,
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """验证数据库 URL 格式。

        Args:
            v: 数据库 URL

        Returns:
            验证后的 URL

        Raises:
            ValueError: URL 不以 sqlite+aiosqlite:// 开头时
        """
        if not v.startswith("sqlite+aiosqlite://"):
            raise ValueError("DATABASE_URL must start with 'sqlite+aiosqlite://'")
        return v

    @field_validator("CHROMA_PERSIST_DIR")
    @classmethod
    def validate_chroma_dir(cls, v: str) -> str:
        """验证 Chroma 目录路径。

        Args:
            v: 目录路径

        Returns:
            验证后的路径
        """
        return str(Path(v).expanduser().absolute())

    def get_llm_api_key(self) -> tuple[str, str]:
        """获取当前 LLM Provider 的 API 密钥和 Base URL。

        Returns:
            tuple[str, str]: (api_key, base_url)

        Raises:
            ValueError: 当 API 密钥未设置时
        """
        if self.LLM_PROVIDER == "dashscope":
            if not self.DASHSCOPE_API_KEY:
                raise ValueError("DASHSCOPE_API_KEY is not set")
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return self.DASHSCOPE_API_KEY, base_url
        elif self.LLM_PROVIDER == "deepseek":
            if not self.DS_API_KEY:
                raise ValueError("DS_API_KEY is not set")
            base_url = "https://api.deepseek.com"
            return self.DS_API_KEY, base_url
        raise ValueError(f"Unsupported LLM provider: {self.LLM_PROVIDER}")


settings = Settings()
