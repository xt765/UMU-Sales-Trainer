"""LLM 服务模块。

支持 DashScope（qwen）和 DeepSeek 两个 Provider 的 LLM 调用。
使用 langchain-core 的 OpenAI 兼容接口实现。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

from pydantic import SecretStr


class LLMServicesError(Exception):
    """LLM 服务异常。"""
    pass


class LLMService:
    """LLM 服务类。

    提供统一的 LLM 调用接口，支持 DashScope 和 DeepSeek 两个 Provider。
    通过工厂方法 create_llm() 创建实例。

    Attributes:
        model: 底层的 ChatOpenAI 模型实例
        provider: 当前使用的 Provider 名称

    Example:
        >>> llm = create_llm("dashscope")
        >>> response = llm.invoke([HumanMessage(content="你好")])
    """

    model: BaseChatModel
    provider: str

    def __init__(self, model: BaseChatModel, provider: str) -> None:
        """初始化 LLM 服务。

        Args:
            model: ChatOpenAI 模型实例
            provider: Provider 名称
        """
        self.model = model
        self.provider = provider

    def invoke(self, messages: Sequence[BaseMessage]) -> BaseMessage:
        """同步调用 LLM。

        Args:
            messages: 消息列表

        Returns:
            AI 回复消息
        """
        return self.model.invoke(messages)

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> BaseMessage:
        """异步调用 LLM。

        Args:
            messages: 消息列表

        Returns:
            AI 回复消息
        """
        return await self.model.ainvoke(messages)

    def bind_tools(self, tools: list[BaseTool]) -> Runnable[Any, Any]:
        """绑定工具函数。

        Args:
            tools: 工具函数列表

        Returns:
            绑定工具后的 Runnable
        """
        return self.model.bind_tools(tools)


def create_llm(provider: Literal["dashscope", "deepseek"] = "dashscope") -> LLMService:
    """工厂方法：创建 LLM 实例。

    根据指定的 Provider 创建对应的 LLM 实例。从环境变量读取 API 密钥。

    Args:
        provider: LLM Provider 名称，"dashscope"（阿里云qwen）或"deepseek"

    Returns:
        LLMService 实例

    Raises:
        LLMServicesError: 当 API 密钥缺失或 Provider 不支持时

    Example:
        >>> llm = create_llm("dashscope")
        >>> llm = create_llm("deepseek")
    """
    if provider == "dashscope":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        if not api_key:
            raise LLMServicesError("DASHSCOPE_API_KEY environment variable not set")
        model: BaseChatModel = ChatOpenAI(
            model="qwen-plus",
            api_key=SecretStr(api_key),
            base_url=base_url,
        )
    elif provider == "deepseek":
        api_key = os.environ.get("DS_API_KEY", "")
        base_url = os.environ.get("DS_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            raise LLMServicesError("DS_API_KEY environment variable not set")
        model = ChatOpenAI(
            model="deepseek-chat",
            api_key=SecretStr(api_key),
            base_url=base_url,
        )
    else:
        raise LLMServicesError(f"Unsupported provider: {provider}")
    return LLMService(model=model, provider=provider)
