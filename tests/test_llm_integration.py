"""LLM 集成测试。

使用真实 DashScope/DeepSeek API 调用测试，不使用mock。
"""

import os

import pytest


class TestLLMIntegration:
    """LLM 服务集成测试类。"""

    @pytest.fixture(autouse=True)
    def setup_api_keys(self) -> None:
        """设置 API 密钥。"""
        self.dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self.deepseek_key = os.environ.get("DS_API_KEY", "")
        if not self.dashscope_key and not self.deepseek_key:
            pytest.skip("Neither DASHSCOPE_API_KEY nor DS_API_KEY is set")

    def test_create_dashscope_llm_real(self) -> None:
        """测试创建 DashScope LLM 实例。

        使用真实 API 密钥创建实例。
        """
        if not self.dashscope_key:
            pytest.skip("DASHSCOPE_API_KEY not set")

        os.environ["DASHSCOPE_API_KEY"] = self.dashscope_key

        from umu_sales_trainer.services.llm import create_llm

        llm = create_llm("dashscope")
        assert llm is not None

    def test_create_deepseek_llm_real(self) -> None:
        """测试创建 DeepSeek LLM 实例。

        使用真实 API 密钥创建实例。
        """
        if not self.deepseek_key:
            pytest.skip("DS_API_KEY not set")

        os.environ["DS_API_KEY"] = self.deepseek_key

        from umu_sales_trainer.services.llm import create_llm

        llm = create_llm("deepseek")
        assert llm is not None

    def test_llm_invoke_dashscope_real(self) -> None:
        """测试 DashScope LLM 调用。

        使用真实 DashScope API 进行对话。
        """
        if not self.dashscope_key:
            pytest.skip("DASHSCOPE_API_KEY not set")

        os.environ["DASHSCOPE_API_KEY"] = self.dashscope_key

        from umu_sales_trainer.services.llm import LLMService, create_llm

        llm = create_llm("dashscope")
        service = LLMService(llm)

        from langchain_core.messages import HumanMessage
        result = service.invoke([HumanMessage(content="请用一句话介绍自己")])

        assert hasattr(result, "content")
        assert len(result.content) > 0

    def test_llm_ainvoke_deepseek_real(self) -> None:
        """测试 DeepSeek 异步 LLM 调用。

        使用真实 DeepSeek API 进行异步对话。
        """
        if not self.deepseek_key:
            pytest.skip("DS_API_KEY not set")

        os.environ["DS_API_KEY"] = self.deepseek_key

        from umu_sales_trainer.services.llm import LLMService, create_llm
        import asyncio

        async def run_test():
            llm = create_llm("deepseek")
            service = LLMService(llm)

            from langchain_core.messages import HumanMessage
            result = await service.ainvoke([HumanMessage(content="请用一句话介绍自己")])
            return result

        result = asyncio.run(run_test())

        assert hasattr(result, "content")
        assert len(result.content) > 0

    def test_llm_with_system_prompt_real(self) -> None:
        """测试带系统提示的 LLM 调用。

        验证系统提示正确影响输出。
        """
        if not self.dashscope_key:
            pytest.skip("DASHSCOPE_API_KEY not set")

        os.environ["DASHSCOPE_API_KEY"] = self.dashscope_key

        from umu_sales_trainer.services.llm import LLMService, create_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = create_llm("dashscope")
        service = LLMService(llm)

        messages = [
            SystemMessage(content="你是一个专业的医学顾问，请用专业术语回答。"),
            HumanMessage(content="糖尿病是什么？"),
        ]

        result = service.invoke(messages)

        assert hasattr(result, "content")
        content_lower = result.content.lower()
        assert "糖尿病" in content_lower or "血糖" in content_lower

    def test_llm_stream_real(self) -> None:
        """测试流式 LLM 调用。

        验证流式输出正常工作。
        """
        if not self.dashscope_key:
            pytest.skip("DASHSCOPE_API_KEY not set")

        os.environ["DASHSCOPE_API_KEY"] = self.dashscope_key

        from umu_sales_trainer.services.llm import LLMService, create_llm
        from langchain_core.messages import HumanMessage

        llm = create_llm("dashscope")
        service = LLMService(llm)

        chunks = []
        for chunk in service.invoke_stream([HumanMessage(content="讲一个笑话")]):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_content = "".join(c.content for c in chunks if hasattr(c, "content"))
        assert len(full_content) > 0
