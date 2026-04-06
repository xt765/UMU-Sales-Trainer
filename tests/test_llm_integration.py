"""LLM 集成测试。

使用真实 DashScope/DeepSeek API 调用测试，不使用mock。
"""

import os

import pytest


@pytest.fixture(autouse=True)
def setup_env():
    """设置环境变量，从 .env 文件加载。"""
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


class TestLLMIntegration:
    """LLM 服务集成测试类。"""

    def test_create_dashscope_llm_real(self) -> None:
        """测试创建 DashScope LLM 实例。"""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not set")

        from umu_sales_trainer.services.llm import create_llm

        llm = create_llm("dashscope")
        assert llm is not None

    def test_create_deepseek_llm_real(self) -> None:
        """测试创建 DeepSeek LLM 实例。"""
        api_key = os.environ.get("DS_API_KEY", "")
        if not api_key:
            pytest.skip("DS_API_KEY not set")

        from umu_sales_trainer.services.llm import create_llm

        llm = create_llm("deepseek")
        assert llm is not None

    def test_llm_invoke_dashscope_real(self) -> None:
        """测试 DashScope LLM 调用。"""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not set")

        from umu_sales_trainer.services.llm import create_llm
        from langchain_core.messages import HumanMessage

        llm = create_llm("dashscope")
        result = llm.invoke([HumanMessage(content="请用一句话介绍自己")])

        assert hasattr(result, "content")
        assert len(result.content) > 0

    def test_llm_ainvoke_deepseek_real(self) -> None:
        """测试 DeepSeek 异步 LLM 调用。"""
        api_key = os.environ.get("DS_API_KEY", "")
        if not api_key:
            pytest.skip("DS_API_KEY not set")

        from umu_sales_trainer.services.llm import create_llm
        from langchain_core.messages import HumanMessage

        import asyncio

        async def run_test():
            llm = create_llm("deepseek")
            result = await llm.ainvoke([HumanMessage(content="请用一句话介绍自己")])
            return result

        result = asyncio.run(run_test())

        assert hasattr(result, "content")
        assert len(result.content) > 0

    def test_llm_with_system_prompt_real(self) -> None:
        """测试带系统提示的 LLM 调用。"""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not set")

        from umu_sales_trainer.services.llm import create_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = create_llm("dashscope")

        messages = [
            SystemMessage(content="你是一个专业的医学顾问，请用专业术语回答。"),
            HumanMessage(content="糖尿病是什么？"),
        ]

        result = llm.invoke(messages)

        assert hasattr(result, "content")
        content_lower = result.content.lower()
        assert "糖尿病" in content_lower or "血糖" in content_lower
