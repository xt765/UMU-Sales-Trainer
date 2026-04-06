# 测试文档

本文档介绍 UMU Sales Trainer 的测试方法和覆盖率报告。

## 测试方法

### 单元测试

使用 pytest + pytest-asyncio 进行单元测试，主要测试核心业务逻辑：

```bash
# 运行所有单元测试
uv run pytest tests/test_analyzer.py tests/test_evaluator.py tests/test_guidance.py -v
```

### 集成测试

使用真实 API 和数据库进行集成测试：

```bash
# 运行集成测试
uv run pytest tests/test_embedding_integration.py tests/test_llm_integration.py -v
```

### 完整测试套件

```bash
# 运行所有测试
uv run pytest tests/ -v

# 运行并显示覆盖率
uv run pytest tests/ --cov=src --cov-report=term-missing
```

## 测试覆盖率

当前测试覆盖率：**86.78%**

| 模块 | 覆盖率 |
|------|--------|
| config.py | 60% |
| analyzer.py | 84% |
| evaluator.py | 94% |
| guidance.py | 85% |
| workflow.py | 98% |
| chroma.py | 93% |
| database.py | 92% |
| embedding.py | 91% |
| llm.py | 82% |

## 运行特定测试

```bash
# 运行特定文件的测试
uv run pytest tests/test_workflow.py -v

# 运行特定测试类
uv run pytest tests/test_evaluator.py::TestEvaluator -v

# 运行特定测试用例
uv run pytest tests/test_evaluator.py::TestEvaluator::test_evaluate_keyword_detection -v
```

## 编写新测试

### 示例：测试 SalesAnalyzer

```python
import pytest
from unittest.mock import MagicMock
from umu_sales_trainer.core.analyzer import SalesAnalyzer


class TestSalesAnalyzer:
    """SalesAnalyzer 测试类。"""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """创建 Mock LLM 服务。"""
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(content='{"key_points": ["产品介绍"]}')
        return mock

    @pytest.fixture
    def analyzer(self, mock_llm: MagicMock) -> SalesAnalyzer:
        """创建 SalesAnalyzer 实例。"""
        return SalesAnalyzer(llm=mock_llm)

    def test_analyze_basic(self, analyzer: SalesAnalyzer) -> None:
        """测试基本分析功能。"""
        result = analyzer.analyze(
            sales_message="这个产品降糖效果很好",
            context={"product_name": "降糖药"}
        )
        assert "key_points" in result
```

### 示例：集成测试

```python
import pytest
from umu_sales_trainer.services.embedding import EmbeddingService


class TestEmbeddingIntegration:
    """Embedding 集成测试类。"""

    def test_encode_real_api(self) -> None:
        """测试真实的 Embedding API 调用。"""
        service = EmbeddingService()
        result = service.encode(["测试文本"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(x, float) for x in result[0])
```

## Mock 与集成测试的选择

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 核心业务逻辑 | Mock | 隔离外部依赖，测试稳定 |
| API 调用 | 集成测试 | 验证真实行为 |
| 数据库操作 | 集成测试 | 验证 SQL 正确性 |
| 向量检索 | 集成测试 | 验证相似度计算 |

## CI/CD

在 CI/CD 流程中运行测试：

```yaml
# .gitlab-ci.yml 示例
test:
  script:
    - uv sync
    - uv run pytest tests/ --cov=src --cov-fail-under=85
```
