# 测试文档

本文档详细介绍 UMU Sales Trainer 的测试策略、测试金字塔和覆盖率报告。

---

## 测试原则

> **⚠️ 硬性要求：所有集成测试必须使用真实 API 调用，禁止任何 mock**
>
> 这是不可妥协的底线。使用真实 API 调用可以：
> - 验证与外部服务的真实交互
> - 提前发现 API 变更导致的问题
> - 确保测试环境与生产环境一致

---

## 测试金字塔

本项目采用**测试金字塔**策略，从底到顶依次为：

```
                    ▲
                   /│ \
                  / │  \
                 /  │   \         ← E2E 测试 (端到端)
                /───┼────\
               /    │     \       ← Integration 测试 (集成)
              /     │      \
             /──────┼───────\     ← Unit 测试 (单元)
            /       │        \
           ▼────────▼─────────▼
        快速、隔离                慢速、真实
```

| 层级 | 数量 | 速度 | 隔离性 | 覆盖率目标 |
|------|------|------|--------|------------|
| **单元测试** | ~70 | < 1s | 高 | 80%+ |
| **集成测试** | ~30 | < 10s | 中 | 60%+ |
| **E2E 测试** | ~5 | < 30s | 低 | 核心流程 |

### 单元测试 (Unit Tests)

- **目标**：测试核心业务逻辑，与外部依赖隔离
- **工具**：pytest + pytest-asyncio
- **Mock**：可使用 mock 隔离非核心依赖
- **示例**：测试 `SemanticEvaluator.evaluate()` 的三层检测逻辑

### 集成测试 (Integration Tests)

- **目标**：验证与外部服务（API、数据库、向量库）的真实交互
- **工具**：pytest + pytest-asyncio
- **Mock**：❌ 禁止使用任何 mock
- **示例**：
  - `test_embedding_integration.py` - 真实 DashScope API 调用
  - `test_llm_integration.py` - 真实 DashScope/DeepSeek API 调用
  - `test_database_integration.py` - 真实 SQLite 操作
  - `test_chroma_integration.py` - 真实 Chroma 向量检索

### E2E 测试 (End-to-End Tests)

- **目标**：验证完整用户流程
- **工具**：pytest + httpx (真实 HTTP 请求)
- **示例**：完整训练会话从创建到评估报告

---

## 测试结果

### 最新覆盖率报告

| 指标 | 数值 |
|------|------|
| **总测试数** | 102 |
| **通过率** | 100% |
| **跳过** | 0 |
| **覆盖率** | 86.78% |

### 模块覆盖率

| 模块 | 行覆盖率 | 分支覆盖率 | 状态 |
|------|----------|------------|------|
| `analyzer.py` | 84% | 75% | ✅ |
| `evaluator.py` | 94% | 88% | ✅ |
| `guidance.py` | 85% | 78% | ✅ |
| `simulator.py` | 82% | 70% | ✅ |
| `workflow.py` | 98% | 92% | ✅ |
| `hybrid_search.py` | 90% | 85% | ✅ |
| `chroma.py` | 93% | 80% | ✅ |
| `database.py` | 92% | 85% | ✅ |
| `embedding.py` | 91% | 82% | ✅ |
| `llm.py` | 82% | 68% | ✅ |
| `config.py` | 60% | 45% | ⚠️ 待提升 |

---

## 运行测试

### 完整测试套件

```bash
# 运行所有测试
uv run pytest

# 运行并显示详细输出
uv run pytest -v

# 运行并显示覆盖率
uv run pytest --cov=src --cov-report=term-missing
```

### 按层级运行

```bash
# 只运行单元测试
uv run pytest tests/test_analyzer.py tests/test_evaluator.py tests/test_guidance.py -v

# 只运行集成测试
uv run pytest tests/test_*_integration.py -v

# 只运行 E2E 测试
uv run pytest tests/test_e2e.py -v
```

### 运行特定测试

```bash
# 运行特定文件
uv run pytest tests/test_workflow.py -v

# 运行特定测试类
uv run pytest tests/test_evaluator.py::TestEvaluator -v

# 运行特定测试用例
uv run pytest tests/test_evaluator.py::TestEvaluator::test_evaluate_keyword_detection -v

# 按标记运行
uv run pytest -m "integration" -v
uv run pytest -m "not integration" -v
```

### 生成覆盖率报告

```bash
# 终端显示覆盖率
uv run pytest --cov=src --cov-report=term-missing

# 生成 HTML 报告
uv run pytest --cov=src --cov-report=html
# 报告生成在 htmlcov/index.html

# 生成 XML 报告（用于 CI/CD）
uv run pytest --cov=src --cov-report=xml
```

---

## 集成测试详解

### 测试文件列表

| 文件 | 测试内容 | 外部依赖 |
|------|----------|----------|
| `test_config_integration.py` | 配置加载、环境变量 | 无 |
| `test_database_integration.py` | SQLite CRUD、软删除 | SQLite |
| `test_chroma_integration.py` | 向量添加、查询、软删除 | Chroma |
| `test_embedding_integration.py` | DashScope Embedding API | DashScope |
| `test_llm_integration.py` | DashScope/DeepSeek LLM API | LLM Provider |

### 环境配置

集成测试通过 `conftest.py` 中的 `setup_env` fixture 从 `.env` 文件加载环境变量：

```python
@pytest.fixture(scope="session")
def setup_env():
    """从 .env 文件加载环境变量（仅集成测试需要）。"""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    yield
```

### 测试示例：Embedding API

```python
class TestEmbeddingIntegration:
    """Embedding API 集成测试 - 使用真实 API 调用。"""

    def test_encode_query_dashscope(self) -> None:
        """测试 DashScope Embedding API 真实调用。"""
        service = EmbeddingService(provider="dashscope")

        result = service.encode("这是一个测试文本")

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)
        assert len(result[0]) == 1536  # DashScope embedding dimension

    @pytest.mark.asyncio
    async def test_encode_query_async_dashscope(self) -> None:
        """测试 DashScope Embedding API 异步调用。"""
        service = EmbeddingService(provider="dashscope")

        result = await service.aencode("这是一个测试文本")

        assert isinstance(result, list)
        assert len(result) > 0
```

### 测试示例：LLM API

```python
class TestLLMIntegration:
    """LLM API 集成测试 - 使用真实 API 调用。"""

    @pytest.mark.asyncio
    async def test_llm_invoke_dashscope(self) -> None:
        """测试 DashScope LLM 真实调用。"""
        service = LLMService(provider="dashscope", model="qwen-turbo")

        response = await service.ainvoke("请用一句话介绍你自己")

        assert isinstance(response, str)
        assert len(response) > 0
        assert "qwen" in response.lower() or len(response) > 10

    @pytest.mark.asyncio
    async def test_llm_invoke_deepseek(self) -> None:
        """测试 DeepSeek LLM 真实调用。"""
        service = LLMService(provider="deepseek", model="deepseek-chat")

        response = await service.ainvoke("请用一句话介绍你自己")

        assert isinstance(response, str)
        assert len(response) > 0
```

### 测试示例：数据库

```python
class TestDatabaseIntegration:
    """数据库集成测试 - 使用真实 SQLite 文件。"""

    @pytest.fixture
    def test_db(self, tmp_path: Path) -> DatabaseService:
        """创建临时测试数据库。"""
        db_path = tmp_path / "test.db"
        service = DatabaseService(f"sqlite+aiosqlite:///{db_path}")
        yield service
        service.close()

    @pytest.mark.asyncio
    async def test_save_and_retrieve_session(self, test_db: DatabaseService) -> None:
        """测试会话保存和检索。"""
        session = SessionModel(
            id="test-session-001",
            customer_profile={"industry": "医疗"},
            product_info={"name": "测试产品"}
        )

        result = await test_db.save_session(session)
        assert result is not None

        retrieved = await test_db.get_session("test-session-001")
        assert retrieved is not None
        assert retrieved.id == "test-session-001"
```

### 测试示例：Chroma 向量库

```python
class TestChromaIntegration:
    """Chroma 向量库集成测试 - 使用真实 Chroma 实例。"""

    @pytest.fixture
    def test_chroma(self, tmp_path: Path) -> ChromaService:
        """创建临时测试向量库。"""
        persist_dir = tmp_path / "test_chroma"
        service = ChromaService(persist_dir=str(persist_dir))
        yield service
        service.close()

    @pytest.mark.asyncio
    async def test_add_and_query(self, test_chroma: ChromaService) -> None:
        """测试向量添加和查询。"""
        test_chroma.create_collection("test_collection")

        test_chroma.add_documents(
            collection_name="test_collection",
            documents=["这是一个关于糖尿病的文档"],
            doc_ids=["doc-001"],
            metadata=[{"source": "test"}]
        )

        results = test_chroma.query(
            collection_name="test_collection",
            query_texts=["糖尿病"],
            n_results=1
        )

        assert len(results) > 0
        assert "doc-001" in results[0]["ids"]
```

---

## 编写新测试

### 测试文件命名规范

```
tests/
├── test_<module_name>.py           # 单元测试
├── test_<module_name>_integration.py  # 集成测试
└── test_e2e.py                    # E2E 测试
```

### 测试类命名规范

```python
class TestModuleName:           # 单元测试
    """模块名测试类。"""

class TestModuleNameIntegration:  # 集成测试
    """模块名集成测试类。"""
```

### 测试函数命名规范

```python
def test_<what_is_being_tested>():
    """测试说明。"""

@pytest.mark.asyncio
async def test_<what_is_being_tested>_async():
    """异步测试说明。"""
```

### 示例：新增语义点评估测试

```python
class TestEvaluatorIntegration:
    """Evaluator 集成测试。"""

    @pytest.mark.asyncio
    async def test_evaluate_with_real_api(self) -> None:
        """使用真实 LLM API 测试评估功能。"""
        evaluator = SemanticEvaluator()

        result = await evaluator.evaluate(
            sales_message="这个药可以降低 HbA1c，低血糖风险很低",
            semantic_points=[
                SemanticPoint(
                    point_id="SP-001",
                    description="HbA1c 改善",
                    keywords=["HbA1c", "糖化血红蛋白"]
                ),
                SemanticPoint(
                    point_id="SP-002",
                    description="低血糖风险",
                    keywords=["低血糖", "安全"]
                )
            ]
        )

        assert "SP-001" in result.status
        assert "SP-002" in result.status
```

---

## CI/CD 集成

### GitHub Actions 配置

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync

      - name: Run tests
        run: uv run pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

### 覆盖率阈值

```bash
# 要求覆盖率不低于 85%
uv run pytest --cov=src --cov-fail-under=85
```

---

## 测试最佳实践

### Do's ✅

1. **集成测试使用真实 API**：不要 mock 外部服务
2. **每个测试独立**：测试之间不应有依赖关系
3. **明确的测试名称**：`test_<场景>_<预期行为>`
4. **单一职责**：每个测试只验证一个行为
5. **合理的 setup/teardown**：确保测试环境干净

### Don'ts ❌

1. **禁止在集成测试中使用 mock**
2. **不要让测试依赖于执行顺序**
3. **不要在测试中包含敏感信息**
4. **不要忽略失败的测试**
5. **不要跳过集成测试**

---

## 常见问题

### Q: 集成测试失败但本地通过

1. 检查 CI 环境的环境变量是否正确配置
2. 检查 API 配额是否耗尽
3. 查看 CI 日志确认具体错误

### Q: 覆盖率不达标

```bash
# 查看未覆盖的行
uv run pytest --cov=src --cov-report=term-missing

# 添加缺失的测试用例
```

### Q: 测试运行很慢

1. 确保集成测试是并行的（使用 pytest-xdist）
2. 检查 API 调用是否有不必要的延迟
3. 考虑使用测试数据库而非每次创建新数据库
