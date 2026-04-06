# 测试策略文档

本文档描述 AI Sales Trainer Chatbot 的测试框架、核心测试场景和运行方法。

## 目录

- [1. 测试框架与配置](#1-测试框架与配置)
- [2. 测试结构](#2-测试结构)
- [3. 核心测试场景](#3-核心测试场景)
- [4. 运行测试](#4-运行测试)
- [5. 添加新测试的规范](#5-添加新测试的规范)

---

## 1. 测试框架与配置

### 1.1 技术栈

| 工具 | 版本 | 用途 |
|------|------|------|
| pytest | >= 8.0 | 测试框架 |
| pytest-asyncio | >= 0.24 | 异步测试支持（asyncio_mode=auto） |
| pytest-cov | >= 6.0 | 覆盖率报告 |
| ruff | >= 0.8 | 代码格式化（非测试专用但影响测试代码风格） |
| basedpyright | >= 1.0 | 静态类型检查 |

### 1.2 pytest 配置

配置位于 `pyproject.toml` 的 `[tool.pytest.ini_options]` 段：

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"       # 自动检测异步测试，无需 @pytest.mark.asyncio
testpaths = ["tests"]         # 测试搜索路径
```

### 1.3 覆盖率配置

```toml
[tool.coverage.run]
source = ["src"]             # 覆盖率统计范围：仅 src/ 目录
branch = true                # 启用分支覆盖率

[tool.coverage.report]
exclude_lines =              # 排除不计入覆盖率的行
    "pragma: no cover"
    "def __repr__"
    "raise AssertionError"
    "raise NotImplementedError"
    'if __name__ == .__main__.:'
```

**覆盖率目标**: > 80%

---

## 2. 测试结构

### 2.1 测试目录树

```
tests/
├── conftest.py                 # 共享 fixtures
├── test_analyzer.py            # ConversationAnalyst 单元测试
├── test_evaluator.py           # SemanticCoverageExpert + ExpressionCoach 测试
├── test_guidance.py            # GuidanceMentor 测试
├── test_workflow.py            # LangGraph 工作流集成测试
├── test_services.py            # 基础服务层测试
├── test_config_integration.py  # 配置集成测试
├── test_database_integration.py # 数据库集成测试
├── test_embedding_integration.py # Embedding 服务集成测试
├── test_llm_integration.py     # LLM 服务集成测试
└── test_chroma_integration.py  # ChromaDB 向量数据库集成测试
```

### 2.2 文件职责划分

| 测试文件 | 被测模块 | 测试类型 | 依赖外部服务 |
|---------|---------|---------|:-----------:|
| `test_analyzer.py` | `core/analyzer.py` | 单元测试 | 是（LLM） |
| `test_evaluator.py` | `core/evaluator.py` | 单元测试 | 部分（LLM 可 mock） |
| `test_guidance.py` | `core/guidance.py` | 单元测试 | 是（LLM） |
| `test_workflow.py` | `core/workflow.py` | 集成测试 | 是（LLM + Embedding） |
| `test_*.py` (integration) | `services/*.py` | 集成测试 | 视具体服务而定 |

---

## 3. 核心测试场景

### 3.1 评分算法测试（重点）

评分算法是系统的核心逻辑，必须覆盖以下场景：

#### 3.1.1 4 因子公式正确性

| 测试用例 | 输入 | 预期输出 | 验证要点 |
|---------|------|---------|---------|
| `test_full_coverage_full_expression` | coverage=1.0, 全 10 分 | 接近满分（受回合惩罚调整） | sqrt(1)*40 + 35 = 75 分原始值 |
| `test_zero_coverage_zero_expression` | coverage=0.0, 全 1 分 | 低分 | 仅回合惩罚和质量调整起作用 |
| `test_sqrt_compression` | coverage 渐增 | 非线性增长 | 验证 sqrt 压缩效果 |
| `test_turn_penalty_progression` | 相同输入，turn 1-5 | 分数递增 | T1 < T2 < T3 < T4 < T5 |

#### 3.1.2 典型分数演进推演

| 测试用例 | 场景 | 预期分数区间 |
|---------|------|-----------|
| `test_turn_1_greeting` | 第 1 轮简单问候 | 8-25 |
| `test_turn_2_introduction` | 第 2 轮初步介绍 | 25-40 |
| `test_turn_3_presentation` | 第 3 轮产品呈现 | 40-55 |
| `test_turn_4_objection` | 第 4 轮异议处理 | 50-65 |
| `test_turn_5_closing` | 第 5 轮缔结 | 60-80 |

#### 3.1.3 消息质量调整

| 测试用例 | 消息长度 | 覆盖率 | 预期调整 |
|---------|---------|-------|---------|
| `test_very_short_message` | 10 字符 | 任意 | -5 |
| `test_short_message` | 20 字符 | 任意 | -2 |
| `test_long_message_with_coverage` | 150 字符 | >= 67% | +3 |
| `test_long_message_full_coverage` | 100 字符 | 100% | +2 |
| `test_normal_message` | 60 字符 | 50% | 0 |

### 3.2 异议识别测试

| 测试类别 | 覆盖范围 |
|---------|---------|
| 9 类关键词全覆盖 | 每类至少 1 个正向测试用例 |
| 多类异议同时存在 | 同一句话包含价格+安全性等 |
| 无异议信号 | 正常陈述句返回空列表 |
| 边界情况 | 部分匹配 / 误报检测 |

### 3.3 引导双态测试

| 测试用例 | 覆盖率 | 综合评分 | 预期 is_actionable |
|---------|-------|---------|:------------------:|
| `test_excellent_state` | >= 80% | >= 70 | **False**（优秀态） |
| `test_low_coverage_improvement` | < 80% | 任意 | **True**（改进态） |
| `test_low_score_improvement` | >= 80% | < 70 | **True**（改进态） |
| `test_both_good_excellent` | >= 80% | >= 70 | **False**（优秀态） |

### 3.4 并行工作流测试

| 测试场景 | 验证要点 |
|---------|---------|
| `test_parallel_execution` | 三个分析 Agent 的结果均出现在最终 WorkflowState 中 |
| `test_synthesize_aggregation` | synthesize 节点正确聚合 coverage + expression + conversation |
| `test_guidance_skip_when_excellent` | 双条件满足时 guidance_result 为 None |
| `test_guidance_execute_when_needed` | 任一条件不满足时 guidance_result 非 None |
| `test_simulate_generates_response` | simulate 节点返回非空 ai_response |

### 3.5 Agent 降级策略测试

| Agent | LLM 失败时的降级行为 | 测试方法 |
|-------|-------------------|---------|
| ConversationAnalyst | `_rule_based_analysis()` 关键词匹配 | Mock LLM 抛出异常 |
| SemanticCoverageExpert | LLM judgment 返回 0.5（中性分数） | Mock LLM 抛出异常 |
| ExpressionCoach | `_rule_based_expression_analysis()` 规则估算 | Mock LLM 抛出异常 |
| GuidanceMentor | 返回基础引导结果（无 LLM 增强） | Mock LLM 抛出异常 |
| CustomerSimulator | `_generate_fallback_response()` 阶段模板 | Mock LLM 抛出异常 |

---

## 4. 运行测试

### 4.1 运行全部测试

```bash
uv run pytest tests/ -v --cov=src
```

参数说明：

| 参数 | 作用 |
|------|------|
| `-v` | 详细输出每个测试用例的通过/失败状态 |
| `--cov=src` | 生成覆盖率报告（需安装 pytest-cov） |

### 4.2 运行单个测试文件

```bash
# 评分算法测试（最常用）
uv run pytest tests/test_evaluator.py -v

# 对话分析测试
uv run pytest tests/test_analyzer.py -v

# 工作流集成测试
uv run pytest tests/test_workflow.py -v
```

### 4.3 运行单个测试函数

```bash
# 运行特定的评分推演测试
uv run pytest tests/test_evaluator.py::test_overall_score_progression -v

# 运行特定类的全部测试
uv run pytest tests/test_evaluator.py::TestExpressionCoach -v
```

### 4.4 仅运行失败的测试（上次）

```bash
uv run pytest tests/ --lf -v
```

### 4.5 查看覆盖率报告

覆盖率报告以表格形式输出到终端，同时生成 HTML 报告：

```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
```

HTML 报告默认生成在 `htmlcov/` 目录下，可用浏览器打开 `htmlcov/index.html` 查看逐行覆盖率。

---

## 5. 添加新测试的规范

### 5.1 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 测试文件 | `test_<module>.py` | `test_evaluator.py` |
| 测试类 | `Test<Class>` | `TestExpressionCoach` |
| 测试函数 | `test_<scenario>_<expected>` | `test_low_coverage_returns_improvement_state` |

### 5.2 Fixture 使用

共享 fixture 定义在 `conftest.py` 中，常用的包括：

| Fixture 名称 | 提供内容 | 作用域 |
|-------------|---------|--------|
| `llm_service` | Mock 的 LLMService 实例 | session（每次测试重建） |
| `embedding_service` | Mock 的 EmbeddingService 实例 | session |
| `sample_customer_profile` | 标准 CustomerProfile 对象 | session |
| `sample_product_info` | 标准 ProductInfo 对象 | session |
| `sample_semantic_points` | 3 个 SemanticPoint 列表 | session |

### 5.3 测试编写模板

```python
"""测试 <模块名> 的 <功能描述>。
"""

from __future__ import annotations

import pytest

from umu_sales_trainer.core.<module> import <Class>


class Test<Class>:
    """<Class> 的单元测试套件。"""

    def test_<场景>_<预期行为>(self, <fixture>) -> None:
        """<简述测试场景和预期结果>。

        Args:
            <fixture>: <fixture 的说明>
        """
        # Arrange: 准备测试数据
        # Act: 执行被测操作
        # Assert: 验证结果
        ...
```
