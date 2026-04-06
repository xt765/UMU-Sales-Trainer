# UMU Sales Trainer

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-green.svg)](https://www.python.org/downloads/)
[![LangGraph v1.0](https://img.shields.io/badge/LangGraph-v1.0-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Test Coverage](https://img.shields.io/badge/Coverage-86%25-brightgreen.svg)](https://github.com/your-repo/umu-sales-trainer)

</div>

## 目录

- [项目简介](#项目简介)
- [核心能力](#核心能力)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [测试覆盖](#测试覆盖)
- [ License](#license)

---

## 项目简介

**UMU Sales Trainer** 是一款基于大语言模型的**AI 销售训练系统**，专注于**信息传递阶段（Value Delivery）**的能力提升。

### 场景描述

系统模拟真实的销售拜访场景：

| 角色 | 描述 |
|------|------|
| **AI 客户** | 扮演内分泌科主任医师，专业、谨慎、注重循证医学证据 |
| **销售人员** | 用户扮演拜访者，向客户介绍产品价值 |
| **系统** | 分析表达内容、判断信息覆盖、引导补充未传达卖点 |

### 训练目标

训练销售人员在拜访中**完整传达产品核心卖点**：

| 语义点 ID | 卖点描述 | 识别关键词 |
|-----------|----------|------------|
| SP-001 | HbA1c 改善 | HbA1c、糖化血红蛋白、血糖控制、降糖 |
| SP-002 | 低血糖风险 | 低血糖、低血糖风险、安全、安心 |
| SP-003 | 用药便利性 | 一周一次、给药便利、依从性、简单 |

---

## 核心能力

### 能力层级体系

本系统采用**三层能力递进架构**，从基础到高阶：

```
┌─────────────────────────────────────────────────────────────────┐
│                      自主性金字塔                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                         ▲                                        │
│                        / │ \                                     │
│                       /  │  \         ← 自主决策层               │
│                      /   │   \           (动态调整策略)          │
│                     /────┼────\                                  │
│                    /     │     \       ← 智能增强层               │
│                   /      │      \         (Agentic RAG)          │
│                  /───────┼───────\                               │
│                 /        │        \     ← 基础执行层               │
│                /         │         \       (规则引擎)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| 层级 | 能力 | 实现方式 |
|------|------|----------|
| **基础执行层** | 关键词匹配、规则路由 | 正则表达式、简单判断 |
| **智能增强层** | Agentic RAG、语义理解 | LangGraph + Chroma 向量检索 |
| **自主决策层** | 动态策略调整、自主优化 | LLM 推理、上下文学习 |

### 核心技术特性

#### 1️⃣ 三层语义检测机制

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────┐
│  第一层：关键词检测 (快速过滤)                  │
│  ───────────────────────────────────         │
│  权重：20%                                   │
│  速度：< 1ms                                 │
│  用途：快速排除明显不相关的内容                 │
└─────────────────┬───────────────────────────┘
                  │ 未命中
                  ▼
┌─────────────────────────────────────────────┐
│  第二层：Embedding 相似度 (语义匹配)           │
│  ───────────────────────────────────         │
│  权重：30%                                   │
│  速度：< 10ms                                │
│  用途：识别近义词、表达变体                   │
└─────────────────┬───────────────────────────┘
                  │ 置信度 < 阈值
                  ▼
┌─────────────────────────────────────────────┐
│  第三层：LLM 零样本分类 (最终判断)             │
│  ───────────────────────────────────         │
│  权重：50%                                   │
│  速度：< 2s                                  │
│  用途：复杂语义、隐含表达                     │
└─────────────────────────────────────────────┘
```

#### 2️⃣ Agentic RAG 系统

传统 RAG vs Agentic RAG：

| 特性 | 传统 RAG | Agentic RAG（本案采用） |
|------|----------|------------------------|
| **检索模式** | 单次检索 | 多 Collection 协同检索 |
| **结果融合** | 简单拼接 | RRF + 动态加权 |
| **工具调用** | 无 | LangGraph Tool Calling |
| **上下文感知** | 有限 | 完整对话上下文 |

**RRF（Reciprocal Rank Fusion）融合公式：**

```
RRF Score = Σ 1 / (k + rankᵢ)

其中：
- rankᵢ: 第 i 个检索结果在该 Collection 中的排名
- k: 常数（通常 k=60，值越大排名权重越平滑）
```

**动态加权策略：**

| 场景 | 异议处理库 | 产品知识库 | 话术示例库 |
|------|------------|------------|------------|
| 客户提出异议 | 0.55 | 0.30 | 0.15 |
| 有未覆盖卖点 | 0.25 | 0.25 | 0.50 |
| 产品介绍阶段 | 0.15 | 0.60 | 0.25 |
| 客户态度消极 | 0.50 | 0.30 | 0.20 |

#### 3️⃣ LangGraph StateGraph 工作流

```
                    ┌──────────────────┐
                    │     START        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  validate_input  │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │         输入有效?            │
              └──────────────┬──────────────┘
                       Yes   │   No
              ┌──────────────┘   └──────────────┐
              ▼                                 ▼
      ┌───────────────┐               ┌───────────────┐
      │   analyze     │               │  return_error │
      └───────┬───────┘               └───────────────┘
              │
              ▼
      ┌───────────────┐
      │   evaluate    │
      └───────┬───────┘
              │
     ┌────────┴────────┐
     │  pending_points │
     │      存在?      │
     └────────┬────────┘
        Yes   │    No
   ┌──────────┘    └──────────┐
   ▼                           ▼
┌────────────┐          ┌────────────┐
│   guide   │          │  simulate  │
└────┬──────┘          └─────┬──────┘
     │                        │
     └───────────┬────────────┘
                 ▼
        ┌────────────────┐
        │  turn >= MAX   │
        │     or all     │
        │   covered?     │
        └────────┬───────┘
          Yes   │    No
   ┌──────────┘    └──────────┐
   ▼                           ▼
┌────────────┐          ┌────────────┐
│  finalize  │          │   loop     │
└─────┬──────┘          └─────┬──────┘
      │                        │
      ▼                        ▼
┌────────────┐          ┌────────────┐
│    END     │          │  analyze   │
└────────────┘          └────────────┘
```

---

## 系统架构

### 分层架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         表现层 (Presentation)                       │
│              原生 HTML5 + CSS3 + ES6+ (无框架依赖)                   │
├─────────────────────────────────────────────────────────────────────┤
│                          API 层 (API Layer)                         │
│                      FastAPI + Uvicorn ASGI                        │
├─────────────────────────────────────────────────────────────────────┤
│                        工作流层 (Workflow Layer)                    │
│                    LangGraph StateGraph v1.0                        │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐       │
│  │ analyze │→│evaluate│→│ decide │→│ guide  │→│respond │       │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘       │
├─────────────────────────────────────────────────────────────────────┤
│                       业务逻辑层 (Business Logic)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────┐  │
│  │ SalesAnalyzer│  │SemanticEval  │  │   Guidance  │  │Simulator │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                         服务层 (Service Layer)                      │
│  ┌────────┐  ┌──────────┐  ┌────────┐  ┌────────┐  ┌───────────┐ │
│  │ LLM    │  │ Embedding │  │ Chroma │  │Database │  │HybridSearch│ │
│  └────────┘  └──────────┘  └────────┘  └────────┘  └───────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                          数据层 (Data Layer)                        │
│          SQLite (结构化数据)          │          Chroma (向量数据)    │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据存储架构

| 存储类型 | 使用场景 | 数据特点 |
|----------|----------|----------|
| **SQLite** | 会话、消息、评估结果 | 结构化、强事务、单文件 |
| **Chroma** | 产品知识、异议处理、话术示例 | 向量检索、语义相似度 |

### 软删除机制

为保证数据一致性，采用**双软删除策略**：

```
删除操作流程：
┌──────────────────────────────────────────────────────────────┐
│ 1. UPDATE sessions SET is_deleted=TRUE                       │
│ 2. UPDATE messages SET is_deleted=TRUE                       │
│ 3. INSERT INTO pending_operations ('delete_chroma', 'xxx')  │
│ 4. 提交 SQLite 事务                                         │
│ 5. 后台补偿任务读取 pending_operations                       │
│ 6. 调用 Chroma API 删除向量                                  │
│ 7. 如果失败，最多重试 3 次 → 告警                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 技术栈

### 核心技术选型

| 类别 | 技术选型 | 选型理由 |
|------|----------|----------|
| **LLM** | DashScope (Qwen) / DeepSeek | OpenAI 兼容接口，国内可用 |
| **Embedding** | DashScope text-embedding-v1 | 高精度、支持中文 |
| **工作流** | LangGraph v1.0 StateGraph | 状态机模式适合对话流程 |
| **向量库** | ChromaDB | 轻量、Python 原生、部署简单 |
| **关系库** | SQLite + SQLAlchemy | 单文件、无依赖、易维护 |
| **API** | FastAPI + Uvicorn | 异步、高性能、自动文档 |

### 完整依赖列表

| 依赖 | 版本 | 用途 |
|------|------|------|
| `langgraph` | ≥1.0.0 | 工作流引擎 |
| `langchain-core` | ≥0.3.0 | 核心抽象 |
| `langchain-openai` | ≥0.2.0 | OpenAI 兼容接口 |
| `dashscope` | ≥1.20.0 | 通义千问 SDK |
| `chromadb` | ≥0.4.0 | 向量数据库 |
| `fastapi` | ≥0.115.0 | Web 框架 |
| `uvicorn` | ≥0.30.0 | ASGI 服务器 |
| `aiosqlite` | ≥0.20.0 | 异步 SQLite |
| `pytest` | ≥8.0.0 | 测试框架 |
| `pytest-asyncio` | ≥0.24.0 | 异步测试 |
| `pytest-cov` | ≥6.0.0 | 覆盖率统计 |
| `ruff` | ≥0.8.0 | 代码格式化 |
| `basedpyright` | ≥1.0.0 | 类型检查 |

---

## 快速开始

### 环境要求

- Python 3.13+
- uv 包管理器

### 安装步骤

```bash
# 1. 克隆项目
git clone https://gitee.com/your-repo/umu-sales-trainer.git
cd umu-sales-trainer

# 2. 安装依赖
uv sync

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入您的 API 密钥

# 4. 初始化数据库
uv run python init_db.py

# 5. 初始化知识库
uv run python init_knowledge.py

# 6. 启动服务
uv run uvicorn umu_sales_trainer.main:app --reload --port 8000
```

### 访问系统

| 页面 | 地址 |
|------|------|
| 前端界面 | http://localhost:8000/static/index.html |
| API 文档 | http://localhost:8000/docs |
| 健康检查 | http://localhost:8000/api/v1/health |

---

## 项目结构

```
umu-sales-trainer/
├── src/umu_sales_trainer/
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── api/
│   │   ├── router.py       # API 路由
│   │   └── middleware.py   # 中间件
│   ├── core/                # 核心业务逻辑
│   │   ├── workflow.py     # LangGraph 工作流
│   │   ├── analyzer.py     # 销售发言分析
│   │   ├── evaluator.py    # 语义点评估
│   │   ├── guidance.py     # 引导话术生成
│   │   ├── simulator.py    # AI 客户模拟
│   │   └── hybrid_search.py # 混合搜索
│   ├── models/              # 数据模型
│   │   ├── customer.py     # 客户画像
│   │   ├── product.py      # 产品信息
│   │   ├── semantic.py     # 语义点
│   │   ├── conversation.py # 对话模型
│   │   └── evaluation.py   # 评估模型
│   └── services/            # 服务层
│       ├── llm.py          # LLM 服务
│       ├── embedding.py    # Embedding 服务
│       ├── chroma.py       # 向量数据库
│       └── database.py     # 关系数据库
├── data/                    # 配置数据
│   ├── customer_profiles/  # 客户画像
│   ├── products/           # 产品资料
│   └── knowledge/          # 知识库
├── static/                  # 前端资源
├── tests/                   # 测试文件
├── init_db.py              # 数据库初始化
├── init_knowledge.py       # 知识库初始化
├── pyproject.toml          # 项目配置
└── ruff.toml                # 代码格式化配置
```

---

## 测试覆盖

### 测试金字塔

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

### 测试结果

| 指标 | 数值 |
|------|------|
| **总测试数** | 102 |
| **通过率** | 100% |
| **覆盖率** | 86.78% |
| **跳过** | 0 |

### 测试原则

> **⚠️ 硬性要求：所有集成测试必须使用真实 API 调用，禁止任何 mock**

```bash
# 运行所有测试
uv run pytest

# 运行测试并查看覆盖率
uv run pytest --cov=src --cov-report=term-missing

# 运行特定测试
uv run pytest tests/test_workflow.py -v
```

---

## License

MIT License

Copyright (c) 2024 UMU

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
