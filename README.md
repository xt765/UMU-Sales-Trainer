# UMU Sales Trainer

AI 销售训练 Chatbot 系统，基于 LangGraph v1.0 和 LangChain v1.2 构建。

## 简介

UMU Sales Trainer 是一个 AI 驱动的销售训练系统，通过模拟真实销售场景帮助销售人员提升沟通能力。系统扮演客户（内分泌科主任），销售人员扮演拜访者，进行多轮对话训练。

## 核心特性

### 🎯 场景化训练
- AI 扮演专业客户（内分泌科主任）
- 模拟真实销售拜访对话
- 多轮交互评估销售表达能力

### 🧠 智能评估
- **三层检测机制**：关键词 → Embedding → LLM
- **语义点覆盖分析**：评估信息传递完整性
- **表达质量评估**：清晰度、专业性、说服力

### 🔍 Agentic RAG
- **RRF 融合检索**：Reciprocal Rank Fusion 多路召回
- **动态加权算法**：根据上下文智能调整权重
- **Chroma + SQLite**：向量库与结构化数据双存储

### 🛠️ 技术架构
- **LangGraph StateGraph**：6 节点工作流 + 条件边路由
- **DashScope/DeepSeek**：多 Provider LLM 支持
- **FastAPI + Uvicorn**：高性能异步 API
- **软删除机制**：SQLite + Chroma 数据一致性

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (SPA)                            │
│                   index.html + app.js                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI 服务层                          │
│  POST /api/v1/sessions        - 创建训练会话               │
│  POST /api/v1/sessions/{id}  - 发送消息                    │
│  GET  /api/v1/sessions/{id}  - 获取评估结果                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 工作流                         │
│  start → analyze → evaluate → [guidance | simulate] → end  │
└─────────────────────────────────────────────────────────────┘
          │           │           │
          ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────┐
│                     核心业务逻辑                             │
│  SalesAnalyzer    - 销售发言分析                            │
│  SemanticEvaluator - 语义点评估                             │
│  GuidanceGenerator - 引导话术生成                            │
│  CustomerSimulator - AI客户模拟                            │
└─────────────────────────────────────────────────────────────┘
          │           │           │
          ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────┐
│                      服务层 (Services)                       │
│  LLMService       - DashScope/DeepSeek 多Provider          │
│  EmbeddingService - DashScope text-embedding-v1              │
│  DatabaseService  - SQLite + 软删除                         │
│  ChromaService   - 向量数据库 + 软删除                       │
│  HybridSearchEngine - RRF + 动态加权融合                    │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.13+
- uv 包管理器

### 安装依赖

```bash
# 克隆项目
git clone https://gitee.com/your-repo/umu-sales-trainer.git
cd umu-sales-trainer

# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入您的 API 密钥
```

### 初始化数据库

```bash
uv run python init_db.py
```

### 初始化知识库

```bash
uv run python init_knowledge.py
```

### 启动服务

```bash
uv run uvicorn umu_sales_trainer.main:app --reload --port 8000
```

访问 http://localhost:8000/static/index.html 开始训练。

## 项目结构

```
umu-sales-trainer/
├── src/umu_sales_trainer/
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── api/
│   │   ├── router.py       # API 路由
│   │   └── middleware.py   # 中间件
│   ├── core/
│   │   ├── workflow.py     # LangGraph 工作流
│   │   ├── analyzer.py     # 销售发言分析
│   │   ├── evaluator.py    # 语义点评估
│   │   ├── guidance.py     # 引导话术生成
│   │   ├── simulator.py    # AI 客户模拟
│   │   └── hybrid_search.py # 混合搜索
│   ├── models/             # 数据模型
│   ├── services/           # 服务层
│   └── repositories/       # 配置仓储
├── data/
│   ├── customer_profiles/  # 客户画像
│   ├── products/           # 产品资料
│   └── knowledge/          # 知识库
├── static/                 # 前端资源
├── tests/                  # 测试文件
├── init_db.py             # 数据库初始化
└── init_knowledge.py      # 知识库初始化
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/sessions` | 创建训练会话 |
| POST | `/api/v1/sessions/{id}/messages` | 发送消息 |
| GET | `/api/v1/sessions/{id}/evaluation` | 获取评估结果 |
| DELETE | `/api/v1/sessions/{id}` | 删除会话 |
| GET | `/api/v1/sessions/{id}/status` | 获取会话状态 |
| GET | `/api/v1/health` | 健康检查 |

## 配置说明

### 环境变量

```bash
# DashScope 配置
DASHSCOPE_API_KEY=your-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# DeepSeek 配置
DS_API_KEY=your-api-key
DS_BASE_URL=https://api.deepseek.com

# 默认 LLM Provider
LLM_PROVIDER=dashscope

# 数据库配置
DATABASE_URL=sqlite+aiosqlite:///./umu_sales.db

# Chroma 配置
CHROMA_PERSIST_DIR=./chroma_db
```

## 测试

```bash
# 运行所有测试
uv run pytest

# 运行测试并查看覆盖率
uv run pytest --cov=src --cov-report=term-missing

# 运行特定测试
uv run pytest tests/test_workflow.py -v
```

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | DashScope, DeepSeek |
| Embedding | DashScope text-embedding-v1 |
| 向量数据库 | ChromaDB |
| 结构化数据库 | SQLite + SQLAlchemy |
| 工作流 | LangGraph v1.0 |
| API | FastAPI + Uvicorn |
| 测试 | pytest + pytest-asyncio + pytest-cov |

## License

MIT
