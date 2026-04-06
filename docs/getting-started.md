# 快速开始指南

本文档指导新用户从零开始搭建和运行 AI Sales Trainer Chatbot 系统。

## 目录

- [1. 环境要求](#1-环境要求)
- [2. 安装步骤](#2-安装步骤)
- [3. 配置说明](#3-配置说明)
- [4. 启动与验证](#4-启动与验证)
- [5. 常见问题排查](#5-常见问题排查)

---

## 1. 环境要求

| 依赖 | 版本要求 | 检查命令 |
|------|---------|---------|
| Python | >= 3.13 | `python --version` |
| uv | 最新版 | `uv --version` |
| 操作系统 | Windows 10+ / Linux / macOS | -- |
| 网络 | 可访问 DashScope 或 DeepSeek API | `ping dashscope.aliyuncs.com` |

### Python 版本确认

本项目要求 Python 3.13 或更高版本。检查当前版本：

```bash
python --version
# 预期输出: Python 3.13.x
```

如果版本不符合要求，请从 [Python 官网](https://www.python.org/downloads/) 下载安装。

### uv 包管理工具

uv 是一个快速的 Python 包管理器，替代传统的 pip + venv 组合。如果尚未安装：

```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后重启终端，验证：

```bash
uv --version
```

---

## 2. 安装步骤

### Step 1: 克隆项目

```bash
git clone <repository-url>
cd UMU_Test
```

### Step 2: 创建虚拟环境

```bash
uv venv
```

此命令在项目根目录下创建 `.venv` 虚拟环境目录。

### Step 3: 安装项目依赖

```bash
uv pip install -e .
```

`-e` 表示"可编辑模式"（editable install），代码修改后无需重新执行 install 即可生效。依赖声明在 `pyproject.toml` 中，包括：

- 核心框架：langgraph, langchain-core, fastapi
- LLM SDK：langchain-openai, openai, dashscope
- 数据存储：chromadb, sqlalchemy, aiosqlite
- 工具库：pydantic, pyyaml, httpx

### Step 4: （可选）安装开发依赖

如果需要运行测试或进行代码检查：

```bash
uv pip install -e ".[dev]"
```

开发依赖包括 pytest、ruff、basedpyright 等。

---

## 3. 配置说明

### 3.1 创建配置文件

项目根目录下提供了 `.env.example` 模板文件。复制并编辑：

```bash
cp .env.example .env
```

### 3.2 配置项说明

`.env` 文件包含 LLM API 密钥配置：

```bash
# ===========================================
# LLM Provider 配置（至少配置其中一个）
# ===========================================

# 方案 A: DashScope（阿里云通义千问）-- 主要推荐
DASHSCOPE_API_KEY="sk-your-dashscope-key-here"
DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# 方案 B: DeepSeek -- 备选
DS_API_KEY="sk-your-deepseek-key-here"
DS_BASE_URL="https://api.deepseek.com"
```

**重要提示**：

- **至少需要配置其中一个 Provider** 的 API 密钥
- 推荐优先使用 DashScope（qwen-plus 模型），它是本系统的主要测试模型
- API 密钥获取途径：
  - DashScope: 登录 [阿里云百炼平台](https://bailian.console.aliyun.com/) 创建 API Key
  - DeepSeek: 登录 [DeepSeek 开放平台](https://platform.deepseek.com/) 创建 API Key

### 3.3 安全提醒

- **切勿将 `.env` 文件提交到版本控制系统**
- `.gitignore` 中已默认排除 `.env` 文件
- 如果不小心提交了，请立即轮换（rotate）API 密钥

---

## 4. 启动与验证

### 4.1 初始化数据库

首次运行前需要初始化 SQLite 数据库：

```bash
uv run python init_db.py
```

此脚本创建 `umu_sales_trainer.db` 数据库文件，包含 sessions、messages、coverage_records 三张表。

### 4.2 启动服务

```bash
uv run uvicorn umu_sales_trainer.main:app --reload --port 8000
```

参数说明：

| 参数 | 作用 |
|------|------|
| `uv run` | 在项目的虚拟环境中执行命令 |
| `umunu_sales_trainer.main:app` | FastAPI 应用实例的导入路径 |
| `--reload` | 代码变更后自动重启服务（开发模式推荐） |
| `--port 8000` | 监听端口 |

启动成功的标志输出：

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process (pid: xxxx)
INFO:     Started server process (pid: xxxx)
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Loaded .env from ...\UMU_Test\.env
INFO:     Database initialized successfully
INFO:     Starting UMU Sales Trainer...
```

### 4.3 验证安装

#### 健康检查

打开新的终端窗口，执行：

```bash
curl http://localhost:8000/api/v1/health
```

预期响应：

```json
{"status": "healthy", "timestamp": "2026-01-15T10:30:00.000000Z"}
```

#### 访问前端页面

浏览器打开 http://localhost:8000 ，应自动重定向到 `/static/index.html` 并显示系统的主界面。

#### 功能验证流程

1. **创建会话**：在前端页面填写客户画像和产品信息，点击"开始训练"
2. **发送消息**：在对话框输入销售话术（如"您好，我是XX制药的销售代表..."），点击发送
3. **观察面板更新**：左侧应同时出现  个分析面板的内容：
   - AI 客户回复面板：显示 AI 模拟客户的回应
   - 语义覆盖面板：显示各卖点的覆盖状态和覆盖率进度条
   - 表达评分面板：显示清晰度/专业性/说服力三维分数
   - 会话洞察面板：显示当前销售阶段和意图
   - 智能引导面板：根据表现显示绿色优秀态或黄色改进态
4. **继续对话**：多轮对话观察分数的自然演进（应从低分逐步上升）

---

## 5. 常见问题排查

### 5.1 启动阶段

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `ModuleNotFoundError: No module named 'umu_sales_trainer'` | 未使用 `-e` 模式安装或不在项目目录下 | 确认已在项目根目录执行 `uv pip install -e .` |
| `DASHSCOPE_API_KEY environment variable not set` | `.env` 文件不存在或密钥为空 | 执行 `cp .env.example .env` 并填写密钥 |
| `port 8000 is already in use` | 上次启动的服务进程未退出 | Windows: `taskkill /F /IM python.exe`; Linux: `killall python` |
| `sqlite3.OperationalError: unable to open database file` | 未执行数据库初始化 | 运行 `uv run python init_db.py` |

### 5.2 运行阶段

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 发送消息后长时间无响应 | LLM API 调用超时或网络不通 | 检查网络连接；查看终端日志确认卡在哪一步 |
| AI 回复为兜底文本（如"我明白了，请继续"） | LLM 调用失败，触发了降级方案 | 检查 API 密钥有效性和余额 |
| 前端页面空白 / 404 | 静态文件挂载失败 | 确认 `static/` 目录存在且包含 `index.html` |
| 评分始终为 0 | 工作流异常终止 | 查看终端日志中的 `[error]` 信息 |
| CORS 错误（浏览器控制台报错） | 前端访问端口不在允许列表中 | 确认使用 localhost:8000 而非 IP 地址访问 |

### 5.3 日志查看

服务的标准输出包含详细的调试日志，关键字段可用于定位问题：

| 日志关键词 | 含义 |
|----------|------|
| `[start]` | 工作流起始节点 |
| `[parallel_fanout]` | 并行分发节点 |
| `[conversation_analyze]` | 对话分析完成，含 stage/intent/objections |
| `[semantic_eval]` | 语义覆盖检测完成，含 rate/uncovered |
| `[expression_eval]` | 表达评估完成，含 clarity/pro/persuasiveness |
| `[synthesize]` | 结果聚合完成，含 overall_score |
| `[guidance]` | 引导生成完成，含 actionable/priority_list 大小 |
| `[simulate]` | AI 客户回复生成完成 |
| `[end]` | 工作流结束 |
| `FAILED` / `Error` | 异常发生，需关注上下文 |
