# 快速开始

本文档帮助您快速搭建和运行 UMU Sales Trainer 系统。

## 环境准备

### 1. 安装 Python 3.13+

建议使用 uv 管理 Python 版本：

```bash
# 安装 uv (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用 pip
pip install uv
```

### 2. 克隆项目

```bash
git clone https://gitee.com/your-repo/umu-sales-trainer.git
cd umu-sales-trainer
```

## 配置

### 1. 创建环境变量文件

```bash
cp .env.example .env
```

### 2. 编辑 .env 文件

```bash
# 填入您的 DashScope API 密钥
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx

# DeepSeek API 密钥 (可选)
DS_API_KEY=sk-xxxxxxxxxxxx

# 默认使用 DashScope
LLM_PROVIDER=dashscope
```

> 获取 API 密钥：
> - DashScope: https://dashscope.console.aliyun.com/
> - DeepSeek: https://platform.deepseek.com/

## 安装与初始化

### 1. 安装依赖

```bash
uv sync
```

### 2. 初始化数据库

```bash
uv run python init_db.py
```

输出示例：
```
[INFO] 开始初始化数据库...
[INFO] 数据库初始化完成: umu_sales.db
```

### 3. 初始化知识库

```bash
uv run python init_knowledge.py
```

输出示例：
```
[INFO] 开始初始化知识库...
[INFO] 创建 Collection: objection_handling
[INFO] 创建 Collection: product_knowledge
[INFO] 创建 Collection: excellent_samples
[INFO] 知识库初始化完成
```

## 启动服务

### 开发模式

```bash
uv run uvicorn umu_sales_trainer.main:app --reload --port 8000
```

### 生产模式

```bash
uv run uvicorn umu_sales_trainer.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 访问系统

启动后访问：

- 前端页面：http://localhost:8000/static/index.html
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/api/v1/health

## 首次使用

1. 打开前端页面
2. 点击"新建会话"按钮
3. 在输入框输入销售话术
4. 系统会分析您的表达并给出引导
5. 根据引导继续练习

## 下一步

- 查看 [API 文档](api.md) 了解接口详情
- 查看 [架构文档](architecture.md) 了解系统设计
- 查看 [测试文档](testing.md) 了解测试方法
