# 系统架构

本文档详细介绍 UMU Sales Trainer 的技术架构和核心设计。

## 整体架构

UMU Sales Trainer 采用分层架构设计，从上到下依次为：

```
┌─────────────────────────────────────────────────────────────┐
│                      表现层 (Presentation)                    │
│                   HTML + CSS + JavaScript SPA                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      API 层 (API Layer)                       │
│                    FastAPI + 中间件                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    工作流层 (Workflow Layer)                   │
│                  LangGraph StateGraph                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    业务逻辑层 (Business Logic)                 │
│     SalesAnalyzer | SemanticEvaluator | GuidanceGenerator    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      服务层 (Service Layer)                   │
│    LLMService | EmbeddingService | ChromaService | ...      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      数据层 (Data Layer)                     │
│              SQLite (结构化) | Chroma (向量)                   │
└─────────────────────────────────────────────────────────────┘
```

## LangGraph 工作流

### 节点设计

系统使用 6 个节点处理对话流程：

| 节点 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `start` | 输入验证 | 销售消息 | 验证结果 |
| `analyze` | 发言分析 | 销售消息 + 上下文 | 分析结果 |
| `evaluate` | 语义评估 | 分析结果 | 评估结果 |
| `guidance` | 生成引导 | 未覆盖语义点 | 引导话术 |
| `simulate` | 客户模拟 | 销售消息 + 上下文 | AI 回复 |
| `end` | 结束处理 | 最终状态 | 完成状态 |

### 条件边路由

```
                    ┌──────────────┐
                    │    start     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   输入有效?   │
                    └──────┬───────┘
                     Yes   │   No
              ┌────────────┘   └────────────┐
              ▼                             ▼
       ┌────────────┐              ┌────────────┐
       │  analyze  │              │    end     │
       └─────┬─────┘              └────────────┘
             │
             ▼
       ┌────────────┐
       │  evaluate  │
       └─────┬─────┘
             │
      ┌─────▼─────┐
      │ 覆盖率<80%? │
      └─────┬─────┘
       Yes  │  No
    ┌──────┘  └──────┐
    ▼                ▼
┌────────┐    ┌────────────┐
│guidance│    │  simulate  │
└────┬───┘    └──────┬─────┘
     │                │
     └───────┬────────┘
             ▼
       ┌────────────┐
       │    end     │
       └────────────┘
```

## 三层语义检测

系统采用三层检测机制评估语义点覆盖：

### 第一层：关键词检测 (权重 20%)

```python
def _keyword_detection(self, message: str, point: SemanticPoint) -> float:
    """检测消息中是否包含语义点的关键词。"""
    keywords_found = sum(1 for kw in point.keywords if kw in message)
    return keywords_found / len(point.keywords) if point.keywords else 0.0
```

### 第二层：Embedding 相似度 (权重 30%)

```python
def _embedding_similarity(self, message: str, point: SemanticPoint) -> float:
    """计算消息与语义点描述的向量相似度。"""
    message_emb = self.embedding_service.encode_query(message)
    point_emb = self.embedding_service.encode_query(point.description)
    return cosine_similarity(message_emb, point_emb)
```

### 第三层：LLM 判断 (权重 50%)

```python
def _llm_judgment(self, message: str, point: SemanticPoint) -> float:
    """使用 LLM 判断语义是否被正确表达。"""
    prompt = f"判断以下销售话术是否覆盖了语义点：'{point.description}'"
    response = self.llm.invoke([HumanMessage(content=prompt)])
    return parse_llm_score(response.content)
```

## Agentic RAG

### RRF 融合算法

Reciprocal Rank Fusion (RRF) 用于融合多路检索结果：

```python
def rrf_fusion(results: List[List[dict]], k: int = 60) -> List[dict]:
    """RRF 融合多路检索结果。

    公式：RRF(score) = Σ 1/(k + rank)
    """
    scores = defaultdict(float)
    for result_list in results:
        for rank, doc in enumerate(result_list):
            doc_id = doc["id"]
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 动态加权融合

根据上下文动态调整不同来源的权重：

```python
def dynamic_weight(results: List[dict], context: dict) -> List[dict]:
    """根据上下文信息动态调整权重。"""
    base_weight = context.get("base_weight", 1.0)
    source_weight = context.get("source_weight", {})

    for doc in results:
        source = doc.get("source")
        doc["score"] = doc["score"] * base_weight * source_weight.get(source, 1.0)

    return results
```

## 软删除机制

### SQLite 软删除

```python
class SessionModel(Base):
    is_deleted = Column(Integer, default=0)  # 0=未删除, 1=已删除
    deleted_at = Column(DateTime, nullable=True)
    deleted_by = Column(String, nullable=True)
```

查询时自动过滤已删除记录：

```python
def get_session(self, session_id: str) -> Optional[SessionModel]:
    stmt = select(SessionModel).where(
        SessionModel.id == session_id,
        SessionModel.is_deleted == 0  # 只返回未删除记录
    )
    return self.session.execute(stmt).scalar_one_or_none()
```

### Chroma 软删除

通过 metadata 标记：

```python
def soft_delete(self, collection_name: str, doc_id: str) -> None:
    """软删除文档。"""
    collection = self.client.get_collection(collection_name)
    collection.update(
        id=doc_id,
        metadata={"is_deleted": True}
    )
```

查询时过滤已删除文档：

```python
def query(self, collection_name: str, query_texts: List[str], n_results: int):
    """查询时自动排除已删除文档。"""
    collection = self.client.get_collection(collection_name)
    results = collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where={"is_deleted": False}  # 只返回未删除文档
    )
    return results
```

## 数据流

### 完整对话流程

```
用户输入 → API Router → 工作流 invoke()
                            │
                            ▼
                    ┌───────────────┐
                    │ 验证输入      │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ 分析销售发言  │ ← SalesAnalyzer
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ 评估语义覆盖  │ ← SemanticEvaluator
                    └───────┬───────┘
                            │
                   ┌────────┴────────┐
                   │ 覆盖率 < 80%?   │
                   └────────┬────────┘
              Yes          │          No
        ┌──────────────┐   │   ┌──────────────┐
        │ 生成引导话术 │   │   │ 生成客户回复  │
        └──────┬───────┘   │   └──────┬───────┘
               │           │           │
               └───────────┴───────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ 返回结果     │
                    └───────────────┘
```

## 技术选型理由

### 为什么选择 LangGraph?

- **状态管理**：StateGraph 提供清晰的状态流转机制
- **条件分支**：支持复杂的条件路由逻辑
- **可视化调试**：便于理解工作流执行过程
- **可扩展性**：易于添加新节点和边

### 为什么选择 RRF + 动态加权?

- **RRF**：对多路召回结果进行公平融合，避免单一来源偏差
- **动态加权**：根据上下文（如客户类型、对话阶段）调整权重，提高检索相关性

### 为什么选择双数据库?

- **SQLite**：轻量、易部署，适合结构化数据（会话、消息、评估结果）
- **Chroma**：专为向量检索优化，适合知识库（产品知识、话术示例、异议处理）
