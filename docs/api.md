# API 参考文档

本文档详细描述 AI Sales Trainer Chatbot 的所有 RESTful API 端点，包括请求/响应格式、字段说明和错误处理。所有端点均以 `/api/v1` 为前缀。

## 目录

- [1. 基础信息](#1-基础信息)
- [2. 核心端点：发送消息](#2-核心端点发送消息)
- [3. 会话管理端点](#3-会话管理端点)
- [4. 辅助端点](#4-辅助端点)
- [5. 数据模型参考](#5-数据模型参考)
- [6. 错误处理](#6-错误处理)

---

## 1. 基础信息

| 属性 | 值 |
|------|---|
| Base URL | `http://localhost:8000/api/v1` |
| 数据格式 | JSON (Content-Type: application/json) |
| 认证方式 | 无（本地工具，不暴露到公网） |
| 编码 | UTF-8 |

---

## 2. 核心端点：发送消息

这是系统最核心的端点，触发完整的 Agentic RAG 工作流。

### 2.1 端点定义

```
POST /api/v1/sessions/{session_id}/messages
```

### 2.2 请求体

**模型**: `SendMessageRequest`

| 字段 | 类型 | 必填 | 约束 | 说明 |
|------|------|:----:|------|------|
| `content` | string | 是 | min_length=1 | 销售人员输入的话术内容 |

**请求示例**:

```json
{
  "content": "张主任您好，我这边有一款新型降糖药想跟您介绍一下，它在HbA1c控制方面有显著优势，而且安全性数据也很好。"
}
```

### 2.3 响应体

**模型**: `SendMessageResponse`

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话唯一标识 |
| `turn` | int | 当前轮次（从 1 开始递增） |
| `ai_response` | string | AI 客户模拟器的回复文本 |
| `evaluation` | object | 完整评估结果（见下方结构） |
| `guidance` | object \| null | 引导建议（优秀态时仍返回对象，is_actionable=false） |

#### evaluation 对象结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `coverage_status` | object | 各语义点的覆盖状态，key 为 point_id，value 为 `"covered"` 或 `"not_covered"` |
| `coverage_labels` | object | 语义点 ID 到中文描述的映射 |
| `coverage_rate` | float | 总覆盖率（0.0 - 1.0） |
| `overall_score` | float/int | 综合评分（0 - 100） |
| `expression_analysis` | object | 三维表达评分 |
| `suggestions` | array | 改进建议列表（仅包含低于 7 分的维度） |
| `conversation_analysis` | object \| null | 对话分析结果 |

**expression_analysis 子结构**:

| 字段 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `clarity` | int | 1-10 | 清晰度评分 |
| `professionalism` | int | 1-10 | 专业性评分 |
| `persuasiveness` | int | 1-10 | 说服力评分 |

**suggestions 数组元素结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `dimension` | string | 维度标识：`"clarity"` / `"professionalism"` / `"persuasiveness"` |
| `current_score` | int | 当前分数（1-10） |
| `advice` | string | 改进建议文字 |
| `example` | string | 参考话术范例 |

**conversation_analysis 子结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `stage` | string | 销售阶段标识 |
| `intent` | string | 意图描述 |
| `objections` | array | 检测到的异议类型列表 |
| `sentiment` | string | 情感倾向 |
| `confidence` | float | 分析置信度 |

#### guidance 对象结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `summary` | string | 一句话总结 |
| `is_actionable` | boolean | 是否需要立即行动（false = 优秀态） |
| `overall_score` | float/int | 综合评分（用于优秀态显示） |
| `priority_list` | array | 按紧急度排序的引导项列表（优秀态时为空数组） |

**priority_list 元素结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `gap` | string | 缺失或不足方面的描述 |
| `urgency` | string | 紧急程度：`"high"` / `"medium"` / `"low"` |
| `suggestion` | string | 具体改进建议 |
| `talking_point` | string | 参考话术范例 |
| `expected_effect` | string | 预期效果说明 |

### 2.4 完整响应示例

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "turn": 3,
  "ai_response": "听起来这款产品在疗效方面确实有数据支撑。不过我想了解一下，对于老年患者或者有并发症的患者，使用这个产品的安全性数据怎么样？有没有特殊人群的亚组分析？",
  "evaluation": {
    "coverage_status": { "SP-001": "covered", "SP-002": "covered", "SP-003": "not_covered" },
    "coverage_labels": { "SP-001": "产品介绍", "SP-002": "疗效效果", "SP-003": "安全性" },
    "coverage_rate": 0.67,
    "overall_score": 45,
    "expression_analysis": { "clarity": 7, "professionalism": 6, "persuasiveness": 5 },
    "suggestions": [
      { "dimension": "persuasiveness", "current_score": 5, "advice": "采用'痛点-方案-证据-行动'四步法构建论证逻辑链", "example": "您提到的XX问题确实存在（痛点），我们的方案是XX（方案），临床证明XX（证据），建议先试用（行动）。" }
    ],
    "conversation_analysis": {
      "stage": "presentation",
      "intent": "呈现产品特点和疗效",
      "objections": [],
      "sentiment": "positive",
      "confidence": 0.8
    }
  },
  "guidance": {
    "summary": "有1项急需改进（共2项），建议优先处理标红项目。",
    "is_actionable": true,
    "overall_score": 45,
    "priority_list": [
      {
        "gap": "未充分覆盖：安全性",
        "urgency": "high",
        "suggestion": "在下次发言中主动提及安全性相关的内容",
        "talking_point": "关于安全性，我想特别强调的是...",
        "expected_effect": "提升语义点覆盖率，当前 67% → 目标 100%"
      },
      {
        "gap": "说服力偏低（5/10分）",
        "urgency": "high",
        "suggestion": "采用'痛点-方案-证据-行动'四步法构建论证逻辑链",
        "talking_point": "您提到的XX问题确实存在（痛点），我们的方案是XX...",
        "expected_effect": "提升说服力至7分以上"
      }
    ]
  }
}
```

### 2.5 调用示例

**curl**:

```bash
curl -X POST "http://localhost:8000/api/v1/sessions/{session_id}/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "产品介绍话术内容"}'
```

**Python (httpx)**:

```python
import httpx

response = httpx.post(
    f"http://localhost:8000/api/v1/sessions/{session_id}/messages",
    json={"content": "产品介绍话术内容"},
)
data = response.json()
print(data["ai_response"])
print(f"Score: {data['evaluation']['overall_score']}")
```

---

## 3. 会话管理端点

### 3.1 创建会话

```
POST /api/v1/sessions
```

**请求体 (`CreateSessionRequest`)**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `customer_profile` | object | 是 | 客户画像信息（name, hospital, position, personality_type 等） |
| `product_info` | object | 是 | 产品信息（name, description, core_benefits, key_selling_points 等） |
| `semantic_points` | array \| 否 | 语义点列表（可选，不传则从 product_info 自动生成） |

**响应 (`CreateSessionResponse`, 201)**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 新创建的会话 ID |
| `status` | string | 会话状态（`"active"`） |
| `created_at` | datetime (ISO 8601) | 创建时间 |

**请求示例**:

```json
{
  "customer_profile": {
    "name": "张主任",
    "hospital": "某三甲医院",
    "position": "内分泌科主任",
    "personality_type": "ANALYTICAL"
  },
  "product_info": {
    "name": "XX降糖药",
    "description": "新型SGLT2抑制剂",
    "core_benefits": [
      "SP_EFFICACY: 强效降糖，HbA1c降低达1.2%",
      "SP_SAFETY: 心血管获益，降低心衰风险",
      "SP_CONVENIENCE: 一日一次口服，依从性高"
    ],
    "key_selling_points": {}
  }
}
```

### 3.2 获取评估结果

```
GET /api/v1/sessions/{session_id}/evaluation
```

重新运行工作流并返回最新评估结果。响应格式同 `SendMessageResponse.evaluation` 字段，类型为 `EvaluationResponse`。

### 3.3 删除会话

```
DELETE /api/v1/sessions/{session_id}?hard=false
```

支持两种删除模式：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hard` | false | true = 物理删除（不可恢复）；false = 软删除（标记状态） |

响应: `204 No Content`

### 3.4 清空所有会话

```
DELETE /api/v1/sessions?hard=true
```

物理删除全部会话及其关联消息和覆盖记录。

### 3.5 获取会话状态

```
GET /api/v1/sessions/{session_id}/status
```

**响应 (`SessionStatusResponse`)**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |
| `status` | string | 会话状态 |
| `created_at` | datetime | 创建时间 |
| `message_count` | int | 已交换的消息数量 |

### 3.6 获取所有会话列表

```
GET /api/v1/sessions
```

**响应 (`SessionListResponse`)**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `sessions` | array | 会话列表（含 session_id / status / created_at / message_count） |
| `total` | int | 总数 |

### 3.7 获取会话消息历史

```
GET /api/v1/sessions/{session_id}/messages
```

**响应 (`MessagesListResponse`)**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |
| `messages` | array | 消息列表（id / role / content / turn / created_at） |
| `total` | int | 总消息数 |

---

## 4. 辅助端点

### 4.1 健康检查

```
GET /api/v1/health
```

**响应 (`HealthResponse`)**:

```json
{
  "status": "healthy",
  "timestamp": "2026-01-15T10:30:00Z"
}
```

用于监控探活和负载均衡健康检查。

---

## 5. 数据模型参考

### 5.1 端点总览图

```mermaid
graph LR
    subgraph 会话生命周期
        C1[POST /sessions] --> S[Session 存在]
        S --> M[POST /sessions/id/messages]
        M --> M2[POST ... (重复)]
        S --> E[GET /sessions/id/evaluation]
        S --> ST[GET /sessions/id/status]
        S --> H[GET /sessions/id/messages]
        S --> D[DELETE /sessions/id]
        DA[DELETE /sessions] --> D
    end

    subgraph 无状态
        HL[GET /health]
    end

    style C1 fill:#c8e6c9
    style M fill:#bbdefb
    style E fill:#e1f5fe
    style D fill:#ffcdd2
    style HL fill:#f5f5f5
```

### 5.2 客户画像字段映射

前端传入的 `customer_profile` 字典会被转换为 `CustomerProfile` 对象：

| 前端字段 | CustomerProfile 属性 | 默认值（缺失时） |
|---------|---------------------|---------------|
| `name` | name | `""` |
| `hospital` | hospital | `""` |
| `position` | position | `""` |
| `personality_type` | personality | 按 ANALYTICAL/DRIVER/EXPRESSIVE/AMIABLE 映射为中文描述 |
| `concerns` | concerns | 根据 position 从预定义表选取（如内分泌科主任 -> HbA1c/低血糖/依从性） |
| `objection_tendencies` | objection_tendencies | `[]` |

### 5.3 性格类型映射表

| `personality_type` 值 | 生成的 personality 描述 |
|----------------------|------------------------|
| `ANALYTICAL` | 专业严谨，注重数据和循证医学证据 |
| `DRIVER` | 果断直接，注重效率和结果 |
| `EXPRESSIVE` | 热情开放，注重创新和关系 |
| `AMIABLE` | 温和谨慎，注重安全和信任 |

---

## 6. 错误处理

### 6.1 HTTP 状态码

| 状态码 | 场景 | 响应体 |
|--------|------|--------|
| 201 | 会话创建成功 | `CreateSessionResponse` |
| 200 | 其他成功请求 | 各端点对应 Response 模型 |
| 204 | 删除成功 | 无响应体 |
| 400 | 请求参数无效 | `{"detail": "错误描述"}` |
| 404 | 会话不存在 | `{"detail": "Session {id} not found"}` |
| 500 | 服务内部错误 | `{"detail": "Failed to create session"}` 或类似 |

### 6.2 常见错误场景

| 错误 | HTTP 状态码 | `detail` 内容 |
|------|------------|-------------|
| session_id 不存在 | 404 | `Session {uuid} not found` |
| content 为空字符串 | 422 | Field required / min_length=1 |
| LLM 服务不可用 | 500 | 内部错误（AI 回复降级为兜底文本） |
| 数据库写入失败 | 500 | `Failed to create session` / `Failed to save message` |
