# API 文档

本文档详细描述 UMU Sales Trainer 系统的所有 API 接口。

## 基础信息

- **Base URL**: `http://localhost:8000/api/v1`
- **Content-Type**: `application/json`

## 接口列表

### 1. 健康检查

检查服务是否正常运行。

**请求**

```
GET /api/v1/health
```

**响应**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### 2. 创建会话

创建一个新的训练会话。

**请求**

```
POST /api/v1/sessions
Content-Type: application/json

{
  "customer_id": "endocrinologist",
  "product_id": "hypoglycemic_drug"
}
```

**响应**

```json
{
  "session_id": "sess_abc123",
  "status": "active",
  "created_at": "2024-01-01T12:00:00"
}
```

---

### 3. 发送消息

发送销售发言，获取 AI 客户的分析和回复。

**请求**

```
POST /api/v1/sessions/{session_id}/messages
Content-Type: application/json

{
  "content": "您好张主任，我是某制药公司的销售代表，想向您介绍一下我们的新产品。"
}
```

**响应**

```json
{
  "message_id": "msg_xyz789",
  "ai_response": "您好，请问你介绍的是什么产品？主要适用于哪些患者？",
  "analysis": {
    "key_points": ["开场白", "产品介绍"],
    "expression_quality": {
      "clarity": 8,
      "professionalism": 7,
      "persuasiveness": 6
    },
    "coverage_rate": 0.45
  },
  "guidance": "建议您先说明产品名称和类别，然后再介绍具体特点。",
  "turn": 1
}
```

---

### 4. 获取评估

获取当前会话的完整评估结果。

**请求**

```
GET /api/v1/sessions/{session_id}/evaluation
```

**响应**

```json
{
  "session_id": "sess_abc123",
  "overall_score": 78,
  "coverage_rate": 0.65,
  "expression_quality": {
    "clarity": 8,
    "professionalism": 7,
    "persuasiveness": 8
  },
  "semantic_points": [
    {
      "point_id": "SP-001",
      "description": "产品名称和类别",
      "is_covered": true,
      "importance": 0.9
    },
    {
      "point_id": "SP-002",
      "description": "降糖效果数据",
      "is_covered": false,
      "importance": 0.85
    }
  ],
  "conversation_summary": {
    "total_turns": 5,
    "ai_questions": ["产品名称", "适应症", "价格"],
    "uncovered_concerns": ["价格", "副作用"]
  }
}
```

---

### 5. 获取会话状态

获取会话的基本信息。

**请求**

```
GET /api/v1/sessions/{session_id}/status
```

**响应**

```json
{
  "session_id": "sess_abc123",
  "status": "active",
  "created_at": "2024-01-01T12:00:00",
  "turn_count": 3,
  "last_message_at": "2024-01-01T12:05:00"
}
```

---

### 6. 删除会话

软删除会话（可恢复）。

**请求**

```
DELETE /api/v1/sessions/{session_id}
```

**响应**

```json
{
  "message": "会话已删除",
  "session_id": "sess_abc123"
}
```

## 错误响应

### 400 Bad Request

请求参数错误。

```json
{
  "detail": "Invalid request parameters",
  "errors": {
    "content": "Message content cannot be empty"
  }
}
```

### 404 Not Found

会话不存在。

```json
{
  "detail": "Session not found"
}
```

### 429 Too Many Requests

请求频率超限。

```json
{
  "detail": "Rate limit exceeded. Please wait 60 seconds."
}
```

### 500 Internal Server Error

服务器内部错误。

```json
{
  "detail": "Internal server error"
}
```

## 速率限制

| 端点 | 限制 |
|------|------|
| 所有端点 | 60 请求/分钟 |

## 使用示例

### cURL

```bash
# 创建会话
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "endocrinologist", "product_id": "hypoglycemic_drug"}'

# 发送消息
curl -X POST http://localhost:8000/api/v1/sessions/sess_abc123/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "您好张主任..."}'

# 获取评估
curl http://localhost:8000/api/v1/sessions/sess_abc123/evaluation
```

### Python

```python
import httpx

base_url = "http://localhost:8000/api/v1"

# 创建会话
response = httpx.post(f"{base_url}/sessions", json={
    "customer_id": "endocrinologist",
    "product_id": "hypoglycemic_drug"
})
session_id = response.json()["session_id"]

# 发送消息
response = httpx.post(
    f"{base_url}/sessions/{session_id}/messages",
    json={"content": "您好张主任，我是某制药公司的销售代表..."}
)
print(response.json()["ai_response"])

# 获取评估
response = httpx.get(f"{base_url}/sessions/{session_id}/evaluation")
print(response.json()["overall_score"])
```

### JavaScript

```javascript
const baseUrl = 'http://localhost:8000/api/v1';

// 创建会话
const sessionRes = await fetch(`${baseUrl}/sessions`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    customer_id: 'endocrinologist',
    product_id: 'hypoglycemic_drug'
  })
});
const { session_id } = await sessionRes.json();

// 发送消息
const messageRes = await fetch(`${baseUrl}/sessions/${session_id}/messages`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ content: '您好张主任...' })
});
const { ai_response } = await messageRes.json();
console.log(ai_response);
```
