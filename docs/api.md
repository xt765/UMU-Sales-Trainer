# API 文档

本文档详细介绍 UMU Sales Trainer 系统的 RESTful API 接口规范。

---

## 基础信息

| 属性 | 值 |
|------|-----|
| **Base URL** | `http://localhost:8000/api/v1` |
| **Content-Type** | `application/json` |
| **字符编码** | UTF-8 |
| **认证方式** | 无（当前版本） |

---

## 认证说明

> ⚠️ **当前版本未实现认证机制**，API 处于开放状态。
>
> 生产环境部署时，请务必添加适当的认证机制（如 API Key、JWT 等）。

---

## 请求格式

所有请求必须包含以下 Header：

```http
Content-Type: application/json
Accept: application/json
```

---

## 响应格式

### 成功响应

```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### 错误响应

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述",
    "details": { ... }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## 接口列表

### 会话管理

| 方法 | 路径 | 描述 |
|------|------|------|
| `POST` | `/sessions` | 创建新会话 |
| `GET` | `/sessions/{id}` | 获取会话详情 |
| `GET` | `/sessions/{id}/status` | 获取会话状态 |
| `DELETE` | `/sessions/{id}` | 删除会话 |

### 消息交互

| 方法 | 路径 | 描述 |
|------|------|------|
| `POST` | `/sessions/{id}/messages` | 发送消息 |
| `GET` | `/sessions/{id}/messages` | 获取消息历史 |

### 评估分析

| 方法 | 路径 | 描述 |
|------|------|------|
| `GET` | `/sessions/{id}/evaluation` | 获取评估结果 |

### 系统

| 方法 | 路径 | 描述 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/health/ready` | 就绪检查 |

---

## 会话管理接口

### 1. 创建会话

创建一个新的 AI 销售训练会话。

```http
POST /api/v1/sessions
```

**请求体：**

```json
{
  "customer_profile_id": "endocrinologist",
  "product_id": "hypoglycemic_drug",
  "config": {
    "max_turns": 15,
    "temperature": 0.7
  }
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `customer_profile_id` | string | 是 | - | 客户画像 ID |
| `product_id` | string | 是 | - | 产品 ID |
| `config.max_turns` | integer | 否 | 15 | 最大对话轮次 |
| `config.temperature` | float | 否 | 0.7 | LLM 温度参数 |

**成功响应 (201 Created)：**

```json
{
  "success": true,
  "data": {
    "session_id": "sess_abc123xyz",
    "status": "active",
    "customer_profile": {
      "id": "endocrinologist",
      "name": "内分泌科主任",
      "position": "内分泌科主任",
      "concerns": ["糖尿病控制率", "药物安全性"]
    },
    "product_info": {
      "id": "hypoglycemic_drug",
      "name": "某降糖药",
      "core_benefits": ["HbA1c改善", "低血糖风险低", "一周一次"]
    },
    "config": {
      "max_turns": 15,
      "temperature": 0.7
    },
    "created_at": "2024-01-01T12:00:00Z",
    "turn": 0
  }
}
```

**错误响应：**

```json
{
  "success": false,
  "error": {
    "code": "INVALID_PROFILE",
    "message": "客户画像不存在",
    "details": {
      "customer_profile_id": "endocrinologist"
    }
  }
}
```

---

### 2. 获取会话详情

获取指定会话的完整信息。

```http
GET /api/v1/sessions/{session_id}
```

**路径参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 ID |

**成功响应 (200 OK)：**

```json
{
  "success": true,
  "data": {
    "session_id": "sess_abc123xyz",
    "status": "active",
    "customer_profile": { ... },
    "product_info": { ... },
    "messages": [
      {
        "id": "msg_001",
        "role": "user",
        "content": "您好张主任...",
        "turn": 1,
        "created_at": "2024-01-01T12:00:00Z"
      },
      {
        "id": "msg_002",
        "role": "assistant",
        "content": "您好，请问你介绍的是什么产品？",
        "turn": 1,
        "created_at": "2024-01-01T12:00:01Z"
      }
    ],
    "created_at": "2024-01-01T12:00:00Z",
    "turn": 1
  }
}
```

---

### 3. 获取会话状态

获取会话的实时状态信息。

```http
GET /api/v1/sessions/{session_id}/status
```

**成功响应 (200 OK)：**

```json
{
  "success": true,
  "data": {
    "session_id": "sess_abc123xyz",
    "status": "active",
    "turn": 3,
    "pending_points": ["SP-001", "SP-002"],
    "is_session_active": true,
    "created_at": "2024-01-01T12:00:00Z",
    "last_activity_at": "2024-01-01T12:05:00Z"
  }
}
```

**会话状态值：**

| 状态 | 说明 |
|------|------|
| `active` | 会话进行中 |
| `completed` | 正常结束 |
| `aborted` | 被中断 |
| `expired` | 已过期 |

---

### 4. 删除会话

软删除指定会话（数据可恢复）。

```http
DELETE /api/v1/sessions/{session_id}
```

**成功响应 (200 OK)：**

```json
{
  "success": true,
  "data": {
    "session_id": "sess_abc123xyz",
    "status": "deleted",
    "deleted_at": "2024-01-01T12:10:00Z"
  }
}
```

> **软删除说明**：会话数据不会立即物理删除，而是标记为 `is_deleted=TRUE`。可通过后台任务恢复或永久删除。

---

## 消息交互接口

### 5. 发送消息

发送销售发言，获取 AI 客户的分析和回复。

```http
POST /api/v1/sessions/{session_id}/messages
```

**请求体：**

```json
{
  "content": "您好张主任，我是某制药公司的销售代表，想向您介绍一下我们的新产品。",
  "end_session": false
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `content` | string | 是 | - | 销售发言内容（1-2000字符） |
| `end_session` | boolean | 否 | false | 是否结束会话 |

**成功响应 (200 OK)：**

```json
{
  "success": true,
  "data": {
    "message_id": "msg_xyz789",
    "session_id": "sess_abc123xyz",
    "turn": 1,
    "ai_response": {
      "content": "您好，请问你介绍的是什么产品？主要适用于哪些患者？",
      "customer_mood": "neutral"
    },
    "analysis": {
      "key_points": ["开场白", "产品介绍"],
      "expression_quality": {
        "clarity": 8,
        "professionalism": 7,
        "persuasiveness": 6
      },
      "coverage_status": {
        "SP-001": {
          "status": "covered",
          "confidence": 0.92
        },
        "SP-002": {
          "status": "not_covered",
          "confidence": 0.85
        },
        "SP-003": {
          "status": "pending",
          "confidence": 0.60
        }
      }
    },
    "pending_points": ["SP-002", "SP-003"],
    "guidance": {
      "message": "建议您介绍一下该药的降糖效果和低血糖风险情况。",
      "strategy": "direct_question"
    },
    "is_session_active": true,
    "created_at": "2024-01-01T12:00:00Z"
  }
}
```

**覆盖状态值：**

| 状态 | 说明 |
|------|------|
| `covered` | 已覆盖 |
| `not_covered` | 未覆盖 |
| `pending` | 待引导 |

---

### 6. 获取消息历史

获取会话的所有消息历史。

```http
GET /api/v1/sessions/{session_id}/messages
```

**查询参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `limit` | integer | 50 | 返回消息数量上限 |
| `offset` | integer | 0 | 消息偏移量 |

**成功响应 (200 OK)：**

```json
{
  "success": true,
  "data": {
    "session_id": "sess_abc123xyz",
    "messages": [
      {
        "id": "msg_001",
        "role": "user",
        "content": "您好张主任...",
        "turn": 1,
        "analysis": { ... },
        "created_at": "2024-01-01T12:00:00Z"
      },
      {
        "id": "msg_002",
        "role": "assistant",
        "content": "您好，请问你介绍的是什么产品？",
        "turn": 1,
        "created_at": "2024-01-01T12:00:01Z"
      }
    ],
    "total": 10,
    "limit": 50,
    "offset": 0
  }
}
```

---

## 评估分析接口

### 7. 获取评估结果

获取会话的完整评估报告。

```http
GET /api/v1/sessions/{session_id}/evaluation
```

**成功响应 (200 OK)：**

```json
{
  "success": true,
  "data": {
    "session_id": "sess_abc123xyz",
    "session_status": "completed",
    "total_turns": 8,
    "duration_seconds": 320,
    "coverage_result": {
      "overall_rate": 1.0,
      "points": {
        "SP-001": {
          "status": "covered",
          "first_mentioned_turn": 2,
          "evidence": "用户提到：'该药物可以显著降低HbA1c达1.5%'"
        },
        "SP-002": {
          "status": "covered",
          "first_mentioned_turn": 4,
          "evidence": "用户提到：'低血糖风险很低'"
        },
        "SP-003": {
          "status": "covered",
          "first_mentioned_turn": 6,
          "evidence": "用户提到：'一周只需给药一次'"
        }
      }
    },
    "expression_quality": {
      "clarity": 8.5,
      "professionalism": 8.0,
      "persuasiveness": 7.5
    },
    "guidance_history": [
      {
        "turn": 3,
        "pending_point": "SP-001",
        "guidance_message": "您提到降糖效果，具体 HbA1c 能降低多少？",
        "was_addressed": true
      }
    ],
    "overall_score": 85,
    "grade": "A",
    "strengths": [
      "开场白得体，专业性强",
      "完整传达了三大核心卖点",
      "表达清晰，有数据支持"
    ],
    "areas_for_improvement": [
      "可以增加更多临床案例分享",
      "建议补充与竞品的对比"
    ],
    "recommendations": [
      "继续保持当前表达结构",
      "建议增加产品差异化优势的说明"
    ]
  }
}
```

**评分等级：**

| 等级 | 分数范围 | 说明 |
|------|----------|------|
| A | 90-100 | 优秀 |
| B | 80-89 | 良好 |
| C | 70-79 | 及格 |
| D | 60-69 | 待提升 |
| F | <60 | 不及格 |

---

## 系统接口

### 8. 健康检查

检查服务是否正常运行。

```http
GET /api/v1/health
```

**成功响应 (200 OK)：**

```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 3600,
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

---

### 9. 就绪检查

检查服务是否已准备好接受请求。

```http
GET /api/v1/health/ready
```

**成功响应 (200 OK)：**

```json
{
  "success": true,
  "data": {
    "ready": true,
    "checks": {
      "database": "ok",
      "chroma": "ok",
      "llm_service": "ok"
    }
  }
}
```

---

## 错误响应

### 错误码对照表

| HTTP 状态码 | 错误码 | 说明 |
|-------------|--------|------|
| `400` | `INVALID_REQUEST` | 请求参数错误 |
| `400` | `VALIDATION_ERROR` | 数据校验失败 |
| `400` | `SESSION_INACTIVE` | 会话已结束 |
| `401` | `UNAUTHORIZED` | 未授权（预留） |
| `404` | `SESSION_NOT_FOUND` | 会话不存在 |
| `404` | `MESSAGE_NOT_FOUND` | 消息不存在 |
| `422` | `INVALID_CONTENT` | 内容不符合要求 |
| `429` | `RATE_LIMIT_EXCEEDED` | 请求频率超限 |
| `500` | `INTERNAL_ERROR` | 服务器内部错误 |
| `503` | `SERVICE_UNAVAILABLE` | 服务不可用 |

### 400 Bad Request

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数校验失败",
    "details": {
      "field": "content",
      "reason": "消息内容不能为空"
    }
  }
}
```

### 404 Not Found

```json
{
  "success": false,
  "error": {
    "code": "SESSION_NOT_FOUND",
    "message": "会话不存在或已删除",
    "details": {
      "session_id": "sess_invalid"
    }
  }
}
```

### 429 Too Many Requests

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "请求频率超限，请稍后再试",
    "details": {
      "retry_after_seconds": 60
    }
  }
}
```

### 500 Internal Server Error

```json
{
  "success": false,
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "服务器内部错误，请联系管理员",
    "details": {
      "request_id": "req_abc123"
    }
  }
}
```

---

## 速率限制

| 端点 | 限制 | 窗口 |
|------|------|------|
| `/health` | 60 请求 | 1 分钟 |
| `/sessions` | 20 请求 | 1 分钟 |
| `/sessions/{id}/messages` | 30 请求 | 1 分钟 |
| 其他端点 | 60 请求 | 1 分钟 |

---

## 使用示例

### cURL

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 创建会话
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"customer_profile_id": "endocrinologist", "product_id": "hypoglycemic_drug"}'

# 发送消息
curl -X POST http://localhost:8000/api/v1/sessions/sess_abc123xyz/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "您好张主任，我是某制药公司的销售代表..."}'

# 获取评估
curl http://localhost:8000/api/v1/sessions/sess_abc123xyz/evaluation

# 删除会话
curl -X DELETE http://localhost:8000/api/v1/sessions/sess_abc123xyz
```

### Python (httpx)

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000/api/v1", timeout=30.0)

# 创建会话
response = client.post("/sessions", json={
    "customer_profile_id": "endocrinologist",
    "product_id": "hypoglycemic_drug"
})
session_id = response.json()["data"]["session_id"]

# 发送消息
response = client.post(f"/sessions/{session_id}/messages", json={
    "content": "您好张主任，我是某制药公司的销售代表..."
})
print(response.json()["data"]["ai_response"]["content"])

# 获取评估
response = client.get(f"/sessions/{session_id}/evaluation")
print(f"总分: {response.json()['data']['overall_score']}")

# 关闭会话
client.close()
```

### JavaScript (fetch)

```javascript
const BASE_URL = 'http://localhost:8000/api/v1';

const api = {
  async createSession(customerProfileId, productId) {
    const response = await fetch(`${BASE_URL}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        customer_profile_id: customerProfileId,
        product_id: productId
      })
    });
    return response.json();
  },

  async sendMessage(sessionId, content) {
    const response = await fetch(`${BASE_URL}/sessions/${sessionId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content })
    });
    return response.json();
  },

  async getEvaluation(sessionId) {
    const response = await fetch(`${BASE_URL}/sessions/${sessionId}/evaluation`);
    return response.json();
  }
};

// 使用示例
const { data: { session_id } } = await api.createSession('endocrinologist', 'hypoglycemic_drug');
const { data } = await api.sendMessage(session_id, '您好张主任，我是某制药公司的销售代表...');
console.log(data.ai_response.content);
```

---

## Changelog

### v1.0.0 (2024-01-01)

- 初始版本
- 支持会话管理、消息交互、评估分析
- 支持 DashScope 和 DeepSeek 双 Provider
