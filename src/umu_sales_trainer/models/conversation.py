"""对话会话数据模型。

定义销售训练场景中的对话消息和会话结构。
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class Message:
    """对话消息。

    表示销售对话中的单条消息，包含发送者角色和内容。

    Attributes:
        session_id: 所属会话ID
        role: 消息发送者角色，"user"表示客户/学员，"assistant"表示AI销售员
        content: 消息内容文本
        turn: 消息轮次序号，从1开始
        created_at: 消息创建时间，默认为当前时间
    """

    session_id: str
    role: str
    content: str
    turn: int
    created_at: Optional[datetime] = None


@dataclass
class ConversationSession:
    """会话。

    表示一次完整的销售训练会话，包含客户画像、产品信息和会话状态。

    Attributes:
        id: 会话唯一标识
        customer_profile: 客户画像JSON字符串
        product_info: 产品信息JSON字符串
        status: 会话状态，"active"表示进行中，"completed"表示已完成
        created_at: 会话创建时间
        ended_at: 会话结束时间
    """

    id: str
    customer_profile: str
    product_info: str
    status: str = "active"
    created_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
