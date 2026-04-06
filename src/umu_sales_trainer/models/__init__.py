"""数据模型包。

提供销售训练系统中使用的数据模型，包括客户画像、产品信息、
语义点、会话管理和评估结果等。
"""

from umu_sales_trainer.models.customer import CustomerProfile
from umu_sales_trainer.models.product import ProductInfo, SellingPoint
from umu_sales_trainer.models.semantic import SemanticPoint
from umu_sales_trainer.models.conversation import ConversationSession, Message
from umu_sales_trainer.models.evaluation import EvaluationResult, ExpressionAnalysis

__all__ = [
    "CustomerProfile",
    "ProductInfo",
    "SellingPoint",
    "SemanticPoint",
    "ConversationSession",
    "Message",
    "EvaluationResult",
    "ExpressionAnalysis",
]
