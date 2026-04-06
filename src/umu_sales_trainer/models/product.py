"""产品信息数据模型。

定义产品相关的核心卖点和产品信息结构。
"""

from dataclasses import dataclass, field


@dataclass
class SellingPoint:
    """销售卖点。

    表示产品或服务的某个核心卖点，包含描述、关键词和示例话术。

    Attributes:
        point_id: 卖点唯一标识，如"SP-001"
        description: 卖点的文字描述
        keywords: 关键词列表，用于语义匹配
        sample_phrases: 示例话术列表，帮助销售人员理解如何表达
    """

    point_id: str
    description: str
    keywords: list[str] = field(default_factory=list)
    sample_phrases: list[str] = field(default_factory=list)


@dataclass
class ProductInfo:
    """产品信息。

    存储产品的基本信息、核心优势和所有销售卖点。

    Attributes:
        name: 产品名称
        description: 产品详细描述
        core_benefits: 核心优势列表
        key_selling_points: 关键卖点字典，key为卖点ID
    """

    name: str
    description: str = ""
    core_benefits: list[str] = field(default_factory=list)
    key_selling_points: dict[str, SellingPoint] = field(default_factory=dict)
