"""语义点（评估标准）数据模型。

定义销售对话评估中的语义点，用于衡量对话内容的覆盖度和质量。
"""

from dataclasses import dataclass, field


@dataclass
class SemanticPoint:
    """语义点（评估标准）。

    表示销售对话中需要评估的某个语义维度，如某个卖点是否被提及、
    某个异议是否被妥善处理等。

    Attributes:
        point_id: 语义点唯一标识，如"SP-001"
        description: 语义点的文字描述
        keywords: 关键词列表，用于语义匹配和识别
        weight: 权重值，用于计算综合评分，默认为1.0
    """

    point_id: str
    description: str
    keywords: list[str] = field(default_factory=list)
    weight: float = 1.0
