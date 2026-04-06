"""评估结果数据模型。

定义销售对话评估的结果结构，包括语义点覆盖情况和表达分析。
"""

from dataclasses import dataclass, field


@dataclass
class ExpressionAnalysis:
    """表达分析结果。

    对销售人员表达能力的评估，包含清晰度、专业性和说服力三个维度。

    Attributes:
        clarity: 表达清晰度评分，1-10分
        professionalism: 表达专业性评分，1-10分
        persuasiveness: 表达说服力评分，1-10分
    """

    clarity: int = 0
    professionalism: int = 0
    persuasiveness: int = 0


@dataclass
class EvaluationResult:
    """评估结果。

    存储销售对话的完整评估结果，包括语义点覆盖状态和表达能力分析。

    Attributes:
        session_id: 所属会话ID
        coverage_status: 语义点覆盖状态字典，key为point_id，value为"covered"或"not_covered"
        expression_analysis: 表达能力分析结果
        coverage_rate: 语义点覆盖率，0.0-1.0
        overall_score: 综合评分，0.0-100.0
    """

    session_id: str
    coverage_status: dict[str, str] = field(default_factory=dict)
    expression_analysis: ExpressionAnalysis = field(default_factory=ExpressionAnalysis)
    coverage_rate: float = 0.0
    overall_score: float = 0.0
