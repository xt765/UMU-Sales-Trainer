"""销售发言分析器模块。

分析销售发言，提取关键信息点，评估表达质量和语义点覆盖情况。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from umu_sales_trainer.models.evaluation import ExpressionAnalysis
from umu_sales_trainer.models.semantic import SemanticPoint
from umu_sales_trainer.services.llm import LLMService

if TYPE_CHECKING:
    from umu_sales_trainer.models.conversation import Message
    from umu_sales_trainer.models.customer import CustomerProfile
    from umu_sales_trainer.models.product import ProductInfo


class SalesAnalyzer:
    """销售发言分析器。

    通过 LLM 分析销售人员的发言，评估表达质量（清晰度、专业性、说服力）
    和语义点覆盖情况，生成结构化的分析结果。

    Attributes:
        llm_service: LLM 服务实例，用于调用大语言模型进行分析

    Example:
        >>> analyzer = SalesAnalyzer(llm_service)
        >>> result = analyzer.analyze(
        ...     sales_message="我们的产品采用先进的技术...",
        ...     context={"customer_profile": profile, ...}
        ... )
    """

    def __init__(self, llm_service: LLMService) -> None:
        """初始化销售发言分析器。

        Args:
            llm_service: LLM 服务实例，用于生成分析结果
        """
        self._llm = llm_service

    def analyze(self, sales_message: str, context: dict[str, Any]) -> dict[str, Any]:
        """分析销售发言。

        分析销售人员的发言内容，提取关键信息点，评估表达质量，
        并检查语义点覆盖情况。

        Args:
            sales_message: 销售人员的发言内容
            context: 分析上下文，包含客户画像、产品信息、对话历史和目标语义点

        Returns:
            结构化分析结果字典，包含以下字段：
            - key_information_points: 关键信息点列表
            - expression_analysis: 表达分析结果（clarity、professionalism、persuasiveness）
            - coverage_status: 语义点覆盖状态字典
            - coverage_rate: 语义点覆盖率（0.0-1.0）
        """
        customer_profile: CustomerProfile | None = context.get("customer_profile")
        product_info: ProductInfo | None = context.get("product_info")
        conversation_history: list[Message] | None = context.get("conversation_history")
        semantic_points: list[SemanticPoint] = context.get("semantic_points", [])

        prompt = self._build_analysis_prompt(
            sales_message, customer_profile, product_info,
            conversation_history, semantic_points
        )
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return self._parse_analysis_response(response.content, semantic_points)

    def _build_analysis_prompt(
        self,
        sales_message: str,
        customer_profile: CustomerProfile | None,
        product_info: ProductInfo | None,
        conversation_history: list[Message] | None,
        semantic_points: list[SemanticPoint],
    ) -> str:
        """构建分析提示词。

        Args:
            sales_message: 销售发言
            customer_profile: 客户画像
            product_info: 产品信息
            conversation_history: 对话历史
            semantic_points: 目标语义点

        Returns:
            格式化的提示词字符串
        """
        customer_info = ""
        if customer_profile:
            customer_info = f"""
客户信息：
- 行业：{customer_profile.industry}
- 职位：{customer_profile.position}
- 关注点：{', '.join(customer_profile.concerns)}
- 性格：{customer_profile.personality}
"""

        product_info_str = ""
        if product_info:
            product_info_str = f"""
产品信息：
- 产品名称：{product_info.name}
- 产品描述：{product_info.description}
- 核心优势：{', '.join(product_info.core_benefits)}
"""

        history_str = ""
        if conversation_history:
            history_str = "\n对话历史：\n" + "\n".join(
                f"[{msg.role}] {msg.content}" for msg in conversation_history[-5:]
            )

        semantic_str = ""
        if semantic_points:
            semantic_str = "\n目标语义点：\n" + "\n".join(
                f"- {sp.point_id}: {sp.description} (关键词: {', '.join(sp.keywords)})"
                for sp in semantic_points
            )

        return f"""你是一位专业的销售培训分析师。请分析以下销售发言：

销售发言：
{sales_message}
{customer_info}
{product_info_str}
{history_str}
{semantic_str}

请以 JSON 格式返回分析结果，包含以下字段：
{{
    "key_information_points": ["关键信息点1", "关键信息点2", ...],
    "expression_analysis": {{
        "clarity": 评分(1-10),
        "professionalism": 评分(1-10),
        "persuasiveness": 评分(1-10)
    }},
    "coverage_status": {{
        "SP-001": "covered"或"not_covered",
        ...
    }}
}}

只返回 JSON，不要有其他内容。"""

    def _parse_analysis_response(
        self,
        response_content: str,
        semantic_points: list[SemanticPoint],
    ) -> dict[str, Any]:
        """解析 LLM 响应内容。

        Args:
            response_content: LLM 返回的原始响应内容
            semantic_points: 目标语义点列表

        Returns:
            结构化分析结果字典
        """
        import json

        content = response_content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {
                "key_information_points": [],
                "expression_analysis": {"clarity": 0, "professionalism": 0, "persuasiveness": 0},
                "coverage_status": {},
            }

        key_points = data.get("key_information_points", [])
        expr = data.get("expression_analysis", {})
        coverage = data.get("coverage_status", {})

        for sp in semantic_points:
            if sp.point_id not in coverage:
                coverage[sp.point_id] = "not_covered"

        covered_count = sum(1 for v in coverage.values() if v == "covered")
        total_count = len(semantic_points) if semantic_points else 1
        coverage_rate = covered_count / total_count

        return {
            "key_information_points": key_points,
            "expression_analysis": ExpressionAnalysis(
                clarity=expr.get("clarity", 0),
                professionalism=expr.get("professionalism", 0),
                persuasiveness=expr.get("persuasiveness", 0),
            ),
            "coverage_status": coverage,
            "coverage_rate": coverage_rate,
        }