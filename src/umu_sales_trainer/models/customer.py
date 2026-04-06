"""客户画像数据模型。

定义销售场景中客户的相关信息，包括行业、职位、关注点、性格和异议倾向等。
"""

from dataclasses import dataclass, field


@dataclass
class CustomerProfile:
    """客户画像数据模型。

    用于存储销售对话中客户的基本信息，帮助销售人员更好地理解客户需求。

    Attributes:
        name: 客户姓名，如"张医生"、"李主任"等
        hospital: 客户所在机构/医院
        industry: 客户所在行业，如"医疗"、"金融"等
        position: 客户职位，如"内分泌科主任"、"采购经理"等
        concerns: 客户关注点列表，如["价格", "质量", "售后"]等
        personality: 客户性格描述，如"谨慎型"、"果断型"等
        objection_tendencies: 客户可能的异议倾向列表
    """

    name: str = ""
    hospital: str = ""
    industry: str = ""
    position: str = ""
    concerns: list[str] = field(default_factory=list)
    personality: str = ""
    objection_tendencies: list[str] = field(default_factory=list)
