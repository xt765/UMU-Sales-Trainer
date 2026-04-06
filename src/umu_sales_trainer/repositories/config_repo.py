"""配置仓储。

从 YAML 文件加载客户画像、产品资料和知识库配置，支持缓存避免重复加载。
"""

import yaml
from pathlib import Path
from typing import Any

from umu_sales_trainer.models import CustomerProfile, ProductInfo, SellingPoint


class ConfigRepository:
    """配置仓储类。

    负责从 YAML 文件加载配置数据，包括客户画像、产品信息和知识库。
    使用缓存机制避免重复加载相同的配置文件。

    Attributes:
        _data_dir: 数据目录路径
        _customer_cache: 客户画像缓存字典
        _product_cache: 产品信息缓存字典
        _kb_cache: 知识库缓存字典
    """

    def __init__(self, data_dir: str = "./data") -> None:
        """初始化配置仓储。

        Args:
            data_dir: 数据目录路径，默认为 "./data"
        """
        self._data_dir: Path = Path(data_dir)
        self._customer_cache: dict[str, CustomerProfile] = {}
        self._product_cache: dict[str, ProductInfo] = {}
        self._kb_cache: dict[str, list[dict[str, Any]]] = {}

    def load_customer_profile(self, name: str) -> CustomerProfile:
        """加载客户画像配置。

        Args:
            name: 客户画像名称（不含扩展名）

        Returns:
            CustomerProfile: 客户画像实例

        Raises:
            FileNotFoundError: 客户画像文件不存在
            ValueError: YAML 数据格式错误
        """
        if name in self._customer_cache:
            return self._customer_cache[name]

        file_path = self._data_dir / "customers" / f"{name}.yaml"
        data = self._load_yaml(file_path)
        profile = CustomerProfile(
            industry=data.get("industry", ""),
            position=data.get("position", ""),
            concerns=data.get("concerns", []),
            personality=data.get("personality", ""),
            objection_tendencies=data.get("objection_tendencies", []),
        )
        self._customer_cache[name] = profile
        return profile

    def load_product_info(self, name: str) -> ProductInfo:
        """加载产品信息配置。

        Args:
            name: 产品名称（不含扩展名）

        Returns:
            ProductInfo: 产品信息实例

        Raises:
            FileNotFoundError: 产品信息文件不存在
            ValueError: YAML 数据格式错误
        """
        if name in self._product_cache:
            return self._product_cache[name]

        file_path = self._data_dir / "products" / f"{name}.yaml"
        data = self._load_yaml(file_path)

        selling_points: dict[str, SellingPoint] = {}
        for sp_id, sp_data in data.get("key_selling_points", {}).items():
            selling_points[sp_id] = SellingPoint(
                point_id=sp_id,
                description=sp_data.get("description", ""),
                keywords=sp_data.get("keywords", []),
                sample_phrases=sp_data.get("sample_phrases", []),
            )

        product = ProductInfo(
            name=data.get("name", name),
            description=data.get("description", ""),
            core_benefits=data.get("core_benefits", []),
            key_selling_points=selling_points,
        )
        self._product_cache[name] = product
        return product

    def load_knowledge_base(self, category: str) -> list[dict[str, Any]]:
        """加载知识库配置。

        Args:
            category: 知识库类别（不含扩展名）

        Returns:
            list[dict]: 知识库条目列表

        Raises:
            FileNotFoundError: 知识库文件不存在
            ValueError: YAML 数据格式错误
        """
        if category in self._kb_cache:
            return self._kb_cache[category]

        file_path = self._data_dir / "knowledge" / f"{category}.yaml"
        data = self._load_yaml(file_path)
        items = data.get("items", [])
        self._kb_cache[category] = items
        return items

    def _load_yaml(self, file_path: Path) -> dict[str, Any]:
        """加载并解析 YAML 文件。

        Args:
            file_path: YAML 文件路径

        Returns:
            dict: 解析后的 YAML 数据

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: YAML 解析错误
        """
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"YAML 解析错误 {file_path}: {e}") from e
