"""知识库初始化脚本。

从 data/knowledge/*.yaml 加载知识库数据，向量化和存储到 Chroma Collections。
支持 --force 强制重建和 --dry-run 只显示将要加载的数据。
"""

import sys

import click

from src.umu_sales_trainer.repositories.config_repo import ConfigRepository
from src.umu_sales_trainer.services.chroma import ChromaService
from src.umu_sales_trainer.services.embedding import EmbeddingService


COLLECTIONS: dict[str, str] = {
    "objection_handling": "异议处理知识库",
    "product_knowledge": "产品知识知识库",
    "excellent_samples": "优秀话术示例知识库",
}

DEFAULT_DATA_DIR = "./data"
DEFAULT_CHROMA_DIR = "./chroma_data"


def _build_doc_text(category: str, item: dict) -> str:
    """构建文档文本。

    Args:
        category: 文档类别
        item: 知识库条目字典

    Returns:
        拼接后的文档文本
    """
    lines = [f"类别: {category}"]
    if "scenario" in item:
        lines.append(f"场景: {item['scenario']}")
    if "title" in item:
        lines.append(f"标题: {item['title']}")
    if "question" in item:
        lines.append(f"问题: {item['question']}")
    if "answer" in item:
        lines.append(f"答案: {item['answer']}")
    if "objection" in item:
        lines.append(f"异议: {item['objection']}")
    if "analysis" in item:
        lines.append(f"分析: {item['analysis']}")
    if "response_strategy" in item:
        strategies = item["response_strategy"]
        if isinstance(strategies, list):
            lines.append("应答策略:")
            for s in strategies:
                lines.append(f"  - {s}")
        else:
            lines.append(f"应答策略: {strategies}")
    if "sample_response" in item:
        lines.append(f"应答示例: {item['sample_response']}")
    if "sample_dialogue" in item:
        lines.append(f"对话示例: {item['sample_dialogue']}")
    if "key_points" in item:
        points = item["key_points"]
        if isinstance(points, list):
            lines.append("关键要点:")
            for p in points:
                lines.append(f"  - {p}")
        else:
            lines.append(f"关键要点: {points}")
    return "\n".join(lines)


def _load_collection_data(
    config_repo: ConfigRepository, collection_name: str
) -> list[tuple[str, dict]]:
    """加载指定 Collection 的知识库数据。

    Args:
        config_repo: 配置仓储实例
        collection_name: Collection 名称

    Returns:
        (文档文本, 元数据) 元组列表
    """
    yaml_key = collection_name
    raw_items = config_repo.loadKnowledgeBase(yaml_key)
    if not raw_items:
        return []

    items = raw_items if isinstance(raw_items, list) else []
    if not items:
        return []

    if isinstance(items[0], dict) and yaml_key in items[0]:
        items = items[0][yaml_key]

    results: list[tuple[str, dict]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        doc_text = _build_doc_text(item.get("category", ""), item)
        metadata = {
            "category": item.get("category", ""),
            "collection": collection_name,
            "index": idx,
        }
        results.append((doc_text, metadata))
    return results


def _init_collection(
    chroma_service: ChromaService,
    embedding_service: EmbeddingService,
    collection_name: str,
    config_repo: ConfigRepository,
) -> int:
    """初始化单个 Collection。

    Args:
        chroma_service: Chroma 服务实例
        embedding_service: Embedding 服务实例
        collection_name: Collection 名称
        config_repo: 配置仓储实例

    Returns:
        添加的文档数量
    """
    collection = chroma_service.create_collection(collection_name)
    docs_data = _load_collection_data(config_repo, collection_name)

    if not docs_data:
        return 0

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    doc_metadatas: list[dict] = []

    for idx, (text, metadata) in enumerate(docs_data):
        doc_id = f"{collection_name}_{idx}"
        doc_ids.append(doc_id)
        doc_texts.append(text)
        doc_metadatas.append(metadata)

    collection.add(documents=doc_texts, metadatas=doc_metadatas, ids=doc_ids)
    return len(doc_ids)


def init_collections(
    data_dir: str = DEFAULT_DATA_DIR,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """初始化所有知识库 Collections。

    Args:
        data_dir: 数据目录路径
        chroma_dir: Chroma 持久化目录
        force: 是否强制重建（删除现有 Collection）
        dry_run: 是否只显示将要加载的数据

    Returns:
        Collection 名称到文档数量的映射
    """
    chroma_service = ChromaService(persist_directory=chroma_dir)
    embedding_service = EmbeddingService()
    config_repo = ConfigRepository(data_dir=data_dir)

    results: dict[str, int] = {}

    for collection_name in COLLECTIONS:
        if force and collection_name in chroma_service.list_collections():
            chroma_service.delete_collection(collection_name)
            click.echo(f"  已删除现有 Collection: {collection_name}")

        if dry_run:
            docs_data = _load_collection_data(config_repo, collection_name)
            click.echo(f"\n[DRY-RUN] {COLLECTIONS[collection_name]} ({collection_name}):")
            for idx, (text, metadata) in enumerate(docs_data):
                preview = text[:100] + "..." if len(text) > 100 else text
                click.echo(f"  文档 {idx}: {preview}")
            click.echo(f"  共 {len(docs_data)} 条文档")
            results[collection_name] = len(docs_data)
        else:
            count = _init_collection(
                chroma_service, embedding_service, collection_name, config_repo
            )
            results[collection_name] = count
            click.echo(f"  已加载 {count} 条文档到 {collection_name}")

    return results


@click.command()
@click.option(
    "--data-dir",
    default=DEFAULT_DATA_DIR,
    help="数据目录路径",
    type=click.Path(),
)
@click.option(
    "--chroma-dir",
    default=DEFAULT_CHROMA_DIR,
    help="Chroma 持久化目录",
    type=click.Path(),
)
@click.option(
    "--force",
    is_flag=True,
    help="强制重建，删除现有 Collection 后重新创建",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="只显示将要加载的数据，不实际写入",
)
def main(
    data_dir: str,
    chroma_dir: str,
    force: bool,
    dry_run: bool,
) -> None:
    """知识库初始化脚本。

    从 data/knowledge/*.yaml 加载知识库数据，向量化和存储到 Chroma。
    """
    click.echo("=" * 60)
    click.echo("知识库初始化脚本")
    click.echo("=" * 60)
    click.echo(f"数据目录: {data_dir}")
    click.echo(f"Chroma 目录: {chroma_dir}")
    click.echo(f"Force 模式: {force}")
    click.echo(f"Dry-run 模式: {dry_run}")
    click.echo(f"Collections: {', '.join(COLLECTIONS.keys())}")
    click.echo("-" * 60)

    try:
        results = init_collections(
            data_dir=data_dir,
            chroma_dir=chroma_dir,
            force=force,
            dry_run=dry_run,
        )

        click.echo("-" * 60)
        if dry_run:
            click.echo("DRY-RUN 完成，共将加载以下文档:")
        else:
            click.echo("初始化完成:")
        for name, count in results.items():
            click.echo(f"  {COLLECTIONS[name]}: {count} 条文档")
        click.echo("=" * 60)

    except FileNotFoundError as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
