"""数据库初始化脚本。

提供命令行工具初始化 SQLite 数据库表结构，支持强制重建和 dry-run 模式。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqlalchemy import create_engine, inspect

from umu_sales_trainer.services.database import Base


def get_existing_tables(engine) -> set[str]:
    """获取数据库中已存在的表名。

    Args:
        engine: SQLAlchemy 数据库引擎

    Returns:
        已存在表名的集合
    """
    inspector = inspect(engine)
    return set(inspector.get_table_names())


def print_sql_statements(tables: list[str]) -> None:
    """打印建表 SQL 语句。

    Args:
        tables: 表名列表
    """
    print("=== DRY-RUN: 以下 SQL 语句将被执行 ===\n")
    for table in tables:
        print(f"CREATE TABLE IF NOT EXISTS {table} (...)")
    print("\n=== DRY-RUN 结束 ===")


def init_database(db_path: str, force: bool, dry_run: bool) -> None:
    """初始化数据库。

    Args:
        db_path: 数据库文件路径
        force: 是否强制重建（删除并重建所有表）
        dry_run: 是否仅显示 SQL 而不执行
    """
    engine = create_engine(f"sqlite:///{db_path}")

    if dry_run:
        table_names = [cls.__tablename__ for cls in Base.__subclasses__()]
        print_sql_statements(table_names)
        return

    existing_tables = get_existing_tables(engine)

    if existing_tables and not force:
        print(f"数据库已存在表: {', '.join(existing_tables)}")
        print("使用 --force 参数强制重建，或使用 --dry-run 查看将执行的 SQL。")
        return

    if force and existing_tables:
        print(f"强制重建: 删除现有表 {existing_tables}")
        Base.metadata.drop_all(engine)
        print("现有表已删除。")

    print("创建数据库表...")
    Base.metadata.create_all(engine)

    new_tables = get_existing_tables(engine)
    print(f"数据库初始化完成。新建表: {', '.join(new_tables)}")


def main() -> None:
    """命令行入口点。"""
    parser = argparse.ArgumentParser(
        description="初始化 SQLite 数据库表结构",
    )
    parser.add_argument(
        "--db-path",
        default="umu_sales.db",
        help="数据库文件路径 (默认: umu_sales.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建：删除现有表并重新创建",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将执行的 SQL，不实际创建表",
    )
    args = parser.parse_args()

    init_database(args.db_path, args.force, args.dry_run)


if __name__ == "__main__":
    main()
