"""Database 服务模块。

提供 SQLite 数据库服务，支持会话管理、消息存储和语义点覆盖记录。
使用 SQLAlchemy 实现 ORM，集成连接池和软删除功能。
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类。

    所有数据库模型必须继承此类以获得 ORM 功能。
    """


class SoftDeleteMixin:
    """软删除混入类。

    提供软删除所需的公共字段和行为：
    - is_deleted: 标记记录是否被删除
    - deleted_at: 记录删除时间
    - deleted_by: 记录删除操作者
    """

    is_deleted: bool
    deleted_at: Optional[datetime]
    deleted_by: Optional[str]


class SessionModel(Base, SoftDeleteMixin):
    """会话数据库模型。

    存储销售训练会话的完整信息，包括客户画像、产品信息和会话状态。

    Attributes:
        id: 会话唯一标识，主键
        customer_profile: 客户画像JSON字符串
        product_info: 产品信息JSON字符串
        status: 会话状态，active/completed/abandoned
        created_at: 会话创建时间
        ended_at: 会话结束时间
        is_deleted: 是否已软删除
        deleted_at: 软删除时间
        deleted_by: 软删除操作者
    """

    __tablename__ = "sessions"

    id = Column(String(100), primary_key=True)
    customer_profile = Column(JSON, nullable=False)
    product_info = Column(JSON, nullable=False)
    status = Column(String(20), nullable=False, default="active")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    is_deleted = Column(Integer, nullable=False, default=0)
    deleted_at = Column(DateTime, nullable=True)
    deleted_by = Column(String(100), nullable=True)


class MessageModel(Base, SoftDeleteMixin):
    """消息数据库模型。

    存储会话中的对话消息，包括发送者角色、内容和时间。

    Attributes:
        id: 消息唯一标识，自增主键
        session_id: 所属会话ID，外键
        role: 消息发送者角色，user/assistant/system
        content: 消息内容文本
        turn: 消息轮次序号
        created_at: 消息创建时间
        is_deleted: 是否已软删除
        deleted_at: 软删除时间
        deleted_by: 软删除操作者
    """

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    turn = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_deleted = Column(Integer, nullable=False, default=0)
    deleted_at = Column(DateTime, nullable=True)
    deleted_by = Column(String(100), nullable=True)


class CoverageRecordModel(Base, SoftDeleteMixin):
    """语义点覆盖记录数据库模型。

    记录会话中每个语义点的覆盖状态和评估详情。

    Attributes:
        id: 记录唯一标识，自增主键
        session_id: 所属会话ID，外键
        point_id: 语义点标识
        point_description: 语义点描述
        is_covered: 是否被覆盖
        coverage_details: 覆盖详情JSON
        created_at: 记录创建时间
        is_deleted: 是否已软删除
        deleted_at: 软删除时间
        deleted_by: 软删除操作者
    """

    __tablename__ = "coverage_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    point_id = Column(String(50), nullable=False)
    point_description = Column(Text, nullable=True)
    is_covered = Column(Integer, nullable=False, default=0)
    coverage_details = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_deleted = Column(Integer, nullable=False, default=0)
    deleted_at = Column(DateTime, nullable=True)
    deleted_by = Column(String(100), nullable=True)


class PendingOperationModel(Base, SoftDeleteMixin):
    """待处理补偿操作数据库模型。

    存储需要执行的补偿操作记录，如知识库更新等。

    Attributes:
        id: 操作唯一标识，自增主键
        operation_type: 操作类型
        payload: 操作载荷JSON
        status: 操作状态，pending/processing/completed/failed
        retry_count: 重试次数
        error_message: 错误信息
        created_at: 创建时间
        processed_at: 处理完成时间
        is_deleted: 是否已软删除
        deleted_at: 软删除时间
        deleted_by: 软删除操作者
    """

    __tablename__ = "pending_operations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    operation_type = Column(String(50), nullable=False)
    payload = Column(JSON, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    retry_count = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    is_deleted = Column(Integer, nullable=False, default=0)
    deleted_at = Column(DateTime, nullable=True)
    deleted_by = Column(String(100), nullable=True)


class DatabaseService:
    """SQLite 数据库服务类。

    提供会话管理、消息存储和语义点覆盖记录等数据库操作。
    支持连接池、事务管理和软删除功能。

    Attributes:
        engine: SQLAlchemy 数据库引擎
        session_factory: Session工厂，用于创建数据库会话
    """

    def __init__(self, db_path: str = "umu_sales.db") -> None:
        """初始化数据库服务。

        Args:
            db_path: 数据库文件路径，默认为 "umu_sales.db"
        """
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
        self.session_factory = sessionmaker(bind=self.engine)

    def init_db(self) -> None:
        """初始化数据库表。

        创建所有定义的数据库表，包括sessions、messages、
        coverage_records和pending_operations。
        """
        Base.metadata.create_all(self.engine)

    @contextmanager
    def create_session(self) -> Session:
        """创建数据库会话的上下文管理器。

        Yields:
            Session: SQLAlchemy 数据库会话，自动处理提交和回滚
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        turn: int,
    ) -> MessageModel:
        """保存对话消息。

        Args:
            session_id: 所属会话ID
            role: 消息发送者角色
            content: 消息内容
            turn: 消息轮次

        Returns:
            保存的消息模型实例
        """
        message = MessageModel(
            session_id=session_id,
            role=role,
            content=content,
            turn=turn,
        )
        with self.create_session() as session:
            session.add(message)
            session.flush()
            session.refresh(message)
            result = MessageModel(
                id=message.id,
                session_id=message.session_id,
                role=message.role,
                content=message.content,
                turn=message.turn,
                created_at=message.created_at,
            )
        return result

    def get_session(self, session_id: str) -> Optional[SessionModel]:
        """获取会话记录。

        Args:
            session_id: 会话ID

        Returns:
            会话模型实例，未找到或已软删除返回None
        """
        with self.create_session() as session:
            result = session.query(SessionModel).filter(
                SessionModel.id == session_id,
                SessionModel.is_deleted == 0,
            ).first()
            if result:
                return SessionModel(
                    id=result.id,
                    customer_profile=result.customer_profile,
                    product_info=result.product_info,
                    status=result.status,
                    created_at=result.created_at,
                    ended_at=result.ended_at,
                )
            return None

    def get_messages(
        self,
        session_id: str,
        include_deleted: bool = False,
    ) -> list[MessageModel]:
        """获取会话消息历史。

        Args:
            session_id: 会话ID
            include_deleted: 是否包含已删除消息，默认False

        Returns:
            消息模型实例列表，按轮次排序
        """
        with self.create_session() as session:
            query = session.query(MessageModel).filter(
                MessageModel.session_id == session_id,
            )
            if not include_deleted:
                query = query.filter(MessageModel.is_deleted == 0)
            results = query.order_by(MessageModel.turn).all()
            return [
                MessageModel(
                    id=r.id,
                    session_id=r.session_id,
                    role=r.role,
                    content=r.content,
                    turn=r.turn,
                    created_at=r.created_at,
                )
                for r in results
            ]

    def save_session(
        self,
        session_id: str,
        customer_profile: dict[str, Any],
        product_info: dict[str, Any],
        status: str = "active",
    ) -> SessionModel:
        """保存会话记录。

        Args:
            session_id: 会话唯一标识
            customer_profile: 客户画像字典
            product_info: 产品信息字典
            status: 会话状态，默认 "active"

        Returns:
            保存的会话模型实例
        """
        session_model = SessionModel(
            id=session_id,
            customer_profile=customer_profile,
            product_info=product_info,
            status=status,
        )
        with self.create_session() as session:
            session.add(session_model)
            session.flush()
            session.refresh(session_model)
            result = SessionModel(
                id=session_model.id,
                customer_profile=session_model.customer_profile,
                product_info=session_model.product_info,
                status=session_model.status,
                created_at=session_model.created_at,
                ended_at=session_model.ended_at,
            )
        return result

    def soft_delete_session(
        self,
        session_id: str,
        deleted_by: Optional[str] = None,
    ) -> bool:
        """软删除会话及其关联消息。

        Args:
            session_id: 会话ID
            deleted_by: 删除操作者标识

        Returns:
            是否成功软删除
        """
        deleted_at = datetime.utcnow()
        with self.create_session() as session:
            session.query(SessionModel).filter(
                SessionModel.id == session_id,
            ).update({
                "is_deleted": 1,
                "deleted_at": deleted_at,
                "deleted_by": deleted_by,
            })
            session.query(MessageModel).filter(
                MessageModel.session_id == session_id,
            ).update({
                "is_deleted": 1,
                "deleted_at": deleted_at,
                "deleted_by": deleted_by,
            })
        return True

    def save_coverage_record(
        self,
        session_id: str,
        point_id: str,
        is_covered: bool,
        point_description: Optional[str] = None,
        coverage_details: Optional[dict[str, Any]] = None,
    ) -> CoverageRecordModel:
        """保存语义点覆盖记录。

        Args:
            session_id: 所属会话ID
            point_id: 语义点标识
            is_covered: 是否被覆盖
            point_description: 语义点描述
            coverage_details: 覆盖详情

        Returns:
            保存的覆盖记录模型实例
        """
        record = CoverageRecordModel(
            session_id=session_id,
            point_id=point_id,
            point_description=point_description,
            is_covered=1 if is_covered else 0,
            coverage_details=coverage_details,
        )
        with self.create_session() as session:
            session.add(record)
            session.flush()
            session.refresh(record)
            result = CoverageRecordModel(
                id=record.id,
                session_id=record.session_id,
                point_id=record.point_id,
                point_description=record.point_description,
                is_covered=record.is_covered,
                coverage_details=record.coverage_details,
                created_at=record.created_at,
            )
        return result


_db_service: Optional[DatabaseService] = None


def get_db_service() -> DatabaseService:
    """获取数据库服务单例。

    Returns:
        DatabaseService实例
    """
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service
