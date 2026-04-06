"""Database 集成测试。

使用真实 SQLite 数据库测试，不使用mock。
"""

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def clean_env():
    """清理环境变量。"""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


class TestDatabaseIntegration:
    """Database 服务集成测试类。"""

    @pytest.fixture
    def temp_db_path(self) -> str:
        """创建临时数据库路径。"""
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_umu_sales.db")
        yield db_path
        try:
            os.unlink(db_path)
        except Exception:
            pass
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass

    @pytest.fixture
    def db_service(self, temp_db_path: str) -> "DatabaseService":
        """创建 Database 服务实例。"""
        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{temp_db_path}"
        from umu_sales_trainer.services.database import DatabaseService

        service = DatabaseService(db_path=temp_db_path)
        service.init_db()
        yield service
        try:
            service.close()
        except Exception:
            pass

    def test_init_db_real(self, temp_db_path: str) -> None:
        """测试数据库初始化。"""
        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{temp_db_path}"
        from umu_sales_trainer.services.database import DatabaseService

        service = DatabaseService(db_path=temp_db_path)
        service.init_db()
        service.close()
        assert os.path.exists(temp_db_path)

    def test_save_and_get_session_real(self, db_service: "DatabaseService") -> None:
        """测试保存和获取会话。"""
        session_id = "test-session-001"
        customer_profile = {"name": "张主任", "position": "内分泌科主任"}
        product_info = {"name": "优血糖", "type": "DPP-4抑制剂"}

        result = db_service.save_session(
            session_id=session_id,
            customer_profile=customer_profile,
            product_info=product_info,
        )
        assert result is not None
        assert result.id == session_id

        session = db_service.get_session(session_id)
        assert session is not None
        assert session.id == session_id
        assert session.customer_profile["name"] == "张主任"

    def test_save_and_get_messages_real(self, db_service: "DatabaseService") -> None:
        """测试保存和获取消息。"""
        session_id = "test-session-msg"
        db_service.save_session(
            session_id=session_id,
            customer_profile={"name": "测试"},
            product_info={"name": "测试产品"},
        )

        db_service.save_message(
            session_id=session_id,
            role="user",
            content="您好，我想了解一下这个产品",
            turn=1,
        )
        db_service.save_message(
            session_id=session_id,
            role="assistant",
            content="您好，请问有什么可以帮助您的？",
            turn=2,
        )

        messages = db_service.get_messages(session_id)
        assert len(messages) == 2

    def test_soft_delete_session_real(self, db_service: "DatabaseService") -> None:
        """测试软删除会话。"""
        session_id = "test-session-delete"
        db_service.save_session(
            session_id=session_id,
            customer_profile={"name": "测试"},
            product_info={"name": "测试产品"},
        )

        result = db_service.soft_delete_session(session_id)
        assert result is True

        deleted_session = db_service.get_session(session_id)
        assert deleted_session is None

    def test_save_coverage_record_real(self, db_service: "DatabaseService") -> None:
        """测试保存覆盖记录。"""
        session_id = "test-session-coverage"
        db_service.save_session(
            session_id=session_id,
            customer_profile={"name": "测试"},
            product_info={"name": "测试产品"},
        )

        result = db_service.save_coverage_record(
            session_id=session_id,
            point_id="SP-001",
            point_description="产品降糖效果",
            is_covered=True,
            coverage_details={"score": 0.9},
        )
        assert result is not None

    def test_multiple_sessions_real(self, db_service: "DatabaseService") -> None:
        """测试多会话管理。"""
        for i in range(3):
            session_id = f"test-session-{i}"
            db_service.save_session(
                session_id=session_id,
                customer_profile={"name": f"客户{i}"},
                product_info={"name": f"产品{i}"},
            )

        session1 = db_service.get_session("test-session-0")
        session2 = db_service.get_session("test-session-1")
        session3 = db_service.get_session("test-session-2")

        assert session1 is not None
        assert session2 is not None
        assert session3 is not None
