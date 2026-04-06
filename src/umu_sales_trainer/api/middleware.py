"""中间件模块。

提供 API 中间件功能：日志记录、限流、Token 统计。
使用 Starlette MiddlewareMixin 实现。
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from starlette.applications import Starlette

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件。

    记录每个 HTTP 请求的 method、path、status code 和处理时长。
    用于监控 API 调用的基本指标。

    Attributes:
        log_format: 日志格式化字符串

    Example:
        >>> app.add_middleware(LoggingMiddleware)
    """

    log_format: str

    def __init__(self, app: "Starlette") -> None:
        """初始化日志中间件。

        Args:
            app: Starlette 应用实例
        """
        super().__init__(app)
        self.log_format = "%(method)s %(path)s %(status)d %(duration).3fs"

    async def dispatch(self, request: Request, call_next: Callable[[Any], Any]) -> Response:
        """处理请求并记录日志。

        Args:
            request: HTTP 请求对象
            call_next: 下一个处理器

        Returns:
            HTTP 响应对象
        """
        start_time = time.perf_counter()
        method = request.method
        path = request.url.path

        response = await call_next(request)

        duration = time.perf_counter() - start_time
        status = response.status_code

        logger.info(
            self.log_format,
            extra={"method": method, "path": path, "status": status, "duration": duration},
        )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于 IP 的限流中间件。

    使用简单内存存储实现滑动窗口限流，默认每分钟 60 请求。
    可后续扩展支持 Redis 等分布式存储。

    Attributes:
        max_requests: 时间窗口内最大请求数
        window_seconds: 时间窗口秒数

    Example:
        >>> app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)
    """

    max_requests: int
    window_seconds: int

    def __init__(
        self, app: "Starlette", max_requests: int = 60, window_seconds: int = 60
    ) -> None:
        """初始化限流中间件。

        Args:
            app: Starlette 应用实例
            max_requests: 时间窗口内最大请求数
            window_seconds: 时间窗口秒数
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[datetime]] = defaultdict(list)
        self._lock = Lock()

    def _get_client_ip(self, request: Request) -> str:
        """获取客户端 IP 地址。

        支持通过 X-Forwarded-For 头获取真实 IP。

        Args:
            request: HTTP 请求对象

        Returns:
            客户端 IP 地址
        """
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, ip: str) -> bool:
        """检查 IP 是否超过限流阈值。

        Args:
            ip: 客户端 IP 地址

        Returns:
            是否被限流
        """
        with self._lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_seconds)

            self._requests[ip] = [
                req_time for req_time in self._requests[ip] if req_time > window_start
            ]

            if len(self._requests[ip]) >= self.max_requests:
                return True

            self._requests[ip].append(now)
            return False

    async def dispatch(self, request: Request, call_next: Callable[[Any], Any]) -> Response:
        """处理请求并进行限流检查。

        Args:
            request: HTTP 请求对象
            call_next: 下一个处理器

        Returns:
            HTTP 响应对象 或 429 状态码
        """
        ip = self._get_client_ip(request)

        if self._is_rate_limited(ip):
            logger.warning(f"Rate limit exceeded for IP: {ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
            )

        return await call_next(request)


class TokenCountMiddleware(BaseHTTPMiddleware):
    """Token 使用量统计中间件。

    统计请求和响应的 token 使用量，记录到日志。
    支持从请求头读取 prompt_tokens，从响应体读取 completion_tokens。

    Attributes:
        total_prompt_tokens: 累计 prompt tokens
        total_completion_tokens: 累计 completion tokens
        total_tokens: 累计总 tokens

    Example:
        >>> app.add_middleware(TokenCountMiddleware)
    """

    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int

    def __init__(self, app: "Starlette") -> None:
        """初始化 Token 统计中间件。

        Args:
            app: Starlette 应用实例
        """
        super().__init__(app)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self._lock = Lock()

    async def dispatch(self, request: Request, call_next: Callable[[Any], Any]) -> Response:
        """处理请求并统计 Token 使用量。

        Args:
            request: HTTP 请求对象
            call_next: 下一个处理器

        Returns:
            HTTP 响应对象
        """
        prompt_tokens = 0
        completion_tokens = 0

        prompt_header = request.headers.get("x-prompt-tokens", "")
        if prompt_header.isdigit():
            prompt_tokens = int(prompt_header)

        response = await call_next(request)

        if hasattr(response, "body") and response.body:
            try:
                body = response.body
                if isinstance(body, bytes):
                    body = body.decode("utf-8")
                data: dict[str, Any] = json.loads(body)
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, AttributeError):
                pass

        if prompt_tokens or completion_tokens:
            with self._lock:
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += prompt_tokens + completion_tokens

            logger.info(
                "Token usage: prompt=%d, completion=%d, total=%d (cumulative: %d)",
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
                self.total_prompt_tokens + self.total_completion_tokens,
            )

        return response


class TokenStats:
    """Token 统计数据类。

    存储累计的 token 使用统计信息。

    Attributes:
        total_prompt_tokens: 累计 prompt tokens
        total_completion_tokens: 累计 completion tokens
        total_tokens: 累计总 tokens
        request_count: 累计请求数
    """

    def __init__(self) -> None:
        """初始化 Token 统计。"""
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.request_count: int = 0

    def get_stats(self) -> tuple[int, int, int, int]:
        """获取当前统计信息。

        Returns:
            (prompt_tokens, completion_tokens, total_tokens, request_count) 元组
        """
        return (
            self.total_prompt_tokens,
            self.total_completion_tokens,
            self.total_tokens,
            self.request_count,
        )
