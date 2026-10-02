"""
轻量全链路追踪
================
基于 contextvars 的 request_id / trace_id 贯穿，与 loguru 配合：
- 在服务入口设置 trace_id（HTTP 中间件 / 后台任务入口）
- 随日志输出 trace_id 便于按请求串联排查
- 提供耗时埋点装饰器/上下文管理器，打印慢节点日志
"""
import time
import threading
import contextvars
import uuid
from functools import wraps
from typing import Optional, Callable

from app.core.logger import logger

# contextvars 保证并发请求之间 trace_id 隔离
_trace_var: contextvars.ContextVar = contextvars.ContextVar("trace_id", default="")


def current_trace_id() -> str:
    """获取当前上下文的 trace_id。"""
    return _trace_var.get()


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """设置新的 trace_id，返回设置后的值。"""
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_var.set(tid)
    return tid


def clear_trace_id() -> None:
    """清除当前上下文的 trace_id。"""
    _trace_var.set("")


class TraceContext:
    """上下文管理器：进入时生成 trace_id，退出时自动清除。"""

    def __init__(self, trace_id: Optional[str] = None):
        self._tid = trace_id

    def __enter__(self) -> str:
        return set_trace_id(self._tid)

    def __exit__(self, *exc) -> None:
        clear_trace_id()


def timed(name: Optional[str] = None, threshold_ms: Optional[int] = None):
    """
    耗时埋点装饰器：记录函数执行耗时，超过阈值打印 WARNING 慢调用日志。
    用法：
        @timed("node_search_embedding")
        def run(state): ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                cost_ms = (time.perf_counter() - start) * 1000
                label = name or func.__name__
                from app.conf.settings import settings
                th = threshold_ms if threshold_ms is not None else settings.trace_slow_threshold_ms
                if cost_ms >= th:
                    logger.warning(
                        f"[trace:{current_trace_id() or '-'}] 慢调用 {label} 耗时 {cost_ms:.0f}ms")
                else:
                    logger.debug(
                        f"[trace:{current_trace_id() or '-'}] {label} 耗时 {cost_ms:.0f}ms")
        return wrapper
    return decorator


class timing:
    """耗时埋点上下文管理器，用法：with timing("xxx", threshold_ms=200): ..."""

    def __init__(self, name: str, threshold_ms: Optional[int] = None):
        self.name = name
        self.threshold_ms = threshold_ms
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self._start is None:
            return
        cost_ms = (time.perf_counter() - self._start) * 1000
        from app.conf.settings import settings
        th = self.threshold_ms if self.threshold_ms is not None else settings.trace_slow_threshold_ms
        if cost_ms >= th:
            logger.warning(f"[trace:{current_trace_id() or '-'}] 慢调用 {self.name} 耗时 {cost_ms:.0f}ms")
        else:
            logger.debug(f"[trace:{current_trace_id() or '-'}] {self.name} 耗时 {cost_ms:.0f}ms")