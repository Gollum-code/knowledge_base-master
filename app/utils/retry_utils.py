"""
通用失败重试工具
==================
对「瞬时失败」（HTTP 429/5xx、网络超时、连接错误）提供指数退避重试，
用于 LLM / MinerU / MCP 等外部依赖调用，提升鲁棒性。

用法：
    from app.utils.retry_utils import with_retry, retry_on
    result = with_retry(fn, args=[...], kwargs={...})

    @retry_on(max_attempts=3)
    def call_api(x): ...
"""
import time
import random
import logging
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, Union

from app.conf.settings import settings

logger = logging.getLogger("app.retry")


def _is_retryable_exception(exc: Exception, retry_codes: Optional[Tuple[int, ...]] = None) -> bool:
    """判断异常是否为可重试类型（网络/超时/限流）。"""
    import requests
    name = type(exc).__name__
    if isinstance(exc, requests.exceptions.RequestException):
        return True
    # httpx / openai 常见错误
    if "Timeout" in name or "ConnectionError" in name or "APIConnectionError" in name \
            or "RateLimitError" in name or "InternalServerError" in name:
        return True
    if retry_codes and isinstance(exc, Exception):
        # openai APIStatusError / APIConnectionError 携带 status_code
        code = getattr(exc, "status_code", None)
        if code in retry_codes:
            return True
    return False


def with_retry(
        func: Callable,
        *,
        max_attempts: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        retry_codes: Optional[Tuple[int, ...]] = None,
        args: Optional[Tuple] = None,
        kwargs: Optional[dict] = None,
) -> Any:
    """
    同步重试包装：对 func(*args, **kwargs) 进行指数退避重试。
    返回 func 的返回值；重试耗尽后抛出最后一次异常。
    """
    attempts = max_attempts or settings.retry_max_attempts
    delay = base_delay if base_delay is not None else settings.retry_base_delay
    cap = max_delay if max_delay is not None else settings.retry_max_delay
    codes = retry_codes or tuple(settings.retry_http_codes)

    args = args or ()
    kwargs = kwargs or {}
    last_exc: Optional[Exception] = None

    for i in range(attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if i == attempts - 1 or not _is_retryable_exception(e, codes):
                break
            sleep_time = min(delay * (2 ** i), cap) + random.uniform(0, 0.2)
            logger.warning(
                f"调用 {getattr(func, '__name__', func)} 失败（{type(e).__name__}: {e}），"
                f"{sleep_time:.1f}s 后进行第 {i + 2}/{attempts} 次重试")
            time.sleep(sleep_time)
    raise last_exc  # type: ignore[misc]


def retry_on(max_attempts: Optional[int] = None, base_delay: Optional[float] = None,
             retry_codes: Optional[Tuple[int, ...]] = None):
    """装饰器版重试：@retry_on(max_attempts=3)"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return with_retry(
                func,
                max_attempts=max_attempts,
                base_delay=base_delay,
                retry_codes=retry_codes,
                args=args,
                kwargs=kwargs,
            )
        return wrapper
    return decorator