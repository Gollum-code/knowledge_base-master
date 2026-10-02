"""
可选 Redis 状态存储
====================
提供 task/sse 状态的持久化存储后端：
- 未配置 REDIS_URL 或 redis 未安装：回退为纯内存实现（与原先一致，兼容单进程）
- 配置了 REDIS_URL 且 redis 可用：使用 Redis，支持多 worker / 重启后状态恢复

对外暴露统一接口（get/set/delete/keys/expire），上层（task_utils 等）按此对接，
因此从内存切换到 Redis 不需要改动业务代码。
"""
import json
import time
from typing import Any, Optional

from app.conf.settings import settings

# 是否启用了 Redis 后端
_REDIS_AVAILABLE = False
_r = None


def _init_redis():
    """尝试初始化 Redis 客户端。失败时回退内存并置 _REDIS_AVAILABLE=False。"""
    global _REDIS_AVAILABLE, _r
    if not settings.redis_url:
        return
    try:
        import redis  # type: ignore
        client = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=3)
        client.ping()
        _r = client
        _REDIS_AVAILABLE = True
    except Exception as e:
        from app.core.logger import logger
        logger.warning(f"Redis 连接失败，回退为内存状态存储：{e}")
        _REDIS_AVAILABLE = False
        _r = None


# 内存后端存储
_MEM: dict = {}


def _get(key: str) -> Optional[Any]:
    if _REDIS_AVAILABLE:
        raw = _r.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return raw
    return _MEM.get(key)


def _set(key: str, value: Any, ttl: Optional[int] = None) -> None:
    if _REDIS_AVAILABLE:
        raw = json.dumps(value, ensure_ascii=False, default=str)
        _r.set(key, raw, ex=ttl or settings.task_data_retention_seconds)
        return
    _MEM[key] = value


def _delete(key: str) -> None:
    if _REDIS_AVAILABLE:
        _r.delete(key)
        return
    _MEM.pop(key, None)


def _scan_keys(pattern: str) -> list:
    if _REDIS_AVAILABLE:
        return [k for k in _r.scan_iter(pattern)]
    return _MEM.keys()


def _exists(key: str) -> bool:
    if _REDIS_AVAILABLE:
        return bool(_r.exists(key))
    return key in _MEM


def _expire(key: str, ttl: int) -> None:
    if _REDIS_AVAILABLE:
        _r.expire(key, ttl)


# ---- 高级便捷封装：按 task_id 管理一组 key ----

def _namespace(prefix: str, task_id: str) -> str:
    return f"kb:{prefix}:{task_id}"


def get_task_field(prefix: str, task_id: str, field: str, default: Any = None) -> Any:
    """示例：get_task_field('status', task_id, 'status')"""
    value = _get(_namespace(prefix, task_id))
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(field, default)
    return value


def set_task_field(prefix: str, task_id: str, field: str, value: Any) -> None:
    key = _namespace(prefix, task_id)
    data = _get(key)
    if not isinstance(data, dict):
        data = {}
    data[field] = value
    _set(key, data)


# 初始化（模块加载时尝试，失败自动回退内存）
_init_redis()