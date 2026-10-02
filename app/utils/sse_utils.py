import json
import asyncio
from typing import Dict, Any, Optional
from fastapi import Request
from app.core.logger import logger


class SSEEvent:
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号


class _SSEQueue:
    """
    基于 asyncio.Queue 的线程安全 SSE 队列。
    - 记录创建时的 event loop，供后台线程通过 call_soon_threadsafe 安全入队
    - 读取方（sse_generator）在事件循环内 await get()，不占用线程池线程
    - 背景任务线程（BackgroundTasks/threadpool）调用 push_to_session 跨线程安全
    """

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        try:
            self.loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    def put_nowait(self, item: Any) -> None:
        """线程安全入队：有 loop 时调度到事件循环，否则直接放入。"""
        if self.loop is None or self.loop.is_running():
            self.queue.put_nowait(item)
        else:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, item)


# 全局 SSE 会话队列存储
# Key: session_id, Value: _SSEQueue
_session_stream: Dict[str, _SSEQueue] = {}


def get_sse_queue(session_id: str) -> Optional[_SSEQueue]:
    """获取指定 session 的队列"""
    return _session_stream.get(session_id)


def create_sse_queue(session_id: str) -> _SSEQueue:
    """创建并注册一个新的 SSE 队列"""
    logger.debug(f"[SSE] Creating queue for session: {session_id}")
    q = _SSEQueue()
    _session_stream[session_id] = q
    return q


def remove_sse_queue(session_id: str):
    """移除指定 session 的队列"""
    logger.debug(f"[SSE] Removing queue for session: {session_id}")
    _session_stream.pop(session_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """
    通过 session_id 推送事件（线程安全，可从后台线程调用）
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put_nowait({"event": event, "data": data})
    else:
        logger.warning(f"[SSE] Warning: No queue found for session {session_id} when pushing {event}")


async def sse_generator(session_id: str, request: Request):
    """
    SSE 生成器，用于 FastAPI 的 StreamingResponse。
    基于 asyncio.Queue，读取在事件循环内完成，不占用线程池线程。
    """
    logger.debug(f"[SSE] Generator started for session: {session_id}")
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        # 如果没有对应的队列，直接结束
        logger.warning(f"[SSE] Error: Queue not found for session {session_id}. Available sessions: {list(_session_stream.keys())}")
        return

    try:
        # 发送连接建立信号
        logger.debug(f"[SSE] Sending ready signal for {session_id}")
        yield _sse_pack("ready", {})

        while True:
            # 若客户端断开，尽快退出
            if await request.is_disconnected():
                logger.debug(f"[SSE] Client disconnected: {session_id}")
                break

            try:
                # 带超时等待，周期性检查断开状态（不占用线程池）
                msg = await asyncio.wait_for(stream_queue.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # 队列暂无新消息，等待
                continue

            event = msg.get("event")
            data = msg.get("data")

            # 特殊关闭事件
            if event == "__close__":
                logger.debug(f"[SSE] Closing signal received for {session_id}")
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        logger.debug(f"[SSE] Client disconnected (Cancelled/Reset/Pipe): {session_id}")
        # 生成器被取消/对端断开：静默退出
        return
    except Exception as e:
        logger.error(f"[SSE] Exception in generator for {session_id}: {e}")
    finally:
        logger.debug(f"[SSE] Generator finished for {session_id}")
        # 清理资源
        remove_sse_queue(session_id)