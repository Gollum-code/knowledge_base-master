"""
任务状态追踪（支持可选 Redis 持久化后端）
=========================================
对外保持原有函数式接口不变（add_done_task / get_task_status 等）。

存储后端：
- 默认：进程内存（单进程演示/开发，零依赖、高性能）
- 可选：Redis（配置 REDIS_URL 且 redis 库可用时自动启用），
  支持多 worker 部署与重启后状态恢复

线程安全：读-改-写临界区使用模块级锁保护。
"""
import json
import threading
import time
from time import time
from typing import Dict, List, Optional

from .sse_utils import push_to_session


def _load_task_data(task_id: str) -> Optional[dict]:
    """从后端读取任务数据（Redis 返回深拷贝副本；内存返回原引用）。"""
    if _BACKEND_REDIS:
        raw = _r.get(f"kb:task:{task_id}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None
    return _mem_tasks.get(task_id)


def _save_task_data(task_id: str, data: dict) -> None:
    if _BACKEND_REDIS:
        _r.set(f"kb:task:{task_id}", json.dumps(data, ensure_ascii=False, default=str),
               ex=settings.task_data_retention_seconds)
    # 内存模式：data 是原引用，原地修改即已生效，无需写回


def _init_backend():
    """初始化存储后端。"""
    global _BACKEND_REDIS, _r, settings
    from app.conf.settings import settings  # noqa: F811
    if settings.redis_url:
        try:
            import redis as _redis_mod
            client = _redis_mod.Redis.from_url(settings.redis_url,
                                               decode_responses=True, socket_timeout=3)
            client.ping()
            _r = client
            _BACKEND_REDIS = True
            return
        except Exception as e:
            from app.core.logger import logger
            logger.warning(f"Redis 不可用，任务状态回退内存后端：{e}")
    _BACKEND_REDIS = False
    _r = None


# 内存后端存储：task_id -> dict
_mem_tasks: Dict[str, dict] = {}
_r = None
_BACKEND_REDIS = False
_lock = threading.Lock()
settings = None

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 任务数据保留时长（秒）：超过后自动清理（内存后端懒触发清理；Redis 依赖 TTL）
TASK_DATA_RETENTION_SECONDS = 3600

# 节点名 -> 中文名映射（用于前端展示）
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_kg": "导入知识图谱",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # --- Query 流程节点 ---
    "node_item_name_confirm": "确认问题产品",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_query_kg": "查询知识图谱",
    "node_join": "多路搜索合并",
}


def _get_or_create(task_id: str) -> dict:
    """读取任务数据，不存在则创建空结构。返回值在锁内可直接原地修改。"""
    data = _load_task_data(task_id)
    if data is None:
        data = {
            "running": [],
            "done": [],
            "status": "",
            "result": {},
            "created_at": time(),
        }
        if _BACKEND_REDIS:
            # Redis 模式：先保存才能拿到唯一副本
            _save_task_data(task_id, data)
            data = _load_task_data(task_id)
        else:
            _mem_tasks[task_id] = data
    return data


def _to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    with _lock:
        data = _get_or_create(task_id)
        running = data["running"]
        if node_name not in running:
            running.append(node_name)
            _save_task_data(task_id, data)
        if is_stream:
            task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    with _lock:
        data = _get_or_create(task_id)
        data["running"] = [n for n in data["running"] if n != node_name]
        done = data["done"]
        if node_name not in done:
            done.append(node_name)
        _save_task_data(task_id, data)
        if is_stream:
            task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    with _lock:
        data = _get_or_create(task_id)
        data["result"][key] = value
        _save_task_data(task_id, data)


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    with _lock:
        data = _get_or_create(task_id)
        return data["result"].get(key, default)


def get_task_status(task_id: str) -> str:
    data = _load_task_data(task_id)
    if data is None:
        return ""
    return data.get("status", "")


def get_done_task_list(task_id: str) -> List[str]:
    data = _load_task_data(task_id)
    if data is None:
        return []
    return [_to_cn(n) for n in data.get("done", [])]


def get_running_task_list(task_id: str) -> List[str]:
    data = _load_task_data(task_id)
    if data is None:
        return []
    return [_to_cn(n) for n in data.get("running", [])]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    with _lock:
        data = _get_or_create(task_id)
        data["status"] = status_name
        _save_task_data(task_id, data)
        if not _BACKEND_REDIS:
            _cleanup_expired_tasks()
        if push_queue:
            task_push_queue(task_id)


def task_push_queue(task_id: str):
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


def clear_task(task_id: str):
    with _lock:
        _mem_tasks.pop(task_id, None)
        if _BACKEND_REDIS:
            try:
                _r.delete(f"kb:task:{task_id}")
            except Exception:
                pass


def _cleanup_expired_tasks() -> None:
    """内存后端：清理超过保留时长的任务数据。"""
    if not _mem_tasks:
        return
    now = time()
    expired = [tid for tid, data in _mem_tasks.items()
               if now - data.get("created_at", 0) > TASK_DATA_RETENTION_SECONDS]
    for tid in expired:
        _mem_tasks.pop(tid, None)


# 初始化存储后端
_init_backend()