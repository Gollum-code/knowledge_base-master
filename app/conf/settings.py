"""
集中配置层（pydantic-settings）
===============================
统一管理新增的可调参数（重试、批量、配额、追踪、可选 Redis 等），
并通过 get_settings() 提供全局唯一实例（懒加载 + 缓存）。

设计说明：
- 保留 app/conf 下原有 dataclass 配置（向后兼容，不破坏既有导入）。
- 新功能参数统一在这里集中加载与校验，避免散落各处 load_dotenv。
- 使用 pydantic-settings（项目 .venv 已安装），未安装时自动回退 os.getenv。
"""
import os
from functools import lru_cache
from typing import Optional, List

from dotenv import load_dotenv

# 确保 .env 只被集中加载一次（幂等）
load_dotenv(override=False)

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class _Settings(BaseSettings):
        """全项目可调参数（均带默认值，缺省不阻断启动）"""

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # ---- 失败重试 ----
        retry_max_attempts: int = 3            # 最大重试次数
        retry_base_delay: float = 1.0          # 指数退避基础间隔（秒）
        retry_max_delay: float = 15.0          # 退避上限（秒）
        retry_http_codes: List[int] = [429, 500, 502, 503, 504]

        # ---- BGE-M3 批量向量化 ----
        bge_batch_size: int = 5                # 每批向量化文本数（按显存调整）

        # ---- 上下文窗口配额（字符） ----
        context_max_chars: int = 12000         # Prompt 参考内容总字符上限
        context_docs_ratio: float = 0.7        # 文档部分占用配额比例
        context_history_ratio: float = 0.3     # 历史部分占用配额比例

        # ---- 可选 Redis 状态存储 ----
        redis_url: Optional[str] = None        # 例：redis://127.0.0.1:6379/0
        task_data_retention_seconds: int = 3600

        # ---- 全链路追踪 ----
        trace_enabled: bool = True
        trace_slow_threshold_ms: int = 500     # 超过该耗时打印慢节点日志

        # ---- 检索阈值（可配置调参） ----
        item_confirm_high_score: float = 0.85  # 商品确认高置信度阈值
        item_confirm_low_score: float = 0.6    # 候选商品置信度阈值
        milvus_min_cosine_score: float = 0.75  # 检索最低余弦分（未用过滤时仅作日志）

        # ---- MinIO 桶策略 ----
        minio_public_prefix: str = "/upload-images"  # 仅该前缀下的对象公开只读

    _USE_PYDANTIC_SETTINGS = True

except ImportError:  # pragma: no cover - 未安装 pydantic-settings 时回退
    class _Settings:  # type: ignore
        """轻量回退实现：直接从环境变量读取"""

        def __init__(self):
            def _int(name: str, default: int) -> int:
                try:
                    return int(os.getenv(name, str(default)))
                except (TypeError, ValueError):
                    return default

            def _float(name: str, default: float) -> float:
                try:
                    return float(os.getenv(name, str(default)))
                except (TypeError, ValueError):
                    return default

            def _bool(name: str, default: bool) -> bool:
                v = os.getenv(name)
                if v is None:
                    return default
                return v.strip().lower() in ("1", "true", "yes", "on")

            def _list(name: str, default: list) -> list:
                raw = os.getenv(name)
                if raw:
                    return [x.strip() for x in raw.split(",") if x.strip()]
                return default

            self.retry_max_attempts = _int("RETRY_MAX_ATTEMPTS", 3)
            self.retry_base_delay = _float("RETRY_BASE_DELAY", 1.0)
            self.retry_max_delay = _float("RETRY_MAX_DELAY", 15.0)
            self.retry_http_codes = _list("RETRY_HTTP_CODES", [429, 500, 502, 503, 504])
            self.bge_batch_size = _int("BGE_BATCH_SIZE", 5)
            self.context_max_chars = _int("CONTEXT_MAX_CHARS", 12000)
            self.context_docs_ratio = _float("CONTEXT_DOCS_RATIO", 0.7)
            self.context_history_ratio = _float("CONTEXT_HISTORY_RATIO", 0.3)
            self.redis_url = os.getenv("REDIS_URL") or None
            self.task_data_retention_seconds = _int("TASK_DATA_RETENTION_SECONDS", 3600)
            self.trace_enabled = _bool("TRACE_ENABLED", True)
            self.trace_slow_threshold_ms = _int("TRACE_SLOW_THRESHOLD_MS", 500)
            self.item_confirm_high_score = _float("ITEM_CONFIRM_HIGH_SCORE", 0.85)
            self.item_confirm_low_score = _float("ITEM_CONFIRM_LOW_SCORE", 0.6)
            self.milvus_min_cosine_score = _float("MILVUS_MIN_COSINE_SCORE", 0.75)
            self.minio_public_prefix = os.getenv("MINIO_PUBLIC_PREFIX", "/upload-images")

    _USE_PYDANTIC_SETTINGS = False


@lru_cache(maxsize=1)
def get_settings() -> _Settings:
    """获取全局唯一配置实例（懒加载 + 缓存）。"""
    return _Settings()


def settings_reload() -> None:
    """清空配置缓存（供测试/热更新使用）。"""
    get_settings.cache_clear()


settings = get_settings()
