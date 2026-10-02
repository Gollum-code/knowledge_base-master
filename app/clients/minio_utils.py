# 导入Python内置模块
import json
# 导入MinIO官方Python SDK核心类
from minio import Minio
# 项目内部配置与日志
from app.conf.minio_config import minio_config
from app.conf.settings import settings
from app.core.logger import logger

# 全局MinIO客户端对象，首次调用 get_minio_client() 时懒加载
minio_client = None
_init_attempted = False


def _init_minio_client():
    """
    懒加载 MinIO 客户端：避免在模块 import 阶段连接远程服务，
    防止 MinIO 不可用或时钟偏差导致整个 import_service 无法启动。
    """
    global minio_client, _init_attempted
    if _init_attempted:
        return minio_client

    _init_attempted = True

    if not minio_config.endpoint:
        logger.warning("MinIO 未配置 MINIO_ENDPOINT，跳过客户端初始化")
        return None

    try:
        client = Minio(
            endpoint=minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.minio_secure,
        )
        bucket_name = minio_config.bucket_name

        if not client.bucket_exists(bucket_name):
            logger.info(f"MinIO存储桶[{bucket_name}]不存在，开始创建")
            client.make_bucket(bucket_name)
            logger.info(f"MinIO存储桶[{bucket_name}]创建成功")
        else:
            logger.info(f"MinIO存储桶[{bucket_name}]已存在，无需重复创建")

        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                # 仅公开图片目录（MINIO_IMG_DIR / MINIO_PUBLIC_PREFIX 配置的前缀），
                # 其余对象（PDF 等）不再匿名可读，需通过签名 URL 访问
                "Resource": [f"arn:aws:s3:::{bucket_name}{settings.minio_public_prefix}/*"],
            }],
        }
        client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))
        logger.info(
            f"MinIO存储桶[{bucket_name}]已配置公网只读策略（仅前缀 {settings.minio_public_prefix}），支持匿名URL访问")

        minio_client = client
        return minio_client

    except Exception as e:
        logger.error(
            f"MinIO客户端初始化失败，服务将继续运行但 MinIO 功能不可用。"
            f"错误信息：{e}",
            exc_info=True,
        )
        if "RequestTimeTooSkewed" in str(e):
            logger.error(
                "检测到 RequestTimeTooSkewed：本机系统时间与 MinIO 服务器偏差过大，"
                "请同步 Windows 系统时钟后重试（设置 → 时间和语言 → 立即同步）。"
            )
        minio_client = None
        return None


def get_minio_client():
    """
    获取 MinIO 客户端实例（懒加载，初始化失败时返回 None）
    """
    if minio_client is not None:
        return minio_client
    return _init_minio_client()


if __name__ == "__main__":
    get_minio_client()
