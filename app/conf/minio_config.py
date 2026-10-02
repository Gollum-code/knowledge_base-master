# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（确保os.getenv能获取到MinIO相关配置）
load_dotenv()


# 定义MinIO对象存储服务配置（与LLMConfig风格一致，字段对应.env配置项）
@dataclass
class MinIOConfig:
    endpoint: str    # MinIO服务地址（含http/https和端口）
    access_key: str  # MinIO访问密钥（对应MINIO_ACCESS_KEY）
    secret_key: str  # MinIO秘钥（对应MINIO_SECRET_KEY）
    bucket_name: str # MinIO默认存储桶名（知识库文件专用）
    minio_img_dir: str #Minio存储图片的文件夹
    minio_secure: bool # 是否使用ssl加密 http 还是 https


# 环境变量布尔解析：兼容 1/0/true/false/yes/no（不区分大小写）
def env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

# 实例化MinIO配置对象，自动从.env读取配置并绑定
minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=env_bool("MINIO_SECURE", False)
)