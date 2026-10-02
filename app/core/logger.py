# app/core/logger.py

"""
项目日志工具类
基于loguru实现，支持.env配置控制台/文件双输出，自动生成logs/app_年月日.log
特性：
1. 配置驱动：通过.env开关输出、修改日志级别
2. 自动路径：文件日志默认输出到 项目根/logs/app_YYYYMMDD.log
3. 自动清理：按配置保留日志，自动删除过期文件
4. 中文友好：utf-8编码，彻底解决中文乱码
5. 异步安全：开启异步入队，支持多线程/异步场景，避免日志错乱
6. 开箱即用：项目所有模块直接导入logger即可使用
7. 位置终极精准：穿透loguru内部+工具类自身，完美显示业务模块实际调用位置
"""
import sys
import inspect
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger


# --- 1. 加载.env配置文件,注入配置项 ---
load_dotenv()

# --- 2. 读取.env配置并转换格式 ---
LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")

# --- 3. 定义日志路径（自动推导项目根路径） ---
# __file__ 是当前文件的路径。resolve()获取绝对路径。
# .parent 向上跳一级。这里跳了三级：logger.py -> core -> app -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

# ---  4. 定义日志格式（彩色、结构化、易读） ---
# <green> 等标签是 loguru 自带的颜色控制
# {time} 时间, {level} 级别, {name} 文件名, {line} 行号, {message} 日志内容
# : <30 表示左对齐并占位30个字符，确保日志排列整齐
# 说明：extra[trace_id] 由下方 fix_log_position patcher 兜底注入（无请求时为 "-"），
# 因此格式中引用 {extra[trace_id]} 是安全的，且在 enqueue=True 下也一致生效。
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <30}</cyan>:<cyan>{line: <4}</cyan> | "
    "<magenta>tid={extra[trace_id]: <12}</magenta> - "
    "<level>{message}</level>"
)

# --- 5. 初始化日志配置（核心方法） ---
def init_logger():
    """
    初始化全局日志配置
    1. 移除loguru默认控制台输出（避免重复打印）
    2. 根据.env配置开启/关闭控制台输出
    3. 根据.env配置开启/关闭文件输出（自动创建logs文件夹）
    4. 配置日志格式、级别、分割、保留策略
    :return: 配置完成的loguru logger实例
    """
    # 1. 移除loguru默认的控制台输出
    logger.remove()

    # 2. 配置控制台输出（若.env开启）
    if LOG_CONSOLE_ENABLE:
        logger.add(
            sink=sys.stdout,            # 输出到标准输出（屏幕）
            level=LOG_CONSOLE_LEVEL,    # 日志级别
            format=LOG_FORMAT,          # 使用上面定义的彩色格式
            colorize=True,              # 开启颜色显示
            enqueue=True                # 开启队列模式，保证多线程安全（不卡主线程）
        )

    # 3. 配置文件输出（若.env开启）
    if LOG_FILE_ENABLE:
        # 确保 logs 目录存在，不存在则递归创建
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=LOG_FILE_PATH,                 # 日志文件路径
            level=LOG_FILE_LEVEL,               # 文件日志级别
            format=LOG_FORMAT,                  # 格式
            rotation="00:00",                   # 分割策略：每天凌晨0点创建一个新文件
            retention=LOG_FILE_RETENTION,       # 保留策略：自动删除旧日志
            encoding="utf-8",                   # 编码：防止中文乱码
            enqueue=True,                       # 异步写入
            backtrace=True,                     # 错误发生时记录详细堆栈
            diagnose=True                       # 诊断模式，显示变量值（辅助排错)
        )

    return logger

# --- 6. 初始化并修正全局logger ---
# 先运行初始化，得到一个基础的 logger 实例
base_logger = init_logger()

def fix_log_position(record):
    """遍历调用栈，跳过loguru内部帧+工具类自身帧，提取业务代码实际调用位置"""
    # 从 contextvars 读取 trace_id 注入日志 extra，实现按请求追踪（永远兜底为 "-"）
    try:
        from app.core.trace import current_trace_id
        tid = current_trace_id() or "-"
    except Exception:
        tid = "-"
    record["extra"]["trace_id"] = tid
    for frame in inspect.stack():
        # 终极过滤：排除loguru内部 + 排除工具类logger.py自身，直接定位业务模块
        if ("_logger.py" in frame.filename or frame.function == "_log") or "logger.py" in frame.filename:
            continue
        # 更新日志字段为业务代码实际位置
        record.update(
            name=frame.filename.split("/")[-1].split("\\")[-1],
            function=frame.function,
            line=frame.lineno
        )
        break

# 应用终极修复，导出全局可用的logger
logger = base_logger.patch(fix_log_position)
