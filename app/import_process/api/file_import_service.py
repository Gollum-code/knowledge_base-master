import os
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.conf.minio_config import minio_config
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
)
from app.utils.sse_utils import create_sse_queue, get_sse_queue, sse_generator, SSEEvent, push_to_session
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具
from app.core.trace import set_trace_id, TraceContext


# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus/KG入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
# 允许来源从环境变量读取，未配置时默认全放行（内网演示）
# 生产环境建议配置为具体域名列表，如：CORS_ALLOWED_ORIGINS=https://kb.example.com
_allowed_origins = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=_allowed_origins != ["*"],  # 仅具体域名时才允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)

# --------------------------
# 前端静态资源（Vue 构建产物，与 query 模块结构一致）
# 访问地址：http://localhost:8000/import.html
# --------------------------
_PAGE_DIR = Path(__file__).absolute().parent.parent / "page" / "frontend" / "dist"
_INDEX_HTML = _PAGE_DIR / "index.html"
_ASSETS_DIR = _PAGE_DIR / "assets"

if _ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="import_assets")


# --------------------------
# 可选鉴权 & 上传安全配置
# --------------------------
# 配置了 API_AUTH_TOKEN 后，/upload 与 /status 需要携带 Bearer Token 才能访问
_api_auth_token = os.getenv("API_AUTH_TOKEN", "").strip()
# 允许上传的文件扩展名白名单
_ALLOWED_UPLOAD_EXTS = {".pdf", ".md", ".txt", ".doc", ".docx"}
# 单文件大小上限（默认 200MB）
_MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_BYTES", 200 * 1024 * 1024))


@app.middleware("http")
async def _upload_auth_middleware(request: Request, call_next):
    if _api_auth_token:
        path = request.url.path
        protected = path == "/upload" or path.startswith("/status/")
        if protected and request.headers.get("Authorization", "") != f"Bearer {_api_auth_token}":
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


def _serve_import_page():
    if not _INDEX_HTML.exists():
        raise HTTPException(
            status_code=404,
            detail=f"前端页面未构建，请先执行: cd app/import_process/page/frontend && npm run build。路径：{_INDEX_HTML}",
        )
    return FileResponse(_INDEX_HTML, media_type="text/html")


@app.get("/")
async def import_index():
    return _serve_import_page()


@app.get("/import.html", response_class=FileResponse)
async def get_import_page():
    logger.info(f"前端页面访问，文件绝对路径：{_INDEX_HTML}")
    return _serve_import_page()



# --------------------------
# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
# --------------------------
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
    LangGraph全流程执行后台任务
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看

    :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
    :param local_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param local_file_path: 上传文件的本地绝对路径
    """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, TASK_STATUS_PROCESSING, push_queue=True)
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")

        # 2. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path
        init_state["is_stream"] = True

        # 3. 流式执行 LangGraph 全流程（节点内通过 SSE 推送进度）
        #    后台线程不继承 contextvars，需显式恢复 trace_id
        with TraceContext(task_id):
            for event in kb_import_app.stream(init_state):
                for node_name in event:
                    logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")

        # 4. 全流程执行完成
        update_task_status(task_id, TASK_STATUS_COMPLETED, push_queue=True)
        push_to_session(task_id, SSEEvent.FINAL, {
            "status": TASK_STATUS_COMPLETED,
            "done_list": get_done_task_list(task_id),
            "running_list": [],
        })
        push_to_session(task_id, SSEEvent.CLOSE, {})
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")

    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED, push_queue=True)
        push_to_session(task_id, SSEEvent.ERROR, {"error": str(e)})
        push_to_session(task_id, SSEEvent.CLOSE, {})
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)



# --------------------------
# 核心接口：文件上传接口
# 支持多文件上传，核心流程：接收文件 → 本地保存 → MinIO上传 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
# --------------------------
@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    文件上传核心接口
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 将文件上传至MinIO对象存储，做持久化保存
    4. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    5. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    for file in files:
        # 生成全局唯一TaskID（UUID4），作为单个文件的全流程标识
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)

        # 文件名安全清洗：仅保留 basename，防止路径穿越（../ 或绝对路径）
        raw_filename = (file.filename or "").replace("\\", "/").split("/")[-1].strip()
        if not raw_filename or raw_filename in (".", ".."):
            raise HTTPException(status_code=400, detail="非法文件名")
        # 扩展名白名单校验
        ext = os.path.splitext(raw_filename)[1].lower()
        if ext not in _ALLOWED_UPLOAD_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {ext or '无扩展名'}，仅支持 {'/'.join(sorted(_ALLOWED_UPLOAD_EXTS))}",
            )
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{raw_filename}，文件类型：{file.content_type}")

        # 3. 创建 SSE 队列并标记「文件上传」阶段为「运行中」
        create_sse_queue(task_id)
        update_task_status(task_id, TASK_STATUS_PROCESSING, push_queue=True)
        add_running_task(task_id, "upload_file", is_stream=True)

        # 4. 构建该任务的本地独立目录：output/YYYYMMDD/TaskID，避免多文件重名冲突
        task_local_dir = os.path.join(date_based_root_dir, task_id)
        os.makedirs(task_local_dir, exist_ok=True)  # 目录不存在则创建，存在则不做处理
        # 构建上传文件的本地保存绝对路径
        local_file_abs_path = os.path.join(task_local_dir, raw_filename)

        # 5. 将上传的文件保存到本地临时目录（流式写入并校验大小上限）
        # 后续MinIO上传/文件解析均基于此文件
        written = 0
        with open(local_file_abs_path, "wb") as file_buffer:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_FILE_SIZE:
                    file_buffer.close()
                    os.remove(local_file_abs_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过大小上限（{_MAX_FILE_SIZE // (1024 * 1024)}MB）：{raw_filename}",
                    )
                file_buffer.write(chunk)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}，大小：{written} 字节")

        # 6. 将本地文件上传至MinIO对象存储，做持久化保存
        # 从环境变量获取MinIO的PDF存储目录配置
        minio_pdf_base_dir = os.getenv("MINIO_PDF_DIR", "pdf_files")  # 缺省值：pdf_files
        # 构建MinIO中的文件对象名：配置目录/YYYYMMDD/文件名（按日期分层，和本地一致）
        minio_object_name = f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{raw_filename}"
        try:
            # 获取MinIO客户端实例
            minio_client = get_minio_client()
            if minio_client is None:
                # MinIO客户端获取失败，抛出500服务异常
                raise HTTPException(status_code=500,
                                    detail="MinIO service connection failed, please check MinIO config")
            # 从环境变量获取MinIO的桶名配置
            minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket")  # 缺省值：kb-import-bucket

            # 本地文件上传至MinIO（同名文件会自动覆盖，保证文件最新）
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=local_file_abs_path,
                content_type=file.content_type  # 传递文件原始MIME类型
            )
            logger.info(f"[{task_id}] 文件已成功上传至MinIO，桶名：{minio_bucket_name}，对象名：{minio_object_name}")
        except Exception as e:
            # MinIO上传失败，记录警告日志（不中断后续流程，本地文件仍可继续处理）
            logger.warning(f"[{task_id}] 文件上传MinIO失败，将继续执行本地处理流程，异常信息：{str(e)}", exc_info=True)

        # 7. 标记「文件上传」阶段为「已完成」
        add_done_task(task_id, "upload_file", is_stream=True)

        # 8. 将LangGraph全流程处理加入FastAPI后台任务（异步执行，不阻塞当前接口响应）
        background_tasks.add_task(run_graph_task, task_id, task_local_dir, local_file_abs_path)
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    # 9. 所有文件处理完毕，返回上传成功信息和所有TaskID（前端基于TaskID轮询进度）
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }





# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info


@app.get("/stream/{task_id}", summary="任务进度 SSE 流")
async def stream_task_progress(task_id: str, request: Request):
    """SSE 实时推送导入任务进度，前端通过 EventSource 订阅。"""
    if get_sse_queue(task_id) is None:
        create_sse_queue(task_id)
    return StreamingResponse(
        sse_generator(task_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/presign-upload", summary="获取文件直传预签名URL", description="返回 MinIO 预签名 PUT URL，前端可直接上传文件到对象存储，减少服务端中转带宽")
async def presign_upload(filename: str = Query(...)):
    """
    图片/文件直传预签名接口：
    前端携带文件名调用本接口，得到一次性 PUT URL 与最终对象名，
    随后浏览器直接 PUT 到 MinIO，避免经服务端转发占用带宽。
    默认仅允许图片目录（与 MinIO 只读策略一致），生产可扩展为任意前缀。
    """
    raw = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if not raw or raw in (".", ".."):
        raise HTTPException(status_code=400, detail="非法文件名")
    client = get_minio_client()
    if client is None:
        raise HTTPException(status_code=500, detail="MinIO service connection failed, please check MinIO config")
    bucket = minio_config.bucket_name or os.getenv("MINIO_BUCKET_NAME", "knowledge-base-files")
    # 图片直传目录：配置前缀 + 当前日期，与 node_md_img 存储结构保持一致
    img_dir = (minio_config.minio_img_dir or os.getenv("MINIO_IMG_DIR", "/upload-images")).lstrip("/")
    object_name = f"{img_dir}/{datetime.now().strftime('%Y%m%d')}/{raw}"
    try:
        url = client.presigned_put_object(bucket_name=bucket, object_name=object_name, expires=3600)
    except Exception as e:
        logger.error(f"生成预签名URL失败：{e}")
        raise HTTPException(status_code=500, detail=f"presign error: {e}")
    return {"presigned_url": url, "object_name": object_name, "bucket": bucket}


# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8000  # 服务端口
    )