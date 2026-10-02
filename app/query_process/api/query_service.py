import os
from pathlib import Path
import json
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, remove_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app
from app.core.logger import logger
from app.core.trace import set_trace_id, current_trace_id, TraceContext

# 后续导入启动图对象
#from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service",description="掌柜智库查询服务！")
# 跨域问题解决：允许来源从环境变量读取，未配置时默认全放行（内网演示）
# 生产环境建议配置为具体域名列表，如：CORS_ALLOWED_ORIGINS=https://kb.example.com,https://admin.example.com
_allowed_origins = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    # 允许来源为具体域名时才允许携带 Cookie 等认证信息（通配符 + credentials 是非法组合）
    allow_credentials=_allowed_origins != ["*"],
)

# 前端静态资源目录（Vue 构建产物）
_PAGE_DIR = Path(__file__).absolute().parent.parent / "page" / "frontend" / "dist"
_INDEX_HTML = _PAGE_DIR / "index.html"
_ASSETS_DIR = _PAGE_DIR / "assets"

if _ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="query_assets")


def _serve_chat_page():
    if not _INDEX_HTML.exists():
        raise HTTPException(
            status_code=404,
            detail=f"前端页面未构建，请先执行: cd app/query_process/page/frontend && npm run build。路径：{_INDEX_HTML}",
        )
    return FileResponse(_INDEX_HTML, media_type="text/html")


@app.get("/")
async def index():
    return _serve_chat_page()


@app.get("/chat.html")
async def chat():
    return _serve_chat_page()


# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., min_length=1, max_length=10000, description="查询内容")  # 必须填写，且非空
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


# 可选鉴权：配置了 API_AUTH_TOKEN 后，/query 与 /history 需要携带 Bearer Token 才能访问
_api_auth_token = os.getenv("API_AUTH_TOKEN", "").strip()


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    if _api_auth_token:
        path = request.url.path
        protected = (
            path == "/query"
            or path.startswith("/history/")
            or (path.startswith("/history") and request.method == "DELETE")
        )
        if protected and request.headers.get("Authorization", "") != f"Bearer {_api_auth_token}":
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    :param background_tasks:
    :param request:
    :return:
    """
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=422, detail="查询内容不能为空")
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # 生成请求级 trace_id（贯穿整个查询链路，便于日志追踪）
    set_trace_id(session_id)

    # 处理是不是流式返回结果
    is_stream = request.is_stream
    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        create_sse_queue(session_id)
    # 更新任务状态
    # 当前会话id作为key! 整体装填处于运行中！
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    logger.info(f"开始处理流程... 是否流式: {is_stream}, 其他参数: {user_query}, session_id: {session_id}")

    if is_stream:
        # 如果是流式，则返回一个流式响应，过程不断地推送
        # 运行执行图对象方法
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        # 返回结果
        logger.info(f"开始处理结果.... session_id: {session_id}")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        # 同步运行：放入线程池执行，避免阻塞事件循环（期间包含 LLM/MCP/模型推理等耗时调用）
        await run_in_threadpool(run_query_graph, session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        sources_raw = get_task_result(session_id, "sources", "[]")
        try:
            sources = json.loads(sources_raw) if sources_raw else []
        except json.JSONDecodeError:
            sources = []
        done_list = get_done_task_list(session_id)
        # 响应读取完成后清理该会话的临时任务数据，避免内存无限增长
        clear_task(session_id)
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
            "done_list": done_list,
        }


# 定义查询接口
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    logger.info(f"开始流程图处理...{session_id} {user_query} {is_stream}")

    default_state = {"original_query": user_query, "session_id": session_id, "is_stream": is_stream}
    try:
        # 后台线程内显式恢复 trace_id（BackgroundTasks 线程不继承 contextvars）
        with TraceContext(session_id):
            # 后期运行
            query_app.invoke(default_state)
        # 整体任务就更新完了！ 接下来就是数据的更新了！
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        logger.error(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
    finally:
        # 流式任务结束后：推送关闭信号并清理 SSE 队列与临时任务数据
        # （若客户端一直未订阅 /stream，队列也会在此释放，避免内存泄漏）
        if is_stream:
            push_to_session(session_id, SSEEvent.CLOSE, {})
            remove_sse_queue(session_id)
            clear_task(session_id)


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    logger.info(f"调用流式/stream... session_id: {session_id}")
    """
    sse 实时返回结果
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    return {"ok": True}


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")

@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count =  clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)



