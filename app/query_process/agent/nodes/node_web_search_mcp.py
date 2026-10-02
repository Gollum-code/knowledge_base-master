import sys
import json
import asyncio
import threading
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerStreamableHttp
from app.core.logger import logger


def run_mcp_call(query: str):
    """
    安全地运行异步 MCP 调用：
    - 若当前线程没有运行中的事件循环（如线程池/后台线程），直接 asyncio.run
    - 若当前线程已有运行中的事件循环（如在事件循环线程中被同步调用），
      则另起一个线程执行，避免 "asyncio.run() cannot be called from a running event loop" 错误
    """
    try:
        asyncio.get_running_loop()
        # 已有运行中的事件循环 -> 新开线程执行
        box = {}

        def _runner():
            try:
                box["result"] = asyncio.run(mcp_call(query))
            except Exception as e:  # pragma: no cover
                logger.error(f"[MCP] 子线程执行 mcp_call 异常: {e}", exc_info=True)
                box["error"] = e

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in box:
            raise box["error"]
        return box.get("result")
    except RuntimeError:
        # 没有运行中的事件循环 -> 直接执行
        return asyncio.run(mcp_call(query))


def node_web_search_mcp(state):
    """
    LangGraph同步节点函数：处理MCP搜索逻辑，作为整个搜索流程的入口。

    该节点会调用 mcp_call 异步函数获取搜索结果，并将其解析为结构化数据存储到 state 中。

    :param state: LangGraph的全局状态对象，包含 session_id, rewritten_query 等信息
    :return: 字典，包含结构化的搜索结果 web_search_docs，供后续节点使用
    """
    logger.info("---node_web_search_mcp 开始处理---")

    # 1. 标记任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 2. 获取查询词
    query = state.get("rewritten_query", "")
    if not query:
        # 尝试回退到原始查询
        query = state.get("original_query", "")

    docs = []

    # 3. 执行搜索
    if query:
        try:
            # 同步-异步桥接：通过run_mcp_call安全执行异步的mcp_call函数
            logger.info(f"启动异步 MCP 调用，Query: {query}")

            # ======================================================================
            # MCP 返回结果格式解析说明
            # ----------------------------------------------------------------------
            # result 是一个 CallToolResult 对象 (定义在 agents.mcp.types 中)
            # result.content 是一个 TextContent 对象的列表，通常只有一项
            # result.content[0].text 是一个 JSON 字符串，包含实际的搜索结果
            #
            # 示例数据结构：
            # result.content[0].text = """
            # {
            #   "pages": [
            #     {
            #       "title": "HAK 180 烫金机使用手册",
            #       "url": "http://example.com/manual",
            #       "snippet": "在出厂默认状态下，若想设置局部转印..."
            #     },
            #     ...
            #   ]
            # }
            # """
            # ======================================================================
            result = run_mcp_call(query)

            # 4. 解析结果
            if result and not result.isError and result.content:
                # 解析MCP原始结果：提取文本内容并转为JSON对象
                # result.content 通常是一个列表，第一项包含文本结果
                raw_text = result.content[0].text
                try:
                    data = json.loads(raw_text)
                    pages = data.get("pages") or []

                    logger.info(f"MCP 返回原始页面数量: {len(pages)}")

                    # 遍历结果，统一封装为结构化格式
                    for item in pages:
                        snippet = (item.get("snippet") or "").strip()
                        url = (item.get("url") or "").strip()
                        title = (item.get("title") or "").strip()

                        # 过滤无核心摘要的结果
                        if not snippet:
                            continue

                        docs.append({"title": title, "url": url, "snippet": snippet})

                except json.JSONDecodeError:
                    logger.error(f"MCP 返回结果解析 JSON 失败: {raw_text[:100]}...")
            else:
                if result and result.isError:
                    logger.error(f"MCP 返回错误: {result}")
                else:
                    logger.warning("MCP 返回结果为空或无效")

            logger.info(f"结构化搜索结果数量: {len(docs)}")

        except Exception as e:
            logger.error(f"MCP 搜索节点执行异常: {e}", exc_info=True)
    else:
        logger.warning("查询词为空，跳过 MCP 搜索")

    # 5. 标记任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info("---node_web_search_mcp 处理结束---")

    # 若有有效搜索结果，返回结果供后续节点使用；无则返回空字典
    if docs:
        return {"web_search_docs": docs}
    return {}



async def mcp_call(query):
    """
    异步调用百炼MCP搜索服务的核心函数。

    该函数负责初始化MCP客户端，建立SSE连接，调用远程工具，并返回原始结果。

    :param query: 搜索查询词（通常是经过改写后的精准Query）
    :return: MCP返回的原始结果对象 (包含 content, isError 等字段)
    """

    api_key = (mcp_config.api_key or "").strip()
    if not api_key:
        raise ValueError("MCP API Key 未配置，请在 .env 中设置 OPENAI_API_KEY（百炼通用 sk-xxx Key）")
    if not mcp_config.mcp_base_url:
        raise ValueError("MCP 服务地址未配置，请在 .env 中设置 MCP_DASHSCOPE_BASE_URL")

    # 百炼 MCP 要求 Authorization: Bearer <sk-xxx>，避免重复拼接 Bearer
    auth_token = api_key.removeprefix("Bearer ").strip()
    headers = {"Authorization": f"Bearer {auth_token}"}

    # ==================================================================================
    # 初始化百炼 MCP Streamable HTTP 客户端
    # ----------------------------------------------------------------------------------
    # 百炼 WebSearch 已升级为 Streamable HTTP 协议，端点形如:
    #   https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp
    # 旧版 SSE 端点 (.../sse) 已废弃，使用 MCPServerSse 会导致连接失败。
    # ==================================================================================
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": headers,
            "timeout": 300,
            "sse_read_timeout": 300,
        },
    )

    try:
        logger.info(f"[MCP] 正在连接百炼 WebSearch 服务: {mcp_config.mcp_base_url}")
        # 建立与MCP服务的SSE连接（异步方法，需await）
        await search_mcp.connect()

        # 打印所有工具名，核对一下
        tools = await search_mcp.list_tools()
        logger.info(f"[MCP] 发现可用工具: {tools}")

        logger.info(f"[MCP] 连接成功，正在调用工具 'bailian_web_search' 查询: {query}")
        # 调用百炼MCP的搜索工具（核心步骤）
        # tool_name: "bailian_web_search" 是百炼官方定义的工具名称
        # arguments: 工具所需的参数，这里需要 "query" (查询词) 和 "count" (返回数量)
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5}
        )
        logger.info("[MCP] 工具调用完成，已获取返回结果")
        return result

    except Exception as e:
        logger.error(f"[MCP] 调用过程中发生异常: {e}", exc_info=True)
        return None

    finally:
        # 无论调用成功/失败，最终都关闭MCP连接（释放资源，异步方法）
        await search_mcp.cleanup()


if __name__ == '__main__':
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("=" * 50)

    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")