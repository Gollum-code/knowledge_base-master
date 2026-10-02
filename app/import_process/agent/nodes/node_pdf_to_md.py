import shutil
import time
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.core.trace import timing
from app.utils.retry_utils import with_retry


class NodePdfToMd(NodeBase):
    """
    节点: PDF转Markdown (node_pdf_to_md)
    核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_pdf_to_md"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph工作流节点：PDF转MD核心处理节点
        核心职责：路径校验 → MinerU上传解析 → 结果下载解压 → 读取MD内容并更新工作流状态
        :param state: 必须包含 task_id(任务ID)、pdf_path(pdf文件路径)、local_dir(输出文件路径)
        :return:  md_path(pdf转成md后存储的路径)、md_content(md文件的内容)
        """

        with timing("node_pdf_to_md(内含MinerU上传+轮询+下载)"):
            # 1：校验PDF路径和输出目录
            pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

            # 步骤2：上传PDF至MinerU并轮询解析结果
            zip_url = self._step_2_upload_and_poll(pdf_path_obj, output_dir_obj)

            # 步骤3：下载ZIP包并解压提取MD文件
            md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

            # 步骤 4：读取内容 (如果文件不存在或编码错误，直接抛出异常，由外层捕获)
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            # 更新状态
            state["md_path"] = str(md_path)
            state["md_content"] = md_content

        return state


    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
        """

        pdf_path = state.get("pdf_path", "").strip()
        local_dir = state.get("local_dir", "").strip()

        # 参数非空校验
        if not pdf_path:
            raise ValueError(f"缺失参数：pdf_path")
        if not local_dir:
            raise ValueError(f"缺失参数：local_dir")

        # 转换为Path对象统一处理路径
        pdf_path_obj = Path(pdf_path)
        output_dir_obj = Path(local_dir)

        # PDF文件有效性校验（存在且为文件，非目录）
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF文件不存在，绝对路径：{pdf_path_obj.absolute()}")
        if not pdf_path_obj.is_file():
            raise FileNotFoundError(f"指定路径非文件（是目录），绝对路径：{pdf_path_obj.absolute()}")

        # 确保输出目录存在，不存在则递归创建
        if not output_dir_obj.exists():
            logger.info(f"输出目录不存在，自动创建：{output_dir_obj.absolute()}")
            output_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, output_dir_obj




    def _step_2_upload_and_poll(self, pdf_path_obj: Path, output_dir_obj: Path):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传（含重试） → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象；output_dir_obj-输出目录Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
        """

        # 前置配置校验，拦截无效配置
        if not mineru_config.base_url or not mineru_config.api_token:
            raise ValueError("MinerU配置缺失：请在.env中正确配置MINERU_BASE_URL和MINERU_API_TOKEN")
        logger.info(f"[配置校验] MinerU基础配置加载成功，开始处理文件：{pdf_path_obj.name}")

        # 构造请求头（符合HTTP规范，Bearer鉴权）
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mineru_config.api_token}"
        }

        # 1. 调用批量接口，获取上传Signed URL和任务batch_id（瞬时失败自动重试）
        url_get_upload = f"{mineru_config.base_url}/file-urls/batch"
        req_data = {
            "files": [{"name": pdf_path_obj.name}],
            "model_version": "vlm"  # 官方推荐解析模型
        }
        logger.info(f"[获取上传链接] 调用接口：{url_get_upload}，请求参数：{req_data}")

        def _request_upload_url():
            resp = requests.post(url=url_get_upload, headers=request_headers, json=req_data, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"[获取上传链接] 网络请求失败，状态码：{resp.status_code}，响应内容：{resp.text}")
            return resp

        resp = with_retry(_request_upload_url, max_attempts=3)

        resp_data = resp.json()
        if resp_data["code"] != 0:
            raise RuntimeError(f"[获取上传链接] API业务错误，返回数据：{resp_data}")

        # 提取核心数据：上传链接和任务唯一标识
        signed_url = resp_data["data"]["file_urls"][0]
        batch_id = resp_data["data"]["batch_id"]
        logger.info(f"[获取上传链接] 成功，batch_id：{batch_id}，上传链接已生成")

        # 2. 读取PDF二进制数据，准备上传
        logger.info(f"[文件上传] 开始读取PDF文件：{pdf_path_obj.name}")
        with open(pdf_path_obj, "rb") as f:
            file_data = f.read()

        # 创建Session（复用TCP连接，禁用代理避免签名验证失败）
        upload_session = requests.Session()
        upload_session.trust_env = False            # 防止代理失效

        try:
            # 首次上传：自动识别文件类型
            put_resp = upload_session.put(url=signed_url, data=file_data, timeout=60)
            # 重试逻辑：首次失败则强制指定PDF的Content-Type
            if put_resp.status_code != 200:
                logger.warning(f"[文件上传] 首次上传失败（状态码：{put_resp.status_code}），强制指定PDF类型重试")
                pdf_headers = {"Content-Type": "application/pdf"}
                put_resp = upload_session.put(url=signed_url, data=file_data, headers=pdf_headers, timeout=60)
                # 重试仍失败则抛出异常
                if put_resp.status_code != 200:
                    raise RuntimeError(f"[文件上传] 重试后仍失败，状态码：{put_resp.status_code}，响应内容：{put_resp.text}")
            logger.info(f"[文件上传] 成功，文件{pdf_path_obj.name}已存入云存储")
        except Exception as e:
            raise RuntimeError(f"[文件上传] 网络异常导致上传失败，错误信息：{str(e)}")
        finally:
            # 无论成败，关闭Session释放网络连接，避免资源泄漏
            upload_session.close()

        # 3. 根据batch_id轮询任务状态，直至完成/失败/超时
        poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()
        timeout_seconds = 600  # 最大超时时间10分钟（适配600页内PDF）
        poll_interval = 3      # 轮询间隔3秒（平衡查询频率和服务端压力）
        logger.info(f"[任务轮询] 开始监控任务状态，batch_id：{batch_id}，最大超时：{timeout_seconds}s")

        while True:
            # 超时检查：超过最大时间直接终止轮询
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"[任务轮询] 超时！任务处理超{int(timeout_seconds)}秒，batch_id：{batch_id}")

            # 发起轮询请求，短超时10秒，异常则重试
            try:
                poll_resp = requests.get(url=poll_url, headers=request_headers, timeout=10)
            except Exception as e:
                logger.warning(f"[任务轮询] 网络请求异常，{poll_interval}秒后重试：{str(e)}")
                time.sleep(poll_interval)
                continue

            # 处理HTTP响应错误：5xx服务端繁忙则重试，其他错误直接抛出
            if poll_resp.status_code != 200:
                if 500 <= poll_resp.status_code < 600:
                    logger.warning(f"[任务轮询] 服务端繁忙（状态码：{poll_resp.status_code}），{poll_interval}秒后重试")
                    time.sleep(poll_interval)
                    continue
                else:
                    raise RuntimeError(f"[任务轮询] HTTP请求失败，状态码：{poll_resp.status_code}，响应内容：{poll_resp.text}")

            # 解析轮询结果，校验业务状态
            poll_data = poll_resp.json()
            if poll_data["code"] != 0:
                raise RuntimeError(f"[任务轮询] API业务错误，返回数据：{poll_data}")

            extract_results = poll_data["data"]["extract_result"]
            # 结果暂空，继续轮询
            if not extract_results:
                logger.info(f"[任务轮询] 结果暂为空，已耗时{int(elapsed_time)}s，继续等待")
                time.sleep(poll_interval)
                continue

            # 解析任务状态，分支处理
            result_item = extract_results[0]
            state_status = result_item["state"]
            # 状态1：任务完成，提取ZIP下载链接
            if state_status == "done":
                logger.info(f"[任务轮询] 解析任务完成！总耗时：{int(elapsed_time)}s，batch_id：{batch_id}")
                full_zip_url = result_item.get("full_zip_url")
                if not full_zip_url:
                    raise RuntimeError("[任务轮询] 任务完成但未返回ZIP包下载链接，batch_id：{batch_id}")
                logger.info(f"[任务轮询] 结果ZIP包下载链接：{full_zip_url}...")
                return full_zip_url
            # 状态2：任务失败，提取错误信息抛出
            elif state_status == "failed":
                err_msg = result_item.get("err_msg", "未知错误，无具体信息")
                raise RuntimeError(f"[任务轮询] 解析任务失败，batch_id：{batch_id}，错误信息：{err_msg}")
            # 状态3：处理中，实时打印进度（覆盖当前行）
            else:
                logger.debug(
                    f"[任务轮询] 处理中（已耗时{int(elapsed_time)}s），状态：{state_status} | 刷新间隔{poll_interval}s",
                    end="\r"        # 覆盖当前行
                )
                time.sleep(poll_interval)



    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        """
        步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件（重命名统一规范）
        核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件（按优先级） → 重命名统一为PDF同名
        参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
        """

        # 1. 下载解析结果ZIP包，120秒超时适配大文件
        logger.info(f"[步骤1/4] 开始下载ZIP包，链接：{zip_url}...")
        resp = requests.get(zip_url, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"[步骤1/4] ZIP包下载失败，HTTP状态码：{resp.status_code}")

        # 拼接ZIP包保存路径，按PDF名称唯一命名
        zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"[步骤1/4] ZIP包下载成功，保存路径：{zip_save_path}")

        # 2. 清理旧解压目录并解压ZIP包（避免旧文件干扰，为每个PDF创建专属目录）
        logger.info(f"[步骤2/4] 开始解压ZIP包...")
        extract_target_dir = output_dir_obj / pdf_stem

        # 清理旧目录，异常则警告不终止
        if extract_target_dir.exists():
            try:
                # 递归删除整个目录树，包括目录本身及其所有子目录和文件。
                shutil.rmtree(extract_target_dir)
                logger.info(f"[步骤2/4] 已清理旧的解压目录：{extract_target_dir}")
            except Exception as e:
                logger.warning(f"[步骤2/4] 清理旧目录失败，可能影响新文件解压：{str(e)}")

        # 重新创建解压目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 核心解压操作，保留原目录结构
        with zipfile.ZipFile(zip_save_path, 'r') as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        logger.info(f"[步骤2/4] ZIP包解压完成，解压目录：{extract_target_dir}")

        # 3. 递归查找解压目录下所有MD文件（适配子目录结构）
        logger.info(f"[步骤3/4] 开始查找解压目录中的MD文件...")
        md_file_list = list(extract_target_dir.rglob("*.md"))
        if not md_file_list:
            raise FileNotFoundError(f"[步骤3/4] 解压目录中未找到任何.md格式文件：{extract_target_dir}")
        logger.info(f"[步骤3/4] 共找到{len(md_file_list)}个MD文件，按优先级匹配目标文件")

        # 4. 按优先级匹配目标MD文件（同名→full.md→第一个，兜底避免流程中断）
        target_md_file = None
        # 优先级1：与PDF纯名称完全同名的MD文件
        for md_file in md_file_list:
            if md_file.stem == pdf_stem:
                target_md_file = md_file
                logger.info(f"[步骤4/4] 匹配到优先级1目标：与PDF同名的MD文件 {target_md_file.name}")
                break
        # 优先级2：MinerU默认生成的full.md（不区分大小写）
        if not target_md_file:
            for md_file in md_file_list:
                if md_file.name.lower() == "full.md":
                    target_md_file = md_file
                    logger.info(f"[步骤4/4] 匹配到优先级2目标：MinerU默认文件 {target_md_file.name}")
                    break
        # 优先级3：兜底取第一个MD文件
        if not target_md_file:
            target_md_file = md_file_list[0]
            logger.info(f"[步骤4/4] 未匹配到前两级目标，兜底取第一个MD文件 {target_md_file.name}")

        # 重命名MD文件：统一为PDF纯名称，便于后续流程处理（仅不同名时执行）
        if target_md_file.stem != pdf_stem:
            logger.info(f"[步骤4/4] 开始重命名MD文件，统一为PDF同名：{pdf_stem}.md")
            new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
            try:
                # 将磁盘上的文件进行重命名
                target_md_file.rename(new_md_path)
                # 更新变量引用
                target_md_file = new_md_path
                logger.info(f"[步骤4/4] MD文件重命名成功：{pdf_stem}.md")
            except OSError as e:
                logger.warning(f"[步骤4/4] MD文件重命名失败，将使用原文件名继续流程：{str(e)}")

        # 转换为字符串绝对路径返回，适配后续仅支持字符串路径的函数
        final_md_path = str(target_md_file.absolute())

        return final_md_path






def test_step2():

    test_pdf_name = os.path.join("doc", "H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态，模拟流程入参
    test_state = create_default_state(
        task_id="task_demo",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    # 执行核心处理流程
    node_pdf_to_md = NodePdfToMd()
    node_pdf_to_md(test_state)

def test_step1to2():

    local_file = os.path.join("doc", "H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf")
    local_file_path = os.path.join(PROJECT_ROOT, local_file)
    local_dir = os.path.join(PROJECT_ROOT, "output")
    initial_state = create_default_state(
        task_id="task_demo",
        local_file_path=local_file_path,
        local_dir=local_dir
    )

    # step1
    from app.import_process.agent.nodes.node_entry import NodeEntry
    node_entry = NodeEntry()
    state1 = node_entry(initial_state)
    # step2
    node_pdf_to_md = NodePdfToMd()
    node_pdf_to_md(state1)

if __name__ == "__main__":

    import os
    from app.import_process.agent.state import create_default_state
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"获取根地址：{PROJECT_ROOT}")

    # test_step2()
    test_step1to2()