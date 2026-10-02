from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None

def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    # 单例模式：已初始化则直接返回，避免重复加载模型
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef

    # 从环境变量加载配置，无配置则使用默认值
    # 本地有可以使用本地地址！ 没有使用 "BAAI/bge-m3" 会自动下载！ 如果云端部署也可以使用url地址！
    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    # 打印模型初始化配置，便于问题排查
    logger.info(
        "开始初始化BGE-M3模型",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True
        }
    )

    try:
        # 初始化BGE-M3模型，开启原生L2归一化（适配Milvus IP内积检索）
        try:
            _bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16,
                normalize_embeddings=True  # 模型原生对稠密+稀疏向量做L2归一化
            )
        except Exception as e:
            # GPU 初始化失败（如显存不足/驱动异常）：自动降级为 CPU + FP32 重试
            if device.startswith("cuda"):
                logger.warning(
                    f"BGE-M3 在 {device} 上初始化失败（{e}），自动降级为 CPU + FP32 重试")
                _bge_m3_ef = BGEM3EmbeddingFunction(
                    model_name=model_name,
                    device="cpu",
                    use_fp16=False,
                    normalize_embeddings=True
                )
            else:
                raise
        logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
        raise  # 向上抛出异常，由调用方处理


def generate_embeddings(texts):
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
        embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        sparse = embeddings["sparse"]
        dense_rows = embeddings["dense"]
        expected = len(texts)

        # 稠密向量行数与输入不一致：同样退化为逐条编码，避免 IndexError
        if len(dense_rows) != expected:
            logger.warning(
                f"稠密向量行数({len(dense_rows)})与输入文本数({expected})不一致，退化为逐条编码")
            dense_rows = []
            sparse_dicts = []
            for t in texts:
                emb = model.encode_documents([t])
                dense_rows.append(emb["dense"][0])
                row = _sparse_to_dicts(emb["sparse"], 1)
                sparse_dicts.append(row[0] if row else {})
        else:
            sparse_dicts = _sparse_to_dicts(sparse, expected, model, texts)

        # 构造最终返回结果，稠密向量转列表（解决numpy数组不可序列化问题）
        result = {
            "dense": [emb.tolist() for emb in dense_rows],  # 嵌套列表，与输入文本一一对应
            "sparse": sparse_dicts  # 字典列表，模型已做L2归一化
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


def _sparse_to_dicts(sparse, expected: int, model=None, texts=None) -> list:
    """
    将 BGE-M3 返回的稀疏向量（CSR 矩阵）安全解析为 [{维度: 数值}, ...] 列表。
    兼容不同 SDK 版本返回结构，并做越界保护：
    - 支持 scipy CSR 矩阵（indptr/indices/data）
    - 支持纯 Python 字典/列表
    - 若 CSR 的行数与输入文本数不一致，退化为逐条重新编码，避免 IndexError
    """
    # 情况A：CSR 矩阵结构（scipy.sparse.csr_matrix）
    if hasattr(sparse, "indptr"):
        indptr = sparse.indptr
        indices = sparse.indices
        data = sparse.data
        row_count = len(indptr) - 1
        # 行数不匹配：逐条重新编码（避免 indptr[i+1] 越界）
        if row_count != expected:
            logger.warning(
                f"稀疏向量行数({row_count})与输入文本数({expected})不一致，退化为逐条编码")
            if model is not None and texts is not None:
                return _encode_row_by_row(model, texts)
            return [dict(zip((indices.tolist() or [])[indptr[i]:indptr[i+1]], data.tolist()[indptr[i]:indptr[i+1]]))
                    for i in range(min(row_count, expected))]
        return [
            dict(zip(
                indices[indptr[i]:indptr[i + 1]].tolist(),
                data[indptr[i]:indptr[i + 1]].tolist(),
            ))
            for i in range(expected)
        ]

    # 情况B：已是字典列表（部分 SDK 直接返回）
    if isinstance(sparse, list):
        out = []
        for item in sparse:
            if isinstance(item, dict):
                out.append({int(k): float(v) for k, v in item.items()})
            else:
                out.append(item)
        return out

    # 情况C：其他未知结构
    return list(sparse) if sparse is not None else [{}] * expected


def _encode_row_by_row(model, texts: list) -> list:
    """逐条编码文本，保证每条都有一条稀疏向量，避免 CSR 结构不匹配。"""
    out = []
    for t in texts:
        try:
            emb = model.encode_documents([t])
            row = _sparse_to_dicts(emb["sparse"], 1)
            out.append(row[0] if row else {})
        except Exception as e:
            logger.error(f"逐条编码文本失败: {e}")
            out.append({})
    return out


"""
核心设计亮点&适配说明：
1. 模型原生归一化：开启normalize_embeddings = True，自动对稠密+稀疏向量做L2归一化，完美适配Milvus IP内积检索（单位化后IP等价于余弦，计算更快）；
2. 彻底解决NumPy类型做key问题：sparse_indices加.tolist()，将np.int64转为Python原生int，满足字典key的可哈希要求，无报错风险；
3. 稀疏值适配序列化：sparse_data加.tolist()，将np.float32转为Python原生float，支持JSON写入/接口返回/Milvus入库等所有场景；
4. 单例模式优化：模型仅初始化一次，避免重复加载耗时耗资源，提升批量处理效率；
5. 格式匹配业务调用：返回dense嵌套列表、sparse字典列表，与vector_result["dense"][0]/sparse_vector["sparse"][0]取值逻辑完美契合；
6. 分级日志覆盖：从模型初始化、向量生成到异常报错，全流程日志记录，便于生产环境问题排查；
7. 入参合法性校验：防止空列表/非列表入参导致的内部报错，提升工具类健壮性。
"""