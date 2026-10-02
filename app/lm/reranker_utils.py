from FlagEmbedding import FlagReranker
from app.conf.reranker_config import reranker_config
from app.core.logger import logger

_reranker_model = None

def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        try:
            _reranker_model = FlagReranker(
                model_name_or_path=reranker_config.bge_reranker_large,
                device=reranker_config.bge_reranker_device,
                use_fp16=reranker_config.bge_reranker_fp16
            )
        except Exception as e:
            # GPU 初始化失败：自动降级为 CPU + FP32
            dev = reranker_config.bge_reranker_device or "cpu"
            if str(dev).startswith("cuda"):
                logger.warning(f"重排序模型在 {dev} 初始化失败（{e}），自动降级为 CPU")
                _reranker_model = FlagReranker(
                    model_name_or_path=reranker_config.bge_reranker_large,
                    device="cpu",
                    use_fp16=False,
                )
            else:
                raise
    return _reranker_model