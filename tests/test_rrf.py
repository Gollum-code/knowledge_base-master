"""
纯逻辑单元测试：RRF 倒数排名融合算法。
不依赖外部服务，仅导入 node_rrf 中的纯算法函数。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.query_process.agent.nodes.node_rrf import reciprocal_rank_fusion, _as_entity_list


def test_rrf_merge_common_docs_on_top():
    source1 = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    source2 = [{"chunk_id": "b"}, {"chunk_id": "c"}, {"chunk_id": "d"}]
    result = reciprocal_rank_fusion([(source1, 1.0), (source2, 1.0)], k=60)
    # 同时出现在两路中的 b、c 应排在前列
    ids = [doc["chunk_id"] for doc, _ in result]
    assert ids.index("b") < ids.index("d")
    assert ids.index("c") < ids.index("d")
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_weight_respected():
    # x 在 source1 中排第 1，在 source2 中排第 3 -> 权重不同导致融合分数不同
    source_a = [{"chunk_id": "x"}, {"chunk_id": "y"}, {"chunk_id": "z"}]
    source_b = [{"chunk_id": "y"}, {"chunk_id": "z"}, {"chunk_id": "x"}]
    result_high_first = reciprocal_rank_fusion([(source_a, 3.0), (source_b, 1.0)], k=60)
    result_high_second = reciprocal_rank_fusion([(source_a, 1.0), (source_b, 3.0)], k=60)
    score_a = {doc["chunk_id"]: score for doc, score in result_high_first}
    score_b = {doc["chunk_id"]: score for doc, score in result_high_second}
    # x 在第一路（排1）权重更高时，融合分应更高
    assert score_a["x"] != score_b["x"]


def test_rrf_max_results_truncation():
    source1 = [{"chunk_id": f"c{i}"} for i in range(10)]
    result = reciprocal_rank_fusion([(source1, 1.0)], k=60, max_results=3)
    assert len(result) <= 3


def test_rrf_skips_missing_ids():
    source1 = [{"chunk_id": "a"}, {"no_id": "b"}]
    result = reciprocal_rank_fusion([(source1, 1.0)], k=60)
    ids = [doc["chunk_id"] for doc, _ in result]
    assert ids == ["a"]


def test_as_entity_list_flat_dicts():
    items = [{"chunk_id": "a", "content": "x"}, {"chunk_id": "b", "content": "y"}]
    out = _as_entity_list(items)
    assert len(out) == 2
    assert out[0]["content"] == "x"


def test_as_entity_list_nested():
    items = [{"entity": {"chunk_id": "a", "content": "x"}, "distance": 0.9}]
    out = _as_entity_list(items)
    assert out[0]["chunk_id"] == "a"
    assert out[0]["score"] == 0.9


def test_as_entity_list_empty():
    assert _as_entity_list(None) == []
    assert _as_entity_list([]) == []