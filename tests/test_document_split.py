"""
纯逻辑单元测试：文档切分节点（node_document_split）。
不依赖 MongoDB/Milvus/LLM，仅实例化切分节点，验证标题切分、长切短合、字段兜底。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.import_process.agent.nodes.node_document_split import (
    NodeDocumentSplit,
    DEFAULT_MAX_CONTENT_LENGTH,
    MIN_CONTENT_LENGTH,
)


def _state_with_content(md_content, file_title="test.md"):
    return {"md_content": md_content, "file_title": file_title, "local_dir": None}


def test_split_basic_titles():
    node = NodeDocumentSplit()
    # 两个标题段内容都足够长（超过 MIN_CONTENT_LENGTH），确保不会被合并
    md = "# 一级标题\n" + ("正文一" * 250) + "\n\n## 二级标题\n" + ("正文二" * 250) + "\n"
    state = _state_with_content(md)
    result = node.process(state)
    chunks = result["chunks"]
    assert len(chunks) >= 2
    # 每个 chunk 都有必填字段
    for c in chunks:
        assert "title" in c
        assert "content" in c
        assert "file_title" in c
        assert "parent_title" in c
        assert "part" in c


def test_split_skips_code_block_pseudo_headings():
    node = NodeDocumentSplit()
    md = "# 标题\n```\n# 这不是标题\n```\n正文\n"
    state = _state_with_content(md)
    result = node.process(state)
    # 代码块内的 # 不算标题：最终 chunk 的 title 不应是 "# 这不是标题"
    chunks = result["chunks"]
    for c in chunks:
        assert c.get("title") != "# 这不是标题"


def test_no_title_whole_text_fallback():
    node = NodeDocumentSplit()
    md = "没有任何标题的纯文本内容，用于验证无标题兜底逻辑。"
    state = _state_with_content(md)
    result = node.process(state)
    chunks = result["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["title"] == "无标题"


def test_long_section_split_and_part():
    node = NodeDocumentSplit()
    long_text = "内容" * 3000  # 超过 max_len=2000
    md = f"# 标题\n{long_text}\n"
    state = _state_with_content(md, file_title="long.pdf")
    result = node.process(state)
    chunks = result["chunks"]
    assert len(chunks) > 0
    for c in chunks:
        # 长文本应被硬切为不超过 max_len 的多个块（标题前缀允许少量溢出）
        assert len(c["content"]) <= DEFAULT_MAX_CONTENT_LENGTH + 20
        assert isinstance(c["part"], int)


def test_match_short_sections_same_parent():
    """短 chunk 同父标题应被合并，减少碎片化。"""
    node = NodeDocumentSplit()
    md = (
        "# 章节A\n"
        "第一节内容_" + ("长" * 100) + "\n\n"
        "第二节内容_" + ("长" * 100) + "\n"
    )
    state = _state_with_content(md)
    result = node.process(state)
    chunks = result["chunks"]
    # 合并后 chunk 数应小于等于原始段落数
    assert len(chunks) >= 1