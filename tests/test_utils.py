"""
纯逻辑单元测试：任务状态追踪（内存后端）、RRF、文档切分、商品对齐、工具函数。
这些用例不依赖外部服务（MongoDB/Milvus/LLM），可在 CI 中稳定运行。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 确保不连接真实外部服务：仅导入被测试的纯逻辑模块
from app.utils.milvus_utils import escape_milvus_string
from app.utils.retry_utils import with_retry, retry_on


# ---------------- Milvus 转义 ----------------

def test_escape_milvus_string_quotes():
    assert escape_milvus_string('商品"名称') == '商品\\"名称'


def test_escape_milvus_string_control_chars():
    assert escape_milvus_string("测试\n文档\t") == "测试 文档 "


def test_escape_milvus_string_none():
    assert escape_milvus_string(None) == ""


def test_escape_milvus_string_backslash():
    assert escape_milvus_string("a\\b") == "a\\\\b"


# ---------------- 重试工具 ----------------

class _FakeFlaky:
    """前 n 次抛连接错误，之后成功。"""

    def __init__(self, fail_times=1, exc=ConnectionError):
        self.calls = 0
        self.fail_times = fail_times
        self.exc = exc

    def __call__(self, *a, **k):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc("temporary")
        return "ok"


def test_with_retry_success_after_failure():
    fn = _FakeFlaky(fail_times=2, exc=ConnectionError)
    assert with_retry(fn, max_attempts=4, base_delay=0.01) == "ok"
    assert fn.calls == 3


def test_with_retry_gives_up_and_raises():
    fn = _FakeFlaky(fail_times=10, exc=ConnectionError)
    try:
        with_retry(fn, max_attempts=2, base_delay=0.01)
        assert False, "should have raised"
    except ConnectionError:
        pass
    assert fn.calls == 2


def test_with_retry_non_retryable_raises_immediately():
    fn = _FakeFlaky(fail_times=10, exc=ValueError)
    try:
        with_retry(fn, max_attempts=5, base_delay=0.01)
        assert False, "should have raised"
    except ValueError:
        pass
    # 非可重试异常：不重试，只调用一次
    assert fn.calls == 1


def test_retry_on_decorator():
    fn = _FakeFlaky(fail_times=1, exc=ConnectionError)

    @retry_on(max_attempts=3, base_delay=0.01)
    def wrapped():
        return fn()

    assert wrapped() == "ok"
    assert fn.calls == 2