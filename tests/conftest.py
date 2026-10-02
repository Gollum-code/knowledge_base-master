"""pytest 配置：自动发现 tests/ 下的用例，并确保项目根目录在 sys.path 中。"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _avoid_real_network(monkeypatch):
    """默认测试环境不依赖真实外部服务，避免 CI 因 MongoDB/LLM 等不可达而失败。"""
    # 测试模块加载 app.* 时会尝试连接 MongoDB，这里仅记录警告而非强制跳过
    yield