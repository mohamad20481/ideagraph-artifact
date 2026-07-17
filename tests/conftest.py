"""
Pytest fixtures + path wiring. Makes imports like `import production_optimization`
work regardless of where pytest is invoked from.
"""
import os
import sys
from pathlib import Path

# Ensure project root is importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reset module-level singletons between tests so state doesn't leak.
import pytest


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Clear global singletons between tests so bucket/cost/cb state is fresh."""
    try:
        import production_optimization as po
        po._SINGLETONS.clear()
    except Exception:
        pass
    yield
    try:
        import production_optimization as po
        po._SINGLETONS.clear()
    except Exception:
        pass
