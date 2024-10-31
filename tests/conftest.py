from typing import Any, List
from typing_extensions import Final

NO_SKIP_OPTION: Final[str] = "--no-skip"

def pytest_addoption(parser):
    parser.addoption(NO_SKIP_OPTION, action="store_true", default=False, help="also run skipped tests")

def pytest_collection_modifyitems(config,
                                  items: List[Any]):
    if config.getoption(NO_SKIP_OPTION):
        for test in items:
            test.own_markers = [marker for marker in test.own_markers if marker.name not in ('skip', 'skipif')]