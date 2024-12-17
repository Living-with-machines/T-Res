import pytest
from typing import Any, List
from typing_extensions import Final

INCLUDE_ALL_OPTION: Final[str] = "--include-all"
INCLUDE_RESOURCES_OPTION: Final[str] = "--include-resources"
INCLUDE_TRAIN_OPTION: Final[str] = "--include-train"
INCLUDE_APP_OPTION: Final[str] = "--include-app"

def pytest_addoption(parser):
    parser.addoption(INCLUDE_ALL_OPTION, action="store_true", default=False, help="run all tests")
    parser.addoption(INCLUDE_RESOURCES_OPTION, action="store_true", default=False, help="include tests dependent on large resources")
    parser.addoption(INCLUDE_TRAIN_OPTION, action="store_true", default=False, help="include model training tests")
    parser.addoption(INCLUDE_APP_OPTION, action="store_true", default=False, help="include HTTP API tests")

def pytest_collection_modifyitems(
        config,
        items: List[Any],
    ):
    if config.getoption(INCLUDE_ALL_OPTION):
        return

    if not config.getoption(INCLUDE_RESOURCES_OPTION):
        skipper = pytest.mark.skip(reason="Skip unless --include-resources or --include-all is given")
        for item in items:
            if "resources" in item.keywords:
                item.add_marker(skipper)
    
    if not config.getoption(INCLUDE_TRAIN_OPTION):
        skipper = pytest.mark.skip(reason="Skip unless --test-train or --include-all is given")
        for item in items:
            if "train" in item.keywords:
                item.add_marker(skipper)
    
    if not config.getoption(INCLUDE_APP_OPTION):
        skipper = pytest.mark.skip(reason="Skip unless --test-app or --include-all is given")
        for item in items:
            if "app" in item.keywords:
                item.add_marker(skipper)
