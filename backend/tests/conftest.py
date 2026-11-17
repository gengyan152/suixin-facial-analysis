import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import pytest


def pytest_addoption(parser):
    parser.addoption("--live",action="store_true",help="Run resident CUDA/FaceLLM integration checks too")


def pytest_collection_modifyitems(config,items):
    if not config.getoption("--live"):
        for item in items:
            if "live" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use --live for downloaded real-model tests"))


@pytest.fixture(scope="session")
def live_pipeline():
    from app.config import Settings
    from app.pipeline import Pipeline
    pipeline=Pipeline(Settings())
    try:
        pipeline.load()
        pipeline.warmup()
        yield pipeline
    finally:
        pipeline.close()
