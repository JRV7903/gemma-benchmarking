import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from .models.model_factory import ModelFactory
from .data_loaders.mmlu_loader import MMLUDataLoader
from .pipeline.benchmark_runner import BenchmarkRunner
