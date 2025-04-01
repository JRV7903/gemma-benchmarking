"""
Gemma MMLU Benchmark Runner Script

This script runs benchmarks for Gemma models on the MMLU dataset.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.benchmark_runner import BenchmarkRunner
from src.pipeline.utils import create_visualizations, load_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MMLU benchmarks for Gemma models")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_config.yaml",
        help="Path to the benchmark configuration YAML file"
    )

    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only create visualizations from existing results"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        help="Path to results directory (for visualization only)"
    )

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config_path = Path(args.config).resolve()

    if args.visualize_only:
        if not args.results_dir:
            logging.error("Must provide --results-dir when using --visualize-only")
            sys.exit(1)

        results_path = Path(args.results_dir).resolve()
        metrics_path = results_path / "metrics.csv"

        if not metrics_path.exists():
            logging.error(f"Metrics file not found: {metrics_path}")
            sys.exit(1)

        logging.info(f"Loading results from {metrics_path}")
        metrics_df = load_results(str(metrics_path))

        vis_dir = results_path / "visualizations"
        logging.info(f"Creating visualizations in {vis_dir}")
        create_visualizations(metrics_df, str(vis_dir))

        logging.info("Visualization complete!")
    else:
        if not config_path.exists():
            logging.error(f"Config file not found: {config_path}")
            sys.exit(1)

        logging.info(f"Starting benchmark with config: {config_path}")

        runner = BenchmarkRunner(str(config_path))
        runner.run()

        logging.info("Benchmark complete!")

if __name__ == "__main__":
    main()
