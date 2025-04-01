import os
import json
import yaml
import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch

from ..data_loaders.mmlu_loader import MMLUDataLoader
from ..models.model_factory import ModelFactory
from ..evaluators.metrics import process_results

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """
    Class to run benchmarks on language models with the MMLU dataset.
    """

    def __init__(self, config_path: str):
        """
        Initialize the benchmark runner.

        Args:
            config_path: Path to the benchmark configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.models_config = self._load_models_config()
        self.results = []

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self._set_seed(self.config["benchmark"]["seed"])

    def _load_config(self, config_path: str) -> Dict:
        """Load the benchmark configuration from a YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _load_models_config(self) -> Dict:
        """Load model configurations from JSON file."""
        config_dir = os.path.dirname(self.config_path)
        models_path = os.path.join(config_dir, "models.json")

        with open(models_path, 'r') as f:
            models_config = json.load(f)
        return models_config

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self):
        """Run the benchmark for all configured models and datasets."""
        logger.info("Starting benchmark run")

        dataset_config = self.config["dataset"]
        data_loader = MMLUDataLoader(
            subsets=dataset_config["subsets"],
            num_few_shot_examples=dataset_config["num_few_shot_examples"],
            max_samples_per_subset=dataset_config["max_samples_per_subset"],
            seed=self.config["benchmark"]["seed"]
        )
        data_loader.load_data()

        for model_name in self.config["models"]:
            if model_name not in self.models_config:
                logger.warning(f"Model '{model_name}' not found in models.json. Skipping.")
                continue

            logger.info(f"Running benchmark for model: {model_name}")

            try:
                model_config = self.models_config[model_name]
                model = ModelFactory.create_model(
                    model_config,
                    device=self.config["hardware"]["device"],
                    precision=self.config["hardware"]["precision"]
                )
                batch_size = ModelFactory.get_batch_size_for_model(model_config)

                model_results = self._run_model_benchmark(
                    model_name, model, data_loader, batch_size
                )

                self.results.extend(model_results)

            except Exception as e:
                logger.error(f"Error running benchmark for model '{model_name}': {str(e)}")

        self._save_results()

    def _run_model_benchmark(
        self,
        model_name: str,
        model,
        data_loader: MMLUDataLoader,
        batch_size: int
    ) -> List[Dict]:
        """
        Run the benchmark for a specific model.

        Args:
            model_name: Name of the model being evaluated
            model: The model instance
            data_loader: MMLUDataLoader instance with loaded data
            batch_size: Batch size for model inference

        Returns:
            List of result dictionaries for each example
        """
        results = []

        categorized_data = data_loader.get_data_by_category()

        for category, subjects in categorized_data.items():
            logger.info(f"Evaluating {model_name} on category: {category}")

            for subject, df in subjects.items():
                logger.info(f"  Subject: {subject} ({len(df)} examples)")

                prompts = []
                examples = []

                for _, row in df.iterrows():
                    prompt = data_loader.get_formatted_prompt(subject, row)
                    prompts.append(prompt)
                    examples.append({
                        "model": model_name,
                        "category": category,
                        "subject": subject,
                        "question_id": row.get("question_id", f"{subject}_{_}"),
                        "question": row["question"],
                        "true_answer": chr(65 + row["answer"]),
                        "prompt": prompt
                    })

                try:
                    all_responses = model.batch_generate(
                        prompts,
                        batch_size=batch_size,
                        max_new_tokens=5,
                        temperature=0.0
                    )

                    for i, response in enumerate(all_responses):
                        predicted_answer = model.extract_answer_from_response(response)

                        examples[i]["response"] = response
                        examples[i]["predicted_answer"] = predicted_answer
                        results.append(examples[i])

                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {subject}: {str(e)}")

                    for example in examples:
                        example["response"] = None
                        example["predicted_answer"] = None
                        results.append(example)

        return results

    def _save_results(self):
        """Process results and save them to files."""
        logger.info("Processing and saving results")

        results_df = pd.DataFrame(self.results)

        output_dir = self.config["benchmark"].get("output_dir", "results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"benchmark_results_{timestamp}"
        os.makedirs(output_path, exist_ok=True)

        raw_results_path = output_path / "raw_results.csv"
        results_df.to_csv(raw_results_path, index=False)
        logger.info(f"Raw results saved to {raw_results_path}")

        metrics = self.config["evaluation"]["metrics"]
        processed_results = process_results(results_df, metrics=metrics)

        metrics_path = output_path / "metrics.csv"
        processed_results.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")

        with open(output_path / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

        self._create_summary(results_df, processed_results, output_path)

    def _create_summary(self, raw_results: pd.DataFrame, metrics: pd.DataFrame, output_path: Path):
        """Create a summary file with key results."""
        summary_path = output_path / "summary.txt"

        with open(summary_path, "w") as f:
            f.write("MMLU Benchmark Summary\n")
            f.write("=====================\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {self.config_path}\n\n")

            f.write("Models evaluated:\n")
            for model in self.config["models"]:
                f.write(f"- {model}\n")
            f.write("\n")

            f.write("Dataset:\n")
            f.write(f"- Subsets: {', '.join(self.config['dataset']['subsets'])}\n")
            f.write(f"- Few-shot examples: {self.config['dataset']['num_few_shot_examples']}\n")
            f.write(f"- Max samples per subset: {self.config['dataset']['max_samples_per_subset']}\n\n")

            f.write("Results Overview:\n")
            f.write("----------------\n\n")

            f.write("Overall Accuracy by Model:\n")
            model_accuracy = metrics.groupby("model")["accuracy"].mean().reset_index()
            for _, row in model_accuracy.iterrows():
                f.write(f"- {row['model']}: {row['accuracy']:.4f}\n")
            f.write("\n")

            f.write("Accuracy by Category:\n")
            category_accuracy = metrics.groupby(["model", "category"])["accuracy"].mean().reset_index()
            for model in self.config["models"]:
                f.write(f"\n{model}:\n")
                for category in self.config["dataset"]["subsets"]:
                    model_cat = category_accuracy[(category_accuracy["model"] == model) &
                                                 (category_accuracy["category"] == category)]
                    if not model_cat.empty:
                        acc = model_cat["accuracy"].values[0]
                        f.write(f"- {category}: {acc:.4f}\n")

            completion_count = (raw_results["predicted_answer"].notna().groupby(
                [raw_results["model"], raw_results["category"]]
            ).mean() * 100).reset_index()

            f.write("\nCompletion Rates (%):\n")
            for model in self.config["models"]:
                f.write(f"\n{model}:\n")
                for category in self.config["dataset"]["subsets"]:
                    model_cat = completion_count[(completion_count["model"] == model) &
                                               (completion_count["category"] == category)]
                    if not model_cat.empty:
                        rate = model_cat[0].values[0]
                        f.write(f"- {category}: {rate:.1f}%\n")

        logger.info(f"Summary saved to {summary_path}")
