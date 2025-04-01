import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

MMLU_CATEGORIES = {
    "stem": [
        "high_school_physics", "college_physics", "high_school_chemistry", "college_chemistry", 
        "high_school_biology", "college_biology", "high_school_computer_science", "college_computer_science",
        "high_school_mathematics", "college_mathematics", "electrical_engineering", "astronomy"
    ],
    "humanities": [
        "high_school_european_history", "high_school_us_history", "high_school_world_history", 
        "philosophy", "high_school_government_and_politics", "world_religions"
    ],
    "social_sciences": [
        "sociology", "high_school_psychology", "professional_psychology", "high_school_microeconomics",
        "high_school_macroeconomics", "econometrics", "high_school_geography", "professional_law"
    ],
    "other": [
        "business_ethics", "management", "marketing", "nutrition", "professional_medicine",
        "professional_accounting", "miscellaneous"
    ]
}

class MMLUDataLoader:
    """
    Loader for the MMLU benchmark dataset with support for different subsets.
    """

    def __init__(
        self,
        subsets: List[str] = None,
        num_few_shot_examples: int = 5,
        max_samples_per_subset: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize the MMLU data loader.

        Args:
            subsets: List of MMLU subsets to load (stem, humanities, social_sciences, other)
                     If None, all subsets are loaded
            num_few_shot_examples: Number of examples to use for few-shot prompting
            max_samples_per_subset: Maximum number of samples per subject
            seed: Random seed for reproducibility
        """
        self.subsets = subsets if subsets else list(MMLU_CATEGORIES.keys())
        self.num_few_shot_examples = num_few_shot_examples
        self.max_samples_per_subset = max_samples_per_subset
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        for subset in self.subsets:
            if subset not in MMLU_CATEGORIES:
                raise ValueError(f"Invalid subset: {subset}. Must be one of {list(MMLU_CATEGORIES.keys())}")

        self.examples = {}
        self.few_shot_examples = {}

    def load_data(self) -> Dict[str, Dict]:
        """
        Load the MMLU dataset for all specified subsets.

        Returns:
            Dictionary mapping subject names to their respective examples
        """
        logger.info(f"Loading MMLU data for subsets: {self.subsets}")

        all_subjects = []
        for subset in self.subsets:
            all_subjects.extend(MMLU_CATEGORIES[subset])

        for subject in tqdm(all_subjects, desc="Loading MMLU subjects"):
            self._load_subject(subject)

        return self.examples

    def _load_subject(self, subject: str) -> None:
        """
        Load a specific subject from the MMLU dataset.

        Args:
            subject: The subject name to load
        """
        try:
            dev_dataset = load_dataset("cais/mmlu", subject, split="dev")

            test_dataset = load_dataset("cais/mmlu", subject, split="test")

            dev_df = pd.DataFrame(dev_dataset)
            test_df = pd.DataFrame(test_dataset)

            if self.max_samples_per_subset and len(test_df) > self.max_samples_per_subset:
                test_df = test_df.sample(self.max_samples_per_subset, random_state=self.seed)

            if len(dev_df) >= self.num_few_shot_examples:
                few_shot = dev_df.sample(self.num_few_shot_examples, random_state=self.seed)
            else:
                few_shot = dev_df
            self.examples[subject] = test_df
            self.few_shot_examples[subject] = few_shot

            logger.info(f"Loaded {len(test_df)} test examples and {len(few_shot)} few-shot examples for {subject}")

        except Exception as e:
            logger.error(f"Error loading subject {subject}: {str(e)}")
            raise

    def get_formatted_prompt(self, subject: str, question_row: pd.Series) -> str:
        """
        Format a prompt for a specific question with few-shot examples.

        Args:
            subject: The subject of the question
            question_row: The question data as a pandas Series

        Returns:
            Formatted prompt string with few-shot examples and the target question
        """
        choices = [question_row[f"choice_{i}"] for i in range(4)]
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

        few_shot_text = ""
        if subject in self.few_shot_examples:
            for _, example in self.few_shot_examples[subject].iterrows():
                example_choices = [example[f"choice_{i}"] for i in range(4)]
                example_choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(example_choices)])
                answer_letter = chr(65 + example['answer'])

                few_shot_text += f"Question: {example['question']}\n{example_choices_text}\nAnswer: {answer_letter}\n\n"

        prompt = f"{few_shot_text}Question: {question_row['question']}\n{choices_text}\nAnswer:"

        return prompt

    def get_data_by_category(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get the loaded data organized by category.

        Returns:
            Dictionary with categories as keys and subjects with their dataframes as values
        """
        result = {}
        for category, subjects in MMLU_CATEGORIES.items():
            if category in self.subsets:
                result[category] = {subject: self.examples[subject] for subject in subjects
                                   if subject in self.examples}
        return result
