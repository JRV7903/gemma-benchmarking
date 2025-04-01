import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def calculate_metrics(
    true_labels: List[int],
    predicted_labels: List[int],
    metrics: List[str] = None,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for classification results.

    Args:
        true_labels: List of ground truth labels
        predicted_labels: List of predicted labels
        metrics: List of metrics to calculate

    Returns:
        Dictionary with metric names as keys and scores as values
    """
    if metrics is None:
        metrics = ["accuracy", "f1_score", "precision", "recall"]

    results = {}

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    valid_indices = ~np.isnan(predicted_labels)
    if not np.all(valid_indices):
        print(f"Warning: {np.sum(~valid_indices)} missing predictions. Using only valid predictions for metrics.")
        true_labels = true_labels[valid_indices]
        predicted_labels = predicted_labels[valid_indices]

    if len(true_labels) == 0:
        return {metric: 0.0 for metric in metrics}

    for metric in metrics:
        if metric == "accuracy":
            results[metric] = accuracy_score(true_labels, predicted_labels)
        elif metric == "f1_score":
            results[metric] = f1_score(true_labels, predicted_labels, average="macro")
        elif metric == "precision":
            results[metric] = precision_score(true_labels, predicted_labels, average="macro")
        elif metric == "recall":
            results[metric] = recall_score(true_labels, predicted_labels, average="macro")

    return results

def process_results(
    results_df: pd.DataFrame,
    group_by: List[str] = None,
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Process benchmark results and calculate metrics for different groupings.

    Args:
        results_df: DataFrame with benchmark results
        group_by: Columns to group results by
        metrics: List of metrics to calculate

    Returns:
        DataFrame with aggregated metrics
    """
    if group_by is None:
        group_by = ["model", "category", "subject"]

    if metrics is None:
        metrics = ["accuracy", "f1_score"]

    required_cols = group_by + ["true_answer", "predicted_answer"]
    for col in required_cols:
        if col not in results_df.columns:
            raise ValueError(f"Required column '{col}' not found in results DataFrame")

    if results_df["true_answer"].dtype == "object":
        results_df["true_answer_idx"] = results_df["true_answer"].apply(lambda x: ord(x) - ord('A') if x else None)
    else:
        results_df["true_answer_idx"] = results_df["true_answer"]

    if results_df["predicted_answer"].dtype == "object":
        results_df["predicted_answer_idx"] = results_df["predicted_answer"].apply(lambda x: ord(x) - ord('A') if x else np.nan)
    else:
        results_df["predicted_answer_idx"] = results_df["predicted_answer"]

    grouped = results_df.groupby(group_by)

    def compute_group_metrics(group):
        return pd.Series(
            calculate_metrics(
                group["true_answer_idx"].tolist(),
                group["predicted_answer_idx"].tolist(),
                metrics=metrics
            )
        )

    results = grouped.apply(compute_group_metrics).reset_index()

    counts = grouped.size().reset_index(name="total_count")

    results = results.merge(counts, on=group_by)

    return results

def calculate_confidence_intervals(
    results_df: pd.DataFrame,
    metric_col: str = "accuracy",
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Calculate confidence intervals for metrics using a bootstrap approach.

    Args:
        results_df: DataFrame with results (one row per example)
        metric_col: The metric to calculate confidence intervals for
        confidence: Confidence level (between 0 and 1)

    Returns:
        DataFrame with lower and upper confidence bounds
    """
    from scipy import stats

    def bootstrap_interval(group, n_bootstraps=1000):
        true_metric = calculate_metrics(
            group["true_answer_idx"].tolist(),
            group["predicted_answer_idx"].tolist(),
            metrics=[metric_col]
        )[metric_col]

        bootstrap_metrics = []
        for _ in range(n_bootstraps):
            sample_idx = np.random.choice(len(group), len(group), replace=True)
            sample_true = group["true_answer_idx"].iloc[sample_idx].tolist()
            sample_pred = group["predicted_answer_idx"].iloc[sample_idx].tolist()

            bootstrap_metric = calculate_metrics(
                sample_true, sample_pred, metrics=[metric_col]
            )[metric_col]
            bootstrap_metrics.append(bootstrap_metric)

        lower_bound, upper_bound = np.percentile(
            bootstrap_metrics, [(1-confidence)*100/2, 100-(1-confidence)*100/2]
        )

        return pd.Series({
            f"{metric_col}": true_metric,
            f"{metric_col}_lower": lower_bound,
            f"{metric_col}_upper": upper_bound
        })

    group_cols = [col for col in results_df.columns if col in ["model", "category", "subject"]]
    result = results_df.groupby(group_cols).apply(bootstrap_interval).reset_index()

    return result
