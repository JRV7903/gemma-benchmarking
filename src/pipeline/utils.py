import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def load_results(results_path: str) -> pd.DataFrame:
    """
    Load benchmark results from a CSV file.

    Args:
        results_path: Path to results CSV file

    Returns:
        DataFrame containing the results
    """
    return pd.read_csv(results_path)

def create_visualizations(
    metrics_df: pd.DataFrame,
    output_dir: str,
    by_category: bool = True,
    by_subject: bool = True
):
    """
    Create visualizations from benchmark results.

    Args:
        metrics_df: DataFrame with processed metrics
        output_dir: Directory to save visualizations
        by_category: Whether to create category-level visualizations
        by_subject: Whether to create subject-level visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")

    _create_model_comparison_chart(metrics_df, output_dir)

    if by_category:
        _create_category_comparison(metrics_df, output_dir)

    if by_subject:
        _create_subject_comparison(metrics_df, output_dir)

def _create_model_comparison_chart(metrics_df: pd.DataFrame, output_dir: str):
    """Create a bar chart comparing overall model performance."""
    model_metrics = metrics_df.groupby("model")["accuracy"].mean().reset_index()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="model", y="accuracy", data=model_metrics)
    ax.set_title("Overall Model Accuracy on MMLU", fontsize=16)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_ylim(0, 1)

    for i, row in model_metrics.iterrows():
        ax.text(i, row["accuracy"]+0.01, f"{row['accuracy']:.3f}",
                ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "model_comparison.pdf"))
    plt.close()

def _create_category_comparison(metrics_df: pd.DataFrame, output_dir: str):
    """Create charts comparing model performance by category."""
    cat_metrics = metrics_df.groupby(["model", "category"])["accuracy"].mean().reset_index()

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="accuracy", y="category", hue="model", data=cat_metrics)
    ax.set_title("Model Accuracy by Category", fontsize=16)
    ax.set_xlabel("Accuracy", fontsize=14)
    ax.set_ylabel("Category", fontsize=14)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_comparison.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "category_comparison.pdf"))
    plt.close()

    for model in cat_metrics["model"].unique():
        model_data = cat_metrics[cat_metrics["model"] == model]

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x="category", y="accuracy", data=model_data)
        ax.set_title(f"{model} - Accuracy by Category", fontsize=16)
        ax.set_xlabel("Category", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_ylim(0, 1)

        for i, row in model_data.iterrows():
            ax.text(i % len(model_data), row["accuracy"]+0.01, f"{row['accuracy']:.3f}",
                    ha='center', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_category_breakdown.png"), dpi=300)
        plt.close()

def _create_subject_comparison(metrics_df: pd.DataFrame, output_dir: str):
    """Create charts comparing model performance by subject."""
    top_subjects_dir = os.path.join(output_dir, "subjects")
    os.makedirs(top_subjects_dir, exist_ok=True)

    for model in metrics_df["model"].unique():
        model_subjects = metrics_df[metrics_df["model"] == model].copy()
        model_subjects = model_subjects.sort_values(by="accuracy", ascending=False)

        plt.figure(figsize=(12, 8))
        top10 = model_subjects.head(10)
        sns.barplot(x="accuracy", y="subject", data=top10)
        plt.title(f"{model} - Top 10 Subjects by Accuracy", fontsize=16)
        plt.xlabel("Accuracy", fontsize=14)
        plt.ylabel("Subject", fontsize=14)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(top_subjects_dir, f"{model}_top10_subjects.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(12, 8))
        bottom10 = model_subjects.tail(10)
        sns.barplot(x="accuracy", y="subject", data=bottom10)
        plt.title(f"{model} - Bottom 10 Subjects by Accuracy", fontsize=16)
        plt.xlabel("Accuracy", fontsize=14)
        plt.ylabel("Subject", fontsize=14)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(top_subjects_dir, f"{model}_bottom10_subjects.png"), dpi=300)
        plt.close()

    interesting_subjects = metrics_df.groupby("subject")["total_count"].mean().nlargest(5).index.tolist()

    for subject in interesting_subjects:
        subject_data = metrics_df[metrics_df["subject"] == subject]
        if len(subject_data) > 1:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="model", y="accuracy", data=subject_data)
            plt.title(f"Model Comparison: {subject}", fontsize=16)
            plt.xlabel("Model", fontsize=14)
            plt.ylabel("Accuracy", fontsize=14)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(top_subjects_dir, f"comparison_{subject}.png"), dpi=300)
            plt.close()

def export_results_to_latex(metrics_df: pd.DataFrame, output_path: str):
    """
    Export benchmark results as LaTeX tables.

    Args:
        metrics_df: DataFrame with processed metrics
        output_path: Path to save the LaTeX file
    """
    with open(output_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Overall Model Performance on MMLU}\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Model & Accuracy \\\\\n")
        f.write("\\midrule\n")

        model_metrics = metrics_df.groupby("model")["accuracy"].mean().reset_index()
        for _, row in model_metrics.iterrows():
            f.write(f"{row['model']} & {row['accuracy']:.4f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model Performance by Category}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")

        categories = sorted(metrics_df["category"].unique())

        f.write("Model")
        for category in categories:
            f.write(f" & {category}")
        f.write(" \\\\\n")
        f.write("\\midrule\n")

        cat_metrics = metrics_df.groupby(["model", "category"])["accuracy"].mean().reset_index()
        for model in sorted(metrics_df["model"].unique()):
            f.write(f"{model}")
            for category in categories:
                model_cat = cat_metrics[(cat_metrics["model"] == model) &
                                       (cat_metrics["category"] == category)]
                if not model_cat.empty:
                    acc = model_cat["accuracy"].values[0]
                    f.write(f" & {acc:.4f}")
                else:
                    f.write(" & -")
            f.write(" \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def export_results_to_html(metrics_df: pd.DataFrame, output_path: str):
    """
    Export benchmark results as HTML tables and charts.

    Args:
        metrics_df: DataFrame with processed metrics
        output_path: Path to save the HTML file
    """
    overall_metrics = metrics_df.groupby("model")[["accuracy", "f1_score"]].mean().reset_index()
    category_metrics = metrics_df.groupby(["model", "category"])["accuracy"].mean().reset_index()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MMLU Benchmark Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .header {{ margin-bottom: 20px; }}
            .section {{ margin-top: 40px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>MMLU Benchmark Results</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
        </div>

        <div class="section">
            <h2>Overall Model Performance</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>F1 Score</th>
                </tr>
    """

    for _, row in overall_metrics.iterrows():
        html += f"""
                <tr>
                    <td>{row['model']}</td>
                    <td>{row['accuracy']:.4f}</td>
                    <td>{row['f1_score']:.4f}</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <div class="section">
            <h2>Performance by Category</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Category</th>
                    <th>Accuracy</th>
                </tr>
    """

    for _, row in category_metrics.iterrows():
        html += f"""
                <tr>
                    <td>{row['model']}</td>
                    <td>{row['category']}</td>
                    <td>{row['accuracy']:.4f}</td>
                </tr>
        """

    html += """
            </table>
        </div>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html)
