# Gemma MMLU Benchmark Results Visualization

This document provides instructions for visualizing the results from the Gemma MMLU benchmarking tool. You can use these code snippets in a Jupyter notebook.

## Setup and Imports

```python
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.pipeline.utils import (
    load_results,
    create_visualizations,
    export_results_to_html,
    export_results_to_latex
)
```

## Load Benchmark Results

```python
results_dir = '../results/benchmark_results_YYYYMMDD_HHMMSS'

raw_results_path = os.path.join(results_dir, 'raw_results.csv')
metrics_path = os.path.join(results_dir, 'metrics.csv')

raw_results = pd.read_csv(raw_results_path)
metrics = pd.read_csv(metrics_path)

print(f"Loaded {len(raw_results)} raw results and {len(metrics)} metric entries")
```

## Examine the Data

```python
raw_results.head()

metrics.head()
```

## Overall Model Performance

```python
model_accuracy = metrics.groupby('model')['accuracy'].mean().reset_index()
model_accuracy['accuracy'] = model_accuracy['accuracy'] * 100

plt.figure(figsize=(10, 6))
sns.barplot(x='model', y='accuracy', data=model_accuracy)
plt.title('Overall Model Accuracy on MMLU', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.ylim(0, 100)

for i, row in model_accuracy.iterrows():
    plt.text(i, row['accuracy']+1, f"{row['accuracy']:.1f}%", ha='center', fontsize=12)

plt.tight_layout()
plt.show()
```

## Performance by Category

```python
cat_metrics = metrics.groupby(['model', 'category'])['accuracy'].mean().reset_index()
cat_metrics['accuracy'] = cat_metrics['accuracy'] * 100

plt.figure(figsize=(12, 8))
sns.barplot(x='accuracy', y='category', hue='model', data=cat_metrics)
plt.title('Model Accuracy by Category', fontsize=16)
plt.xlabel('Accuracy (%)', fontsize=14)
plt.ylabel('Category', fontsize=14)
plt.xlim(0, 100)
plt.legend(title='Model', loc='lower right')

plt.tight_layout()
plt.show()
```

## Subject Analysis

```python
for model in metrics['model'].unique():
    model_subjects = metrics[metrics['model'] == model].copy()
    model_subjects['accuracy'] = model_subjects['accuracy'] * 100
    model_subjects = model_subjects.sort_values(by='accuracy', ascending=False)

    plt.figure(figsize=(10, 6))
    top5 = model_subjects.head(5)
    sns.barplot(x='subject', y='accuracy', data=top5)
    plt.title(f"{model} - Top 5 Subjects", fontsize=16)
    plt.xlabel('Subject', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')

    for i, row in top5.reset_index().iterrows():
        plt.text(i, row['accuracy']+1, f"{row['accuracy']:.1f}%", ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    bottom5 = model_subjects.tail(5)
    sns.barplot(x='subject', y='accuracy', data=bottom5)
    plt.title(f"{model} - Bottom 5 Subjects", fontsize=16)
    plt.xlabel('Subject', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')

    for i, row in bottom5.reset_index().iterrows():
        plt.text(i, row['accuracy']+1, f"{row['accuracy']:.1f}%", ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()
```

## Performance Gap Analysis

```python
if len(metrics['model'].unique()) > 1:
    pivot_cat = cat_metrics.pivot(index='category', columns='model', values='accuracy')

    models = list(metrics['model'].unique())
    if len(models) >= 2:
        pivot_cat['gap'] = pivot_cat[models[0]] - pivot_cat[models[1]]

        plt.figure(figsize=(10, 6))
        pivot_cat = pivot_cat.sort_values(by='gap')
        sns.barplot(x=pivot_cat.index, y=pivot_cat['gap'])
        plt.title(f'Performance Gap: {models[0]} vs {models[1]}', fontsize=16)
        plt.xlabel('Category', fontsize=14)
        plt.ylabel('Accuracy Difference (%)', fontsize=14)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
```

## Generate Summary Report

```python
html_path = os.path.join(results_dir, 'report.html')
export_results_to_html(metrics, html_path)
print(f"HTML report saved to: {html_path}")

latex_path = os.path.join(results_dir, 'tables.tex')
export_results_to_latex(metrics, latex_path)
print(f"LaTeX tables saved to: {latex_path}")
```

## Create Automated Visualizations

```python
vis_dir = os.path.join(results_dir, 'visualizations')
create_visualizations(metrics, vis_dir)
print(f"Visualizations saved to: {vis_dir}")
```
