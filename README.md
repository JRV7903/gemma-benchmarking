# Gemma Benchmarking

A lightweight framework for benchmarking Gemma models against other open models like Mistral on the MMLU dataset.

## Features

- MMLU dataset subset benchmarking
- Support for Gemma 2B and 7B models
- Comparison with Mistral 7B
- Automated pipeline for model loading, inference, and metrics calculation
- Visualization of benchmark results

## Installation

```bash
git clone https://github.com/yourusername/gemma-benchmarking.git
cd gemma-benchmarking

pip install -r requirements.txt
```

## Setup

Gemma models require authentication with a Hugging Face token. You can set it up using:

```bash
python scripts/setup_env.py --hf-token YOUR_HF_TOKEN

export HF_TOKEN=YOUR_HF_TOKEN
set HF_TOKEN=YOUR_HF_TOKEN
```

You can get your Hugging Face token from your [account settings](https://huggingface.co/settings/tokens).

## Usage

1. Configure the benchmark in `configs/benchmark_config.yaml`
2. Run the benchmark:

```bash
python scripts/run_benchmark.py
```

3. Visualize results using the provided script:

```bash
jupyter notebook notebooks/visualize_results.ipynb

```

## Project Structure

```
gemma-benchmarking/
├── configs/                  # Configuration files
│   ├── models.json           # Model configurations
│   └── benchmark_config.yaml # Benchmark parameters
├── src/                      # Source code
│   ├── data_loaders/         # Dataset loading utilities
│   ├── models/               # Model adapters
│   ├── evaluators/           # Metrics and evaluation
│   └── pipeline/             # Benchmark pipeline
├── scripts/                  # Command-line scripts
│   ├── run_benchmark.py      # Main benchmark script
│   └── setup_env.py          # Environment setup script
├── results/                  # Benchmark results
└── notebooks/                # Analysis notebooks
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Hugging Face account with access to Gemma models
- See `requirements.txt` for full dependencies

## Comparing Models

This framework allows you to benchmark and compare:

- **Gemma 2B**: Google's smaller Gemma model with 2 billion parameters
- **Gemma 7B**: Google's larger Gemma model with 7 billion parameters
- **Mistral 7B**: The Mistral model with 7 billion parameters

The benchmark provides accuracy metrics across different categories of the MMLU dataset, allowing for a comprehensive comparison of model capabilities.
