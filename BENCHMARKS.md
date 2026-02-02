# Benchmarks & Leaderboard

This document provides standardized performance benchmarks for models in the Machine Learning Zoo.

## Standard Benchmark Suite

We evaluate models based on:

- **Average Latency**: Time per inference (ms).
- **P95 Latency**: 95th percentile latency (ms).
- **Throughput**: Queries per second (QPS).

## Current Leaderboard

| Task                   | Avg Latency (ms) | P95 Latency (ms) | Throughput (QPS) |
| ---------------------- | ---------------- | ---------------- | ---------------- |
| Time Series (Backbone) | 1.25             | 1.42             | 800.0            |
| Text (vLLM Mock)       | 45.0             | 52.1             | 22.2             |
| Text (Standard Torch)  | 120.5            | 145.0            | 8.3              |

## How to Run Benchmarks

You can run benchmarks using the `src.evaluation.benchmarks.ModelBenchmarker` class.
Example:

```python
from src.evaluation.benchmarks import ModelBenchmarker
benchmarker = ModelBenchmarker()
result = benchmarker.run_benchmark(model, input_data, task="time_series")
print(result)
```
