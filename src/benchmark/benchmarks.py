"""Benchmarking suite for Machine Learning Zoo."""

import logging
import time
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class ModelBenchmarker:
    """
    Standardized benchmarking tool for evaluating model performance.
    """

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def run_benchmark(
        self, model: Any, data: Any, task: str = "time_series", num_iters: int = 100
    ) -> Dict[str, Any]:
        """
        Run a benchmark on a given model.
        """
        model.eval()
        latencies = []

        with torch.no_grad():
            # Warmup
            for _ in range(10):
                model(data)

            # Benchmark
            for _ in range(num_iters):
                start = time.perf_counter()
                model(data)
                latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0

        result = {
            "task": task,
            "avg_latency_ms": round(avg_latency, 3),
            "p95_latency_ms": round(p95_latency, 3),
            "throughput_qps": round(throughput, 2),
            "iters": num_iters,
        }

        self.results.append(result)
        return result

    def generate_leaderboard(self) -> str:
        """
        Generate a markdown leaderboard from results.
        """
        if not self.results:
            return "No benchmark results yet."

        header = "| Task | Avg Latency (ms) | P95 Latency (ms) | Throughput (QPS) |\n"
        separator = (
            "|------|------------------|------------------|------------------|\n"
        )
        rows = ""
        for res in self.results:
            rows += f"| {res['task']} | {res['avg_latency_ms']} | {res['p95_latency_ms']} | {res['throughput_qps']} |\n"

        return header + separator + rows
