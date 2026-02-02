import unittest
import torch
from src.benchmark.benchmarks import ModelBenchmarker


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.lin(x)


class TestBenchmarking(unittest.TestCase):
    def test_run_benchmark(self):
        model = MockModel()
        benchmarker = ModelBenchmarker()
        data = torch.randn(1, 10)

        result = benchmarker.run_benchmark(model, data, num_iters=20)

        self.assertIn("avg_latency_ms", result)
        self.assertIn("throughput_qps", result)
        self.assertEqual(result["iters"], 20)

        print(f"\nBenchmark Result: {result}")

    def test_leaderboard_gen(self):
        benchmarker = ModelBenchmarker()
        # Add some dummy results
        benchmarker.results.append(
            {
                "task": "dummy",
                "avg_latency_ms": 1.0,
                "p95_latency_ms": 1.2,
                "throughput_qps": 1000.0,
                "iters": 10,
            }
        )

        leaderboard = benchmarker.generate_leaderboard()
        self.assertIn("| Task |", leaderboard)
        self.assertIn("| dummy |", leaderboard)
        print(f"\nGenerated Leaderboard:\n{leaderboard}")


if __name__ == "__main__":
    unittest.main()
