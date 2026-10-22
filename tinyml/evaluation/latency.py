import time

import torch


class LatencyBenchmark:

    def __init__(
        self,
        warmup: int = 3,
        iterations: int = 10,
    ):
        self.warmup = warmup
        self.iterations = iterations

    def measure(
        self,
        model,
        forward_fn,
    ):
        model.eval()

        with torch.no_grad():
            for _ in range(self.warmup):
                forward_fn()

            start = time.perf_counter()

            for _ in range(self.iterations):
                forward_fn()

            elapsed = time.perf_counter() - start

        latency = elapsed / self.iterations

        return {
            "latency_seconds": latency,
            "latency_ms": latency * 1000.0,
            "fps": (
                1.0 / latency
                if latency > 0
                else 0.0
            ),
            "iterations": self.iterations,
        }