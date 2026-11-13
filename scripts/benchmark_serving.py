from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import threading
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Sequence

import psutil
import torch

from regenrec.data import SequenceExample, load_sequence_examples_jsonl
from regenrec.serving import RecommendationEngine, ServingSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the real frozen online recommendation path"
    )
    parser.add_argument("--beam-size", action="append", type=int, dest="beam_sizes")
    parser.add_argument("--batch-size", action="append", type=int, dest="batch_sizes")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument(
        "--long-tail-mode", choices=("off", "mild", "aggressive"), default="off"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/serving_benchmark/report.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("docs/PERFORMANCE_REPORT.md")
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile values cannot be empty")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class PeakMemorySampler:
    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.process = psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> PeakMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def batched_examples(
    examples: Sequence[SequenceExample], batch_size: int, iterations: int
) -> list[list[SequenceExample]]:
    stream = cycle(examples)
    return [list(islice(stream, batch_size)) for _ in range(iterations)]


def benchmark_scenario(
    engine: RecommendationEngine,
    examples: Sequence[SequenceExample],
    *,
    beam_size: int,
    batch_size: int,
    warmup: int,
    iterations: int,
    long_tail_mode: str,
) -> dict[str, object]:
    batches = batched_examples(examples, batch_size, warmup + iterations)
    for batch in batches[:warmup]:
        engine.recommend(
            [example.history for example in batch],
            beam_size=beam_size,
            long_tail_mode=long_tail_mode,
        )
    gc.collect()
    if torch.cuda.is_available() and engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
        torch.cuda.reset_peak_memory_stats(engine.device)
    latencies_ms: list[float] = []
    total_histories = 0
    started_all = time.perf_counter()
    with PeakMemorySampler() as memory:
        for batch in batches[warmup:]:
            started = time.perf_counter()
            engine.recommend(
                [example.history for example in batch],
                beam_size=beam_size,
                long_tail_mode=long_tail_mode,
            )
            if torch.cuda.is_available() and engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            total_histories += len(batch)
    elapsed = time.perf_counter() - started_all
    gpu_peak = (
        int(torch.cuda.max_memory_allocated(engine.device))
        if torch.cuda.is_available() and engine.device.type == "cuda"
        else 0
    )
    return {
        "beam_size": beam_size,
        "batch_size": batch_size,
        "long_tail_mode": long_tail_mode,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "histories": total_histories,
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
            "mean": statistics.fmean(latencies_ms),
            "per_history_mean": statistics.fmean(latencies_ms) / batch_size,
        },
        "qps": total_histories / elapsed,
        "wall_seconds": elapsed,
        "memory": {
            "rss_baseline_mib": memory.baseline / 1024**2,
            "rss_peak_mib": memory.peak / 1024**2,
            "rss_peak_delta_mib": (memory.peak - memory.baseline) / 1024**2,
            "accelerator_peak_allocated_mib": gpu_peak / 1024**2,
        },
    }


def markdown_report(report: dict[str, object]) -> str:
    environment = report["environment"]
    lines = [
        "# 在线推理性能报告",
        "",
        "> 该报告由真实冻结模型、真实商品目录和真实测试历史生成；不包含模拟延迟。",
        "",
        "## 固定环境",
        "",
        f"- 平台：`{environment['platform']}`",
        f"- 处理器：`{environment['processor']}`",
        f"- 逻辑核心：`{environment['logical_cpu_count']}`",
        f"- 内存：`{environment['memory_total_gib']:.2f} GiB`",
        f"- Python：`{environment['python']}`",
        f"- PyTorch：`{environment['torch']}`",
        f"- 推理设备：`{environment['device']}`",
        f"- PyTorch 线程：`{environment['torch_threads']}`",
        "",
        "## 结果",
        "",
        "| Beam | Batch | 模式 | P50 ms | P95 ms | P99 ms | QPS | 峰值 RSS MiB | 增量 MiB |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        latency = row["latency_ms"]
        memory = row["memory"]
        lines.append(
            f"| {row['beam_size']} | {row['batch_size']} | {row['long_tail_mode']} | "
            f"{latency['p50']:.2f} | {latency['p95']:.2f} | {latency['p99']:.2f} | "
            f"{row['qps']:.2f} | {memory['rss_peak_mib']:.2f} | "
            f"{memory['rss_peak_delta_mib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "延迟按一次 API 等价调用统计；批量场景的 QPS 按处理的用户历史数计算。"
            "Beam-1 只返回 1 个候选，Beam-10/20 返回最多 10 个商品。",
            "",
            "## 可复现性",
            "",
            f"- 测试样本：`{report['sample']['count']}` 条，按测试文件前缀固定选取；",
            f"- 模型 SHA-256：`{report['provenance']['model_sha256']}`；",
            f"- Catalog SHA-256：`{report['provenance']['catalog_sha256']}`；",
            f"- 数据审计 SHA-256：`{report['provenance']['data_audit_sha256']}`；",
            f"- 预热次数：`{report['protocol']['warmup']}`；正式重复：`{report['protocol']['iterations']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.sample_count <= 0:
        raise ValueError("warmup must be non-negative; iterations and sample-count positive")
    beam_sizes = args.beam_sizes or [1, 10, 20]
    batch_sizes = args.batch_sizes or [1, 8]
    settings = ServingSettings.from_env()
    engine = RecommendationEngine.load(settings)
    examples = load_sequence_examples_jsonl(
        engine.config["data"]["test_path"], limit=args.sample_count
    )
    results = []
    for batch_size in batch_sizes:
        if batch_size > settings.max_batch_size:
            raise ValueError(f"batch size exceeds service limit: {batch_size}")
        for beam_size in beam_sizes:
            print(
                f"benchmark beam={beam_size} batch={batch_size} "
                f"mode={args.long_tail_mode}",
                flush=True,
            )
            results.append(
                benchmark_scenario(
                    engine,
                    examples,
                    beam_size=beam_size,
                    batch_size=batch_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    long_tail_mode=args.long_tail_mode,
                )
            )
    audit_path = Path(engine.config["data"]["data_audit_path"])
    report = {
        "schema_version": 1,
        "synthetic_data_used": False,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or "not_reported",
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total_gib": psutil.virtual_memory().total / 1024**3,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(engine.device),
            "torch_threads": torch.get_num_threads(),
        },
        "protocol": {
            "beam_sizes": beam_sizes,
            "batch_sizes": batch_sizes,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "long_tail_mode": args.long_tail_mode,
            "latency_scope": "model, constrained beam, optional reranking, response assembly",
        },
        "sample": {
            "dataset": "Amazon Reviews 2023 / Musical Instruments",
            "split": "test",
            "selection": "deterministic prefix",
            "count": len(examples),
        },
        "provenance": {
            "model_sha256": sha256(settings.artifact_dir / "generator.pt"),
            "catalog_sha256": sha256(settings.artifact_dir / "catalog.json"),
            "data_audit_sha256": sha256(audit_path),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
