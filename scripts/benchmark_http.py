from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Sequence

import httpx
import psutil

from regenrec.config import load_config
from regenrec.data import SequenceExample, load_sequence_examples_jsonl
from regenrec.serving import ServingSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end HTTP concurrency benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--concurrency", action="append", type=int, dest="concurrencies")
    parser.add_argument("--batch-size", action="append", type=int, dest="batch_sizes")
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument(
        "--long-tail-mode", choices=("off", "mild", "aggressive"), default="off"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/http_benchmark/report.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("docs/HTTP_PERFORMANCE_REPORT.md")
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
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def event_payload(example: SequenceExample) -> list[dict[str, object]]:
    return [
        {
            "item_id": event.item_id,
            "behavior": event.behavior,
            "behavior_intensity": event.behavior_intensity,
            "time_gap_bucket": event.time_gap_bucket,
        }
        for event in example.history
    ]


def request_payload(
    examples: Sequence[SequenceExample],
    *,
    worker_index: int,
    batch_size: int,
    beam_size: int,
    long_tail_mode: str,
) -> tuple[str, dict[str, object]]:
    if batch_size == 1:
        example = examples[worker_index % len(examples)]
        return "/v1/recommend", {
            "history": event_payload(example),
            "beam_size": beam_size,
            "long_tail_mode": long_tail_mode,
        }
    requests = []
    for offset in range(batch_size):
        example = examples[(worker_index * batch_size + offset) % len(examples)]
        requests.append(
            {
                "request_id": f"bench-{worker_index}-{offset}",
                "history": event_payload(example),
            }
        )
    return "/v1/recommend/batch", {
        "requests": requests,
        "beam_size": beam_size,
        "long_tail_mode": long_tail_mode,
    }


async def benchmark_scenario(
    client: httpx.AsyncClient,
    examples: Sequence[SequenceExample],
    *,
    concurrency: int,
    batch_size: int,
    beam_size: int,
    long_tail_mode: str,
    warmup: int,
    duration_seconds: float,
) -> dict[str, object]:
    endpoint, warmup_payload = request_payload(
        examples,
        worker_index=0,
        batch_size=batch_size,
        beam_size=beam_size,
        long_tail_mode=long_tail_mode,
    )
    for _ in range(warmup):
        response = await client.post(endpoint, json=warmup_payload)
        response.raise_for_status()
    latencies_ms: list[float] = []
    statuses: dict[int, int] = {}
    timeouts = 0
    transport_errors = 0
    validation_errors = 0
    stop_at = time.perf_counter() + duration_seconds

    async def worker(worker_index: int) -> None:
        nonlocal timeouts, transport_errors, validation_errors
        request_number = worker_index
        while time.perf_counter() < stop_at:
            endpoint, payload = request_payload(
                examples,
                worker_index=request_number,
                batch_size=batch_size,
                beam_size=beam_size,
                long_tail_mode=long_tail_mode,
            )
            started = time.perf_counter()
            try:
                response = await client.post(endpoint, json=payload)
                latency = (time.perf_counter() - started) * 1000.0
                statuses[response.status_code] = statuses.get(response.status_code, 0) + 1
                if response.status_code == 200:
                    body = response.json()
                    rows = [body] if batch_size == 1 else body.get("results", [])
                    expected_count = min(10, beam_size)
                    if len(rows) != batch_size or any(
                        int(row.get("returned_count", -1)) != expected_count for row in rows
                    ):
                        validation_errors += 1
                    else:
                        latencies_ms.append(latency)
            except httpx.TimeoutException:
                timeouts += 1
            except httpx.HTTPError:
                transport_errors += 1
            request_number += concurrency

    started = time.perf_counter()
    await asyncio.gather(*(worker(index) for index in range(concurrency)))
    wall_seconds = time.perf_counter() - started
    successful_requests = len(latencies_ms)
    successful_histories = successful_requests * batch_size
    total_requests = sum(statuses.values()) + timeouts + transport_errors
    return {
        "concurrency": concurrency,
        "batch_size": batch_size,
        "beam_size": beam_size,
        "long_tail_mode": long_tail_mode,
        "duration_target_seconds": duration_seconds,
        "wall_seconds": wall_seconds,
        "requests": total_requests,
        "successful_requests": successful_requests,
        "successful_histories": successful_histories,
        "status_counts": {str(key): value for key, value in sorted(statuses.items())},
        "timeouts": timeouts,
        "transport_errors": transport_errors,
        "response_validation_errors": validation_errors,
        "error_rate": (
            (total_requests - successful_requests) / total_requests
            if total_requests
            else 0.0
        ),
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
            "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        },
        "request_qps": successful_requests / wall_seconds,
        "history_qps": successful_histories / wall_seconds,
    }


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# HTTP 端到端并发压测",
        "",
        "> 延迟覆盖 HTTP 收发、JSON 解析、模型推理、Trie Beam、结果组装与 JSON 序列化。",
        "",
        "## 固定协议",
        "",
        f"- 平台：`{report['environment']['platform']}`；",
        f"- CPU：`{report['environment']['logical_cpu_count']}` 逻辑核；",
        f"- Beam：`{report['protocol']['beam_size']}`；",
        f"- 每个组合持续：`{report['protocol']['duration_seconds']:.1f}` 秒；",
        f"- 长尾模式：`{report['protocol']['long_tail_mode']}`；",
        "- 样本：Amazon Reviews 2023 / Musical Instruments 真实 Test 历史；",
        "- 服务进程：单 worker，模型推理串行保护。",
        "",
        "## 结果",
        "",
        "| 并发 | Batch | P50 ms | P95 ms | P99 ms | 请求 QPS | 历史 QPS | 错误率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        latency = row["latency_ms"]
        lines.append(
            f"| {row['concurrency']} | {row['batch_size']} | {latency['p50']:.2f} | "
            f"{latency['p95']:.2f} | {latency['p99']:.2f} | "
            f"{row['request_qps']:.2f} | {row['history_qps']:.2f} | "
            f"{row['error_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "并发升高时，单 worker 的模型锁会把请求排队，因此尾延迟会增加。Batch 推理以一次"
            "请求处理多条历史，提高用户历史吞吐，但单请求响应体也更大。所有成功响应均校验"
            "返回条数；超时、非 200 和响应契约错误单独计数。",
            "",
            "模型 SHA-256："
            f"`{report['provenance']['model_sha256']}`。",
            "",
        ]
    )
    return "\n".join(lines)


async def async_main(args: argparse.Namespace) -> dict[str, object]:
    settings = ServingSettings.from_env()
    settings.validate(require_long_tail=args.long_tail_mode != "off")
    config = load_config(settings.config_path)
    examples = load_sequence_examples_jsonl(
        config["data"]["test_path"], limit=args.sample_count
    )
    limits = httpx.Limits(
        max_connections=max(args.concurrencies),
        max_keepalive_connections=max(args.concurrencies),
    )
    timeout = httpx.Timeout(args.timeout_seconds)
    results = []
    async with httpx.AsyncClient(
        base_url=args.base_url,
        limits=limits,
        timeout=timeout,
        trust_env=False,
    ) as client:
        ready = await client.get("/health/ready")
        ready.raise_for_status()
        for batch_size in args.batch_sizes:
            for concurrency in args.concurrencies:
                print(
                    f"http benchmark concurrency={concurrency} batch={batch_size}",
                    flush=True,
                )
                results.append(
                    await benchmark_scenario(
                        client,
                        examples,
                        concurrency=concurrency,
                        batch_size=batch_size,
                        beam_size=args.beam_size,
                        long_tail_mode=args.long_tail_mode,
                        warmup=args.warmup,
                        duration_seconds=args.duration_seconds,
                    )
                )
    return {
        "schema_version": 1,
        "synthetic_data_used": False,
        "environment": {
            "platform": platform.platform(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total_gib": psutil.virtual_memory().total / 1024**3,
            "python": platform.python_version(),
            "httpx": httpx.__version__,
        },
        "protocol": {
            "base_url": args.base_url,
            "concurrencies": args.concurrencies,
            "batch_sizes": args.batch_sizes,
            "beam_size": args.beam_size,
            "warmup": args.warmup,
            "duration_seconds": args.duration_seconds,
            "timeout_seconds": args.timeout_seconds,
            "long_tail_mode": args.long_tail_mode,
            "server_workers": 1,
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
        },
        "results": results,
    }


def main() -> None:
    args = parse_args()
    args.concurrencies = args.concurrencies or [1, 4, 8, 16]
    args.batch_sizes = args.batch_sizes or [1, 8]
    if any(value <= 0 for value in (*args.concurrencies, *args.batch_sizes)):
        raise ValueError("concurrency and batch sizes must be positive")
    if args.duration_seconds <= 0 or args.timeout_seconds <= 0 or args.warmup < 0:
        raise ValueError("duration/timeout must be positive and warmup non-negative")
    report = asyncio.run(async_main(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
