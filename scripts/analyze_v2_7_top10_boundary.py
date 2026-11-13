from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import iter_sequence_examples_jsonl
from scripts.train_v2_5_bounded_residual import (
    catalog_groups,
    evaluate,
    load_cache,
    make_features,
    score_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only V2.7 candidate-oracle and Top-10 boundary diagnosis"
    )
    parser.add_argument("--config", default="configs/v2_final_default.yaml")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--v2-6-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="scale_1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--minimum-tail-oracle-gain", type=float, default=0.01)
    parser.add_argument("--minimum-retrieval-only-tail-rows", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def rank_positions(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return one-based stable ranks and -1 outside the candidate mask."""
    if scores.shape != mask.shape or scores.ndim != 2:
        raise ValueError("scores and mask must be equal two-dimensional tensors")
    ordered = torch.argsort(
        scores.masked_fill(~mask, float("-inf")),
        dim=1,
        descending=True,
        stable=True,
    )
    ranks = torch.empty_like(ordered)
    values = torch.arange(1, scores.size(1) + 1, device=scores.device)
    ranks.scatter_(1, ordered, values.unsqueeze(0).expand_as(ordered))
    return ranks.masked_fill(~mask, -1)


def topk_margin(
    scores: torch.Tensor,
    mask: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Target score minus the current kth score; NaN if target is unavailable."""
    if k <= 0 or k > scores.size(1):
        raise ValueError("k must be within the candidate width")
    contained = target_positions >= 0
    safe = target_positions.clamp_min(0).unsqueeze(1)
    target = scores.gather(1, safe).squeeze(1)
    kth = scores.masked_fill(~mask, float("-inf")).topk(k, dim=1).values[:, -1]
    return (target - kth).masked_fill(~contained, float("nan"))


def changed_topk_rows(
    left_scores: torch.Tensor,
    right_scores: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = left_scores.masked_fill(~mask, float("-inf")).topk(k, dim=1).indices
    right = right_scores.masked_fill(~mask, float("-inf")).topk(k, dim=1).indices
    left = left.sort(dim=1).values
    right = right.sort(dim=1).values
    rows = (left != right).any(dim=1)
    swaps = torch.zeros(len(rows), dtype=torch.long)
    for index in torch.nonzero(rows, as_tuple=False).flatten().tolist():
        swaps[index] = len(set(left[index].tolist()) - set(right[index].tolist()))
    return rows, swaps


def quantiles(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().float()
    values = values[torch.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
    }


def summarize_boundary(
    selected: torch.Tensor,
    *,
    baseline_margin: torch.Tensor,
    candidate_margin: torch.Tensor,
    required_lift: torch.Tensor,
    learned_pair_lift: torch.Tensor,
    maximum_pair_lift: torch.Tensor,
    candidate_rank: torch.Tensor,
    semantic_tail_rank: torch.Tensor,
    top_k: int,
) -> dict[str, object]:
    count = int(selected.sum())
    if not count:
        return {"examples": 0}
    return {
        "examples": count,
        "baseline_target_minus_topk": quantiles(baseline_margin[selected]),
        "v2_6_target_minus_topk": quantiles(candidate_margin[selected]),
        "required_pair_lift": quantiles(required_lift[selected]),
        "learned_pair_lift": quantiles(learned_pair_lift[selected]),
        "theoretical_maximum_pair_lift": quantiles(maximum_pair_lift[selected]),
        "theoretically_reachable": int(
            (maximum_pair_lift[selected] >= required_lift[selected]).sum()
        ),
        "converted_to_top10": int((candidate_rank[selected] <= top_k).sum()),
        "blair_tail_rank": quantiles(semantic_tail_rank[selected].float()),
    }


def make_bins(
    values: torch.Tensor, definitions: tuple[tuple[str, int, int | None], ...]
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, lower, upper in definitions:
        selected = values >= lower
        if upper is not None:
            selected &= values <= upper
        output[name] = selected
    return output


def render_report(report: dict[str, object]) -> str:
    oracle = report["candidate_opportunity"]
    boundary = report["top10_boundary"]
    change = report["ranking_change"]
    decision = report["decision"]
    return "\n".join(
        (
            "# V2.7 Validation-only 候选与 Top-10 边界诊断",
            "",
            f"- 状态：**{report['status']}**；Test evaluated：`false`。",
            f"- Validation 样本：{report['validation_examples']:,}。",
            f"- Observed-Tail 基线 Recall@10：{oracle['baseline_tail_recall_at_10']:.6f}。",
            f"- Observed-Tail union oracle：{oracle['union_tail_oracle_recall']:.6f}；绝对空间：{oracle['union_tail_oracle_gain']:.6f}。",
            f"- BLaIR-only 尾部目标：{oracle['retrieval_only_tail_rows']:,}；V2.6 转入 Top-10：{boundary['retrieval_only_tail']['converted_to_top10']:,}。",
            f"- V2.6 scale-1 改变 Top-10 集合的行：{change['rows_with_any_top10_change']:,}；尾部新增/损失命中：{change['tail_wins']}/{change['tail_losses']}。",
            "",
            "## 决策",
            "",
            f"工程可行性门禁：**{'PASS' if decision['passed'] else 'FAIL'}**。这只说明候选中存在可学习空间，不构成模型增益证据。",
            f"下一步：{decision['next_step']}。",
            "",
            "## 科学边界",
            "",
            "本诊断只读取冻结 Validation 候选缓存；Test 未读取，不能据此报告泛化增益。",
            "",
        )
    )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.minimum_retrieval_only_tail_rows <= 0 or args.torch_threads <= 0:
        raise ValueError("top-k, row threshold and torch threads must be positive")
    if args.minimum_tail_oracle_gain <= 0:
        raise ValueError("minimum tail oracle gain must be positive")
    torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "state.json"
    write_json(
        state_path,
        {"status": "running", "phase": "validation_boundary_diagnosis", "test_evaluated": False},
    )
    try:
        config = load_config(Path(args.config))
        catalog = SemanticIdCatalog.load(args.artifact_dir / "catalog.json")
        payload = load_cache(args.validation_cache, "validation")
        cache_report = json.loads(args.cache_report.read_text(encoding="utf-8"))
        v2_6 = json.loads(args.v2_6_report.read_text(encoding="utf-8"))
        if cache_report.get("split") != "validation" or cache_report.get("test_evaluated") is not False:
            raise ValueError("candidate source must be a Test-isolated Validation run")
        if v2_6.get("test_evaluated") is not False or v2_6.get("status") != "completed":
            raise ValueError("V2.6 report must be completed and Test-isolated")
        if args.variant not in v2_6["variants"]:
            raise ValueError(f"missing V2.6 variant: {args.variant}")
        cache_provenance = payload["metadata"].get("provenance", {})
        report_provenance = cache_report.get("provenance", {})
        for name in ("model_sha256", "catalog_sha256", "embedding_sha256"):
            if cache_provenance.get(name) != report_provenance.get(name):
                raise ValueError(f"candidate cache/report provenance mismatch for {name}")
        if sha256_file(args.validation_cache) != v2_6["provenance"]["validation_cache_sha256"]:
            raise ValueError("V2.6 did not use the supplied frozen Validation cache")

        groups_lookup = catalog_groups(config, catalog)
        candidate, row, base = make_features(payload)
        mask = payload["candidate_mask"].bool()
        positions = payload["target_positions"].long()
        safe = positions.clamp_min(0).unsqueeze(1)
        contained = positions >= 0
        targets = payload["target_catalog_indices"].long()
        target_groups = groups_lookup[targets]
        finite_beam = torch.isfinite(payload["beam_scores"])
        target_in_beam = contained & finite_beam.gather(1, safe).squeeze(1)
        retrieval_only = contained & ~finite_beam.gather(1, safe).squeeze(1)
        retrieval_only_tail = retrieval_only & (target_groups == 2)

        model = v2_6["model"]
        variant = v2_6["variants"][args.variant]
        scale = float(variant["deployment_scale"])
        bound = float(v2_6["hyperparameters"]["bound"])
        scores, gate, residual = score_candidates(
            candidate,
            row,
            base,
            torch.tensor(model["residual_weights"]),
            torch.tensor(model["gate_weights"]),
            torch.tensor(model["gate_bias"]),
            bound=bound * scale,
        )
        baseline_metrics, baseline_hits = evaluate(
            base, payload, groups_lookup, catalog_size=len(catalog.item_ids)
        )
        candidate_metrics, candidate_hits = evaluate(
            scores, payload, groups_lookup, catalog_size=len(catalog.item_ids)
        )
        expected = variant["metrics"]["ranking"]["recall_at_10"]
        if abs(candidate_metrics["ranking"]["recall_at_10"] - expected) > 1e-9:
            raise ValueError("recomputed V2.6 metric does not match the frozen report")

        candidate_rank = rank_positions(scores, mask)
        semantic_tail_mask = mask & (payload["groups"].long() == 2)
        semantic_tail_rank_matrix = rank_positions(
            payload["semantic_scores"].float(), semantic_tail_mask
        )
        target_candidate_rank = candidate_rank.gather(1, safe).squeeze(1).masked_fill(~contained, -1)
        semantic_tail_rank = semantic_tail_rank_matrix.gather(1, safe).squeeze(1).masked_fill(~contained, -1)

        baseline_margin = topk_margin(base, mask, positions, k=args.top_k)
        candidate_margin = topk_margin(scores, mask, positions, k=args.top_k)
        cutoff_positions = base.masked_fill(~mask, float("-inf")).topk(args.top_k, dim=1).indices[:, -1]
        target_base = base.gather(1, safe).squeeze(1)
        cutoff_base = base.gather(1, cutoff_positions.unsqueeze(1)).squeeze(1)
        required_lift = (cutoff_base - target_base).clamp_min(0.0)
        target_residual = residual.gather(1, safe).squeeze(1)
        cutoff_residual = residual.gather(1, cutoff_positions.unsqueeze(1)).squeeze(1)
        learned_pair_lift = target_residual - cutoff_residual
        maximum_pair_lift = 2.0 * bound * scale * gate

        frequencies: Counter[str] = Counter()
        for example in iter_sequence_examples_jsonl(config["data"]["train_path"]):
            frequencies[example.target_item_id] += 1
        frequency_lookup = torch.tensor(
            [frequencies.get(item, 0) for item in catalog.item_ids], dtype=torch.long
        )
        target_frequencies = frequency_lookup[targets]
        history_lengths = payload["history_lengths"].long()
        blair_bins = make_bins(
            semantic_tail_rank,
            (("rank_1_5", 1, 5), ("rank_6_10", 6, 10), ("rank_11_20", 11, 20), ("rank_21_50", 21, 50)),
        )
        history_bins = make_bins(
            history_lengths,
            (("history_1_10", 1, 10), ("history_11_20", 11, 20), ("history_21_50", 21, 50)),
        )
        frequency_bins = make_bins(
            target_frequencies,
            (("frequency_1", 1, 1), ("frequency_2", 2, 2), ("frequency_3_5", 3, 5), ("frequency_6_10", 6, 10), ("frequency_11_plus", 11, None)),
        )
        summary_kwargs = {
            "baseline_margin": baseline_margin,
            "candidate_margin": candidate_margin,
            "required_lift": required_lift,
            "learned_pair_lift": learned_pair_lift,
            "maximum_pair_lift": maximum_pair_lift,
            "candidate_rank": target_candidate_rank,
            "semantic_tail_rank": semantic_tail_rank,
            "top_k": args.top_k,
        }
        breakdown = {
            "blair_rank": {
                name: summarize_boundary(retrieval_only_tail & selected, **summary_kwargs)
                for name, selected in blair_bins.items()
            },
            "history_length": {
                name: summarize_boundary(retrieval_only_tail & selected, **summary_kwargs)
                for name, selected in history_bins.items()
            },
            "target_train_frequency": {
                name: summarize_boundary(retrieval_only_tail & selected, **summary_kwargs)
                for name, selected in frequency_bins.items()
            },
        }

        changed_rows, swaps = changed_topk_rows(base, scores, mask, k=args.top_k)
        tail = target_groups == 2
        oracle = cache_report["candidate_oracle"]
        oracle_union_hits = oracle["union"]["overall"]["hits"]
        if int(contained.sum()) != oracle_union_hits:
            raise ValueError("candidate cache no longer matches its union oracle")
        baseline_tail_recall = baseline_metrics["target_groups"]["observed_tail"]["recall_at_10"]
        union_tail_oracle = oracle["union"]["target_groups"]["observed_tail"]["recall"]
        union_tail_gain = union_tail_oracle - baseline_tail_recall
        decision_checks = {
            "union_tail_oracle_gain_ge_threshold": union_tail_gain >= args.minimum_tail_oracle_gain,
            "retrieval_only_tail_rows_ge_threshold": int(retrieval_only_tail.sum()) >= args.minimum_retrieval_only_tail_rows,
            "validation_only_and_test_untouched": True,
        }
        passed = all(decision_checks.values())
        report = {
            "status": "completed",
            "protocol": "V2.7 candidate opportunity and Top-10 boundary diagnosis",
            "dataset": cache_report["dataset"],
            "category": cache_report["category"],
            "split": "validation",
            "test_evaluated": False,
            "synthetic_data_used": False,
            "seed": int(payload["metadata"]["seed"]),
            "validation_examples": len(targets),
            "candidate_opportunity": {
                "baseline_tail_recall_at_10": baseline_tail_recall,
                "beam20_tail_oracle_recall": oracle["generator"]["target_groups"]["observed_tail"]["recall"],
                "blair50_tail_oracle_recall": oracle["blair_observed_tail_retrieval"]["target_groups"]["observed_tail"]["recall"],
                "union_tail_oracle_recall": union_tail_oracle,
                "union_tail_oracle_gain": union_tail_gain,
                "union_tail_oracle_hits": oracle["union"]["target_groups"]["observed_tail"]["hits"],
                "retrieval_only_tail_rows": int(retrieval_only_tail.sum()),
                "blair_tail_rank_le_10_rows": int((retrieval_only_tail & (semantic_tail_rank <= 10)).sum()),
                "target_in_beam_rows": int(target_in_beam.sum()),
                "target_in_union_rows": int(contained.sum()),
            },
            "top10_boundary": {
                "retrieval_only_tail": summarize_boundary(retrieval_only_tail, **summary_kwargs),
                "all_observed_tail_available": summarize_boundary(contained & tail, **summary_kwargs),
                "breakdown": breakdown,
                "gate_all": quantiles(gate),
                "gate_retrieval_only_tail": quantiles(gate[retrieval_only_tail]),
                "absolute_target_residual_retrieval_only_tail": quantiles(target_residual[retrieval_only_tail].abs()),
                "absolute_cutoff_residual_retrieval_only_tail": quantiles(cutoff_residual[retrieval_only_tail].abs()),
            },
            "ranking_change": {
                "variant": args.variant,
                "rows_with_any_top10_change": int(changed_rows.sum()),
                "rows_with_any_top10_change_rate": float(changed_rows.float().mean()),
                "top10_swaps": quantiles(swaps[changed_rows].float()),
                "overall_wins": int((candidate_hits & ~baseline_hits).sum()),
                "overall_losses": int((~candidate_hits & baseline_hits).sum()),
                "tail_wins": int((candidate_hits & ~baseline_hits & tail).sum()),
                "tail_losses": int((~candidate_hits & baseline_hits & tail).sum()),
                "baseline_recall_at_10": baseline_metrics["ranking"]["recall_at_10"],
                "candidate_recall_at_10": candidate_metrics["ranking"]["recall_at_10"],
            },
            "decision": {
                "type": "engineering feasibility gate; not an inferential gain claim",
                "thresholds": {
                    "minimum_union_tail_oracle_gain": args.minimum_tail_oracle_gain,
                    "minimum_retrieval_only_tail_rows": args.minimum_retrieval_only_tail_rows,
                },
                "checks": decision_checks,
                "passed": passed,
                "next_step": (
                    "freeze a V2.7 candidate-level listwise joint scorer protocol"
                    if passed
                    else "stop long-tail model development and retain original V2"
                ),
            },
            "provenance": {
                "validation_cache_sha256": sha256_file(args.validation_cache),
                "cache_report_sha256": sha256_file(args.cache_report),
                "v2_6_report_sha256": sha256_file(args.v2_6_report),
                "config_sha256": sha256_file(Path(args.config)),
                "catalog_sha256": sha256_file(args.artifact_dir / "catalog.json"),
            },
        }
        write_json(args.output_dir / "report.json", report)
        (args.output_dir / "REPORT.md").write_text(render_report(report), encoding="utf-8")
        write_json(
            state_path,
            {
                "status": "completed",
                "phase": None,
                "test_evaluated": False,
                "feasibility_gate_passed": passed,
                "report": str((args.output_dir / "report.json").resolve()),
            },
        )
        print(json.dumps({"status": "completed", "feasibility_gate_passed": passed}, indent=2))
    except Exception as error:
        write_json(
            state_path,
            {
                "status": "failed",
                "phase": "validation_boundary_diagnosis",
                "test_evaluated": False,
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
