from __future__ import annotations

from dataclasses import dataclass

from evaluation.metrics import EvalResult, citation_coverage


@dataclass
class AblationRun:
    name: str
    results: list[EvalResult]


def run_ablation_comparison(records: list[dict]) -> list[AblationRun]:
    """Compute simple ablation summaries from run records.

    Expected per-record keys:
    - variant
    - answer
    - used_chunk_ids
    """

    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["variant"], []).append(record)

    output: list[AblationRun] = []
    for variant, runs in grouped.items():
        scores = [citation_coverage(r["answer"], r.get("used_chunk_ids", [])).value for r in runs]
        avg_score = sum(scores) / max(len(scores), 1)
        output.append(AblationRun(name=variant, results=[EvalResult("avg_citation_coverage", avg_score, {"n": len(scores)})]))

    return sorted(output, key=lambda x: x.name)


def to_markdown_table(ablation_runs: list[AblationRun]) -> str:
    lines = ["| Variant | Metric | Value |", "|---|---:|---:|"]
    for run in ablation_runs:
        for result in run.results:
            lines.append(f"| {run.name} | {result.metric_name} | {result.value:.4f} |")
    return "\n".join(lines)
