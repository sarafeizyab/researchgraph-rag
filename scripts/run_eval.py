from __future__ import annotations

import json
from pathlib import Path

from evaluation.ablations import run_ablation_comparison, to_markdown_table


def main() -> None:
    input_path = Path("artifacts/ablation_records.json")
    if not input_path.exists():
        raise FileNotFoundError("Expected artifacts/ablation_records.json")

    records = json.loads(input_path.read_text())
    runs = run_ablation_comparison(records)

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_out = output_dir / "ablation_summary.json"
    md_out = output_dir / "ablation_summary.md"

    serializable = [
        {
            "variant": run.name,
            "metrics": [{"name": r.metric_name, "value": r.value, "details": r.details} for r in run.results],
        }
        for run in runs
    ]

    json_out.write_text(json.dumps(serializable, indent=2))
    md_out.write_text(to_markdown_table(runs))

    print(f"Wrote {json_out} and {md_out}")


if __name__ == "__main__":
    main()
