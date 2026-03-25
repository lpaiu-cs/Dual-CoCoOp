import argparse
import json
import math
import re
from pathlib import Path


ACC_PATTERN = re.compile(r"\* accuracy:\s+([0-9.]+)%")


def read_accuracy(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = ACC_PATTERN.findall(text)
    if not matches:
        raise ValueError(f"Could not find accuracy in {log_path}")
    return float(matches[-1])


def harmonic_mean(base: float, new: float) -> float:
    if base + new == 0:
        return 0.0
    return 2 * base * new / (base + new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        train_log = (
            args.output_root
            / "train_base"
            / args.dataset
            / "shots_16"
            / args.trainer
            / args.config
            / f"seed{seed}"
            / "log.txt"
        )
        test_log = (
            args.output_root
            / "test_new"
            / args.dataset
            / "shots_16"
            / args.trainer
            / args.config
            / f"seed{seed}"
            / "log.txt"
        )
        base = read_accuracy(train_log)
        new = read_accuracy(test_log)
        rows.append({"seed": seed, "base": base, "new": new, "harmonic": harmonic_mean(base, new)})

    base_mean = sum(row["base"] for row in rows) / len(rows)
    new_mean = sum(row["new"] for row in rows) / len(rows)
    harm_mean = sum(row["harmonic"] for row in rows) / len(rows)

    result = {
        "dataset": args.dataset,
        "trainer": args.trainer,
        "config": args.config,
        "rows": rows,
        "mean": {
            "base": base_mean,
            "new": new_mean,
            "harmonic": harm_mean,
            "harmonic_from_means": harmonic_mean(base_mean, new_mean),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
