"""
Summarize A/A benchmark runs produced by the aa-noise workflow.

Each run compares a commit against an empty commit on top of it, so every
difference asv reports is noise. Expects a directory containing one
subdirectory per run, each holding meta.json and an asv results directory.
Writes a Markdown summary to stdout and the per-run log ratios to a CSV file.

Usage: python aa_summary.py RUNS_DIR CSV_OUTPUT
"""

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from asv.commands.compare import _is_result_better, unroll_result
from asv.results import iter_results_for_machine_and_hash

FACTORS = [1.1, 1.05, 1.02]

# Two-sided z value for a 1% false-positive rate, used to turn the noise
# level into the smallest change that could be trusted.
Z_99 = 2.576


def load_values(results_dir, machine, commit):
    values = {}
    for result in iter_results_for_machine_and_hash(results_dir, machine, commit):
        for key in result.get_all_result_keys():
            params = result.get_result_params(key)
            for name, value, stats, samples in unroll_result(
                key,
                params,
                result.get_result_value(key, params),
                result.get_result_stats(key, params),
                result.get_result_samples(key, params),
            ):
                values[name] = (value, (stats, samples))
    return values


def asv_flags(base, head, factor):
    """Whether asv compare would report a change, using its own decision rule."""
    (time_1, ss_1), (time_2, ss_2) = base, head
    return _is_result_better(time_2, time_1, ss_2, ss_1, factor) or _is_result_better(
        time_1, time_2, ss_1, ss_2, factor
    )


def collect(runs_dir):
    """Return {platform: [run]}, where run maps benchmark name to (base, head)."""
    runs = defaultdict(list)
    for meta_path in sorted(Path(runs_dir).glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        results_dir = str(meta_path.parent / "results")
        base = load_values(results_dir, meta["platform"], meta["base"])
        head = load_values(results_dir, meta["platform"], meta["head"])
        pairs = {}
        for name in base.keys() & head.keys():
            if all(isinstance(v[0], float) and v[0] > 0 for v in (base[name], head[name])):
                pairs[name] = (base[name], head[name])
        runs[meta["platform"]].append({"meta": meta, "pairs": pairs})
    return runs


def format_time(seconds):
    for unit, scale in (("s", 1), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)):
        if seconds >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds:.3g} s"


def main(runs_dir, csv_path):
    runs = collect(runs_dir)
    platforms = sorted(runs)
    names = sorted({name for platform in platforms for run in runs[platform] for name in run["pairs"]})

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["platform", "repeat", "started", "benchmark", "base", "head", "log_ratio"])
        for platform in platforms:
            for run in runs[platform]:
                for name, ((t1, _), (t2, _)) in sorted(run["pairs"].items()):
                    writer.writerow(
                        [platform, run["meta"]["repeat"], run["meta"]["started"], name, t1, t2, math.log(t2 / t1)]
                    )

    # rms[platform][name] is the RMS of log(head/base) across runs. The true
    # ratio is 1, so RMS about zero (rather than the standard deviation)
    # also captures any systematic bias between the first and second commit.
    rms = defaultdict(dict)
    median_time = defaultdict(dict)
    flagged = defaultdict(dict)
    for platform in platforms:
        for name in names:
            pairs = [run["pairs"][name] for run in runs[platform] if name in run["pairs"]]
            if not pairs:
                continue
            log_ratios = [math.log(head[0] / base[0]) for base, head in pairs]
            rms[platform][name] = math.sqrt(statistics.fmean(r * r for r in log_ratios))
            median_time[platform][name] = statistics.median(base[0] for base, _ in pairs)
            flagged[platform][name] = (sum(asv_flags(base, head, FACTORS[0]) for base, head in pairs), len(pairs))

    print("# A/A benchmark noise\n")
    print("Each run compares a commit with an empty commit on top of it, so any reported change is a false positive.\n")

    print("## Overall\n")
    print("| | " + " | ".join(platforms) + " |")
    print("|---|" + "---|" * len(platforms))
    print("| Runs | " + " | ".join(str(len(runs[p])) for p in platforms) + " |")
    print(
        "| Median noise (RMS of ratio) | "
        + " | ".join(f"{statistics.median(rms[p].values()) * 100:.2f}%" for p in platforms)
        + " |"
    )
    print(
        "| Median change trustable at 1% FPR | "
        + " | ".join(f"{(math.exp(Z_99 * statistics.median(rms[p].values())) - 1) * 100:.1f}%" for p in platforms)
        + " |"
    )
    for factor in FACTORS:
        cells = []
        for p in platforms:
            per_bench = sum(asv_flags(b, h, factor) for run in runs[p] for b, h in run["pairs"].values())
            total = sum(len(run["pairs"]) for run in runs[p])
            any_flag = sum(any(asv_flags(b, h, factor) for b, h in run["pairs"].values()) for run in runs[p])
            cells.append(f"{per_bench / total * 100:.1f}% / {any_flag}/{len(runs[p])}")
        print(f"| False positives at factor {factor} (per benchmark / runs with any) | " + " | ".join(cells) + " |")

    print("\n## Per benchmark\n")
    print(f"Noise is the RMS of the ratio across runs; flagged counts use asv's rule at factor {FACTORS[0]}.\n")
    print("| Benchmark | " + " | ".join(f"{p} time | {p} noise | {p} flagged" for p in platforms) + " |")
    print("|---|" + "---|---|---|" * len(platforms))
    for name in names:
        cells = []
        for p in platforms:
            if name in rms[p]:
                n_flagged, n_runs = flagged[p][name]
                cells += [format_time(median_time[p][name]), f"{rms[p][name] * 100:.2f}%", f"{n_flagged}/{n_runs}"]
            else:
                cells += ["", "", ""]
        print(f"| `{name}` | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
