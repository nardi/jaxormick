"""Compares benchmark results with a baseline, relative to the plain functions.

Usage: `python compare.py results.json baseline.json`.

Each time is divided by the time of the plain function in the same run, for the same
function and kind of measurement. These ratios are compared, so the comparison holds
across machines. The script prints a table and exits with status 1 when a ratio grew by
more than the allowed factor.
"""

import json
import sys
from pathlib import Path

ALLOWED_GROWTH = 1.5
"""Factor by which a ratio may grow before it counts as a regression."""


def ratios(path: Path) -> dict[tuple[str, str, str], float]:
    """Returns the time relative to the plain function for every benchmark in a file."""
    benchmarks = json.loads(path.read_text())["benchmarks"]
    times = {
        (info["kind"], info["function"], info["transform"]): entry["stats"]["median"]
        for entry in benchmarks
        for info in [entry["extra_info"]]
    }
    return {
        key: time / times[(key[0], key[1], "plain")]
        for key, time in times.items()
        if key[2] != "plain"
    }


def main(results: Path, baseline: Path) -> int:
    """Prints the comparison and returns the exit status."""
    current, previous = ratios(results), ratios(baseline)
    regressions = 0
    print(
        f"{'kind':<8} {'function':<18} {'transform':<15} {'ratio':>8} {'baseline':>9}"
    )
    for key in sorted(current):
        ratio = current[key]
        reference = previous.get(key)
        mark = ""
        if reference is not None and ratio > ALLOWED_GROWTH * reference:
            mark = "  slower"
            regressions += 1
        shown = f"{reference:9.1f}" if reference is not None else f"{'-':>9}"
        print(f"{key[0]:<8} {key[1]:<18} {key[2]:<15} {ratio:8.1f} {shown}{mark}")
    print(f"\n{regressions} regressions beyond {ALLOWED_GROWTH}x.")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
