"""Run each released reference and cross-check its result against an independent calculation."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from reference_checks import verify

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.objective_check import objective_matches

CATALOG = ROOT / "data" / "laflor" / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ids", nargs="*", help="Optional LAFLOR IDs; the default verifies all 20."
    )
    args = parser.parse_args()
    rows = json.loads(CATALOG.read_text(encoding="utf-8"))["tasks"]
    requested = set(args.ids)
    unknown = requested - {row["task_id"] for row in rows}
    if unknown:
        parser.error(f"Unknown IDs: {', '.join(sorted(unknown))}")
    finite = infeasible = 0
    with tempfile.TemporaryDirectory() as folder:
        for row in rows:
            tid = row["task_id"]
            if requested and tid not in requested:
                continue
            result = subprocess.run(
                [sys.executable, "-B", str(CATALOG.parent / row["reference"])],
                cwd=folder,
                text=True,
                capture_output=True,
                timeout=300,
            )
            if result.returncode:
                raise RuntimeError(
                    f"{tid} reference failed:\n{result.stdout[-2000:]}\n{result.stderr}"
                )
            observed = json.loads(result.stdout.strip().splitlines()[-1])
            if observed["status"] != row["status"]:
                raise ValueError(f"{tid}: reference status differs from the manifest")
            independent = verify(tid, observed)
            if row["status"] == "infeasible":
                if any(
                    value is not None
                    for value in (
                        row["ground_truth"],
                        row["verified_objective"],
                        observed["objective"],
                        independent,
                    )
                ):
                    raise ValueError(f"{tid}: infeasible models have no objective")
                infeasible += 1
                print(f"{tid:<14} infeasible (geometry and binary feasibility model)")
            else:
                value = observed["objective"]
                if not all(
                    math.isclose(value, estimate, rel_tol=1e-8, abs_tol=1e-8)
                    for estimate in (independent, row["verified_objective"])
                ):
                    raise ValueError(
                        f"{tid}: reference {value}, independent {independent}, catalog {row['verified_objective']}"
                    )
                if not objective_matches(value, row["ground_truth"]):
                    raise ValueError(
                        f"{tid}: objective outside the default absolute evaluation tolerance"
                    )
                finite += 1
                print(f"{tid:<14} {value:.10f}")
    count = finite + infeasible
    noun = "case" if count == 1 else "cases"
    print(f"Verified {count} {noun} ({finite} optimal, {infeasible} infeasible).")


if __name__ == "__main__":
    main()
