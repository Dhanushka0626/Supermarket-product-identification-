"""Run the whole pipeline (or selected steps) with the same settings.      Owner: shared

    python scripts/run_all.py --config configs/config.yaml
    python scripts/run_all.py --steps prepare train validate test outcomes export --set run_cv=false
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = dict(prepare="prepare_data.py", train="train.py", validate="validate.py", test="evaluate_test.py",
             cv="cross_validate.py", outcomes="build_outcomes.py", export="export_model.py")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE.parent / "configs" / "config.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--steps", nargs="*", default=list(STEPS), choices=list(STEPS))
    args = ap.parse_args()
    for step in args.steps:
        print(f"\n{'=' * 70}\n  STEP: {step}\n{'=' * 70}", flush=True)
        cmd = [sys.executable, str(HERE / STEPS[step]), "--config", args.config]
        if args.set:
            cmd += ["--set", *args.set]
        subprocess.run(cmd, check=True)
    print("\nAll requested steps finished.")
