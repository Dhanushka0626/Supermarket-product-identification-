"""Phase 6a - collect every result into outcomes/ (+ rpc17_outcomes.zip).  Owner: Member 2

    python scripts/build_outcomes.py --config configs/config.yaml
"""
import pandas as pd
from _common import parse_args

from smartcheckout.reporting import figures, outcomes

if __name__ == "__main__":
    args, cfg, paths = parse_args(__doc__)
    counts = paths.results / "class_counts.csv"
    if counts.exists():
        figures.class_counts(pd.read_csv(counts, index_col=0), paths.figures / "class_counts.png")
    outcomes.build(cfg, paths)
    print((paths.outcomes / "outcomes_report.md").read_text(encoding="utf-8"))
