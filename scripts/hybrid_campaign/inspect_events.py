#!/usr/bin/env python3
"""Print the detected event onsets for every long run, separating pre-collapse
dynamics from the terminal plunge."""
import glob
import json
import statistics as st

for f in sorted(glob.glob("analysis/stability/long_*_events.json")):
    d = json.load(open(f))
    case = f.split("/")[-1].replace("_events.json", "")
    sec = d.get("secular_contraction") or {}
    unb = d.get("unbounded_oscillation") or {}
    deps = d.get("cohort_departures") or []
    first_dep = min((x["onset_t_over_P"] for x in deps), default=None)
    classes = sorted(set(x["st_class"] for x in deps))
    print(case)
    print(f"   secular onset t/P: {sec.get('onset_t_over_P')}")
    print(f"   unbounded onset t/P: {unb.get('onset_t_over_P')}")
    print(f"   departures: n={len(deps)} first_onset={first_dep} classes={classes}")
    rates = [x["efold_per_P"] for x in deps if x.get("efold_per_P")]
    if rates:
        print(f"   e-folds/P: median={st.median(rates):.2f} "
              f"range=[{min(rates):.2f},{max(rates):.2f}]")
