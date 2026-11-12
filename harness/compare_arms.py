"""Assemble every evaluated arm into one comparison: results/full_test.json and results/full_test.md
(make_report.py renders the same JSON as a page).

    python3 compare_arms.py            # reads work/eval_*/summary.json, runs/full_*/turns/*/run.json, finetune/*stats.json, costs.json

Rows: the six held-out sets (accuracy and paired gain over the base arm ± SE), the calibration halves, the steering gap and
teacher-forced index, the severe-error rate (stated eligibility errors + invented specifics), then the cost model: build cost
(measured where a log exists), attended hours (assumed, costs.json), operating cost per 1,000 queries, refresh, mean gain per
$100 of build cost, severe errors avoided per $100, and the break-even query volume against the frontier arm.
"""
import glob
import json
import os

import numpy as np

WSP = os.path.dirname(os.path.abspath(__file__))
WORK, RUNS, RES, FT = (os.path.join(WSP, d) for d in ("work", "runs", "results", "finetune"))
ORDER = ["base", "astra_published", "fable", "opus", "astra", "kimi", "sonnet", "astra80", "astra_xhigh", "context", "finetune", "frontier", "fable_context", "opus_context", "astra_context", "kimi_context", "sonnet_context", "astra80_context", "astra_xhigh_context"]
SETS = {"heldout": "facts + scenarios", "heldout_gpa": "academic level", "heldout_eligibility": "eligibility",
        "heldout_fabrication": "fabrication", "heldout_calibration": "calibration", "heldout_steering": "steering (accuracy)"}


def editor_spend(arm):
    tot, turns, passes = 0.0, 0, None
    for p in glob.glob(os.path.join(RUNS, f"full_{arm}", "turns", "t*", "run.json")):
        tot += json.load(open(p)).get("turn_cost_usd", 0)
        turns += 1
    sp = os.path.join(RUNS, f"full_{arm}", "state.json")
    if os.path.exists(sp):
        passes = json.load(open(sp)).get("passes_used")
    return (tot if turns else None), turns, passes


def editor_minutes(arm):
    """wall time of the editor run, from arm.json 'started' to the final answer's mtime."""
    d = os.path.join(RUNS, f"full_{arm}")
    try:
        import time as _t
        t0 = _t.mktime(_t.strptime(json.load(open(os.path.join(d, "arm.json")))["started"], "%Y-%m-%d %H:%M:%S"))
        return (os.path.getmtime(os.path.join(d, "final_answer.md")) - t0) / 60
    except Exception:  # noqa: BLE001
        return None


def build_cost(arm, summ, costs):
    """measured build cost in USD and its provenance."""
    c = costs["arms"].get(arm, {})
    mac = costs.get("mac_usd_per_hour", 0.02)
    if arm in ("fable", "opus", "astra", "kimi", "astra80", "astra_xhigh", "sonnet"):
        usd, turns, passes = editor_spend(arm)
        return usd, (f"editor API spend, {turns} turns, {passes} passes" if usd is not None else "run not finished")
    if arm.endswith("_context"):
        usd, turns, passes = editor_spend(arm.replace("_context", ""))
        return (usd + 0 if usd is not None else None), "the edit's editor spend + the context file (shared)"
    if arm == "finetune":
        cur = os.path.join(FT, "sft_stats.json")
        tr = os.path.join(FT, "train_stats.json")
        if os.path.exists(cur):
            s = json.load(open(cur))
            mins = json.load(open(tr)).get("minutes_total", 0) if os.path.exists(tr) else 0
            return s.get("cost_usd", 0) + mac * mins / 60, f"curation API ${s.get('cost_usd', 0):.2f} ({s.get('kept')} pairs) + {mins:.0f} min of training"
        return None, "not run"
    if arm == "context":
        return 0.0, "a 1.3k-token file written by hand"
    if arm == "frontier":
        return 0.0, "no build; pays per query"
    if arm == "astra_published":
        return 0.0, "reuse of the paper's edit (their discovery: 50 passes, 158 turns)"
    return 0.0, ""


def main():
    costs = json.load(open(os.path.join(WSP, "costs.json")))
    arms = {}
    for p in sorted(glob.glob(os.path.join(WORK, "eval_*", "summary.json"))):
        name = os.path.basename(os.path.dirname(p))[5:]          # keyed by folder name, not the file's arm field
        s = json.load(open(p))
        if name.startswith("smoke") or name.endswith("_n1"):     # smoke runs and the n=1 previews are not arms
            continue
        if not s.get("sets") or any("mean_tokens" not in v for v in s["sets"].values()):
            print(f"skipping work/eval_{name}: summary in an older format", flush=True)
            continue
        s["arm"] = name
        arms[name] = s
    cols = [a for a in ORDER if a in arms] + sorted(a for a in arms if a not in ORDER)
    if not cols:
        print("no evaluations yet (work/eval_<arm>/summary.json)")
        return
    out = {"arms": {}, "sets": SETS, "order": cols}
    for a in cols:
        s = arms[a]
        c = costs["arms"].get(a, {"lever": a, "label": a, "attended_hours": None, "per_1k_queries_usd": None, "refresh": ""})
        usd, prov = build_cost(a, s, costs)
        per1k = s.get("api_usd_per_1k_queries", c.get("per_1k_queries_usd"))
        row = {"label": c["label"], "lever": c["lever"], "n_samples": s.get("n_samples"), "sets": {}, "mean_acc": s.get("mean_acc"),
               "mean_gain_pp": s.get("mean_gain_pp"), "severe_rate": s.get("severe_rate"), "steering_tf": s.get("steering_tf", {}).get("index_pp"),
               "build_usd": usd, "build_note": prov, "attended_hours": c.get("attended_hours"), "per_1k_queries_usd": per1k, "refresh": c.get("refresh"),
               "mean_tokens": float(np.mean([v["mean_tokens"] for v in s["sets"].values()])) if s.get("sets") else None}
        for k, v in s["sets"].items():
            row["sets"][k] = {"acc": v["acc"], "n": v["n"], "vs_base": v.get("vs_base"), "severe_rate": v.get("severe_rate"),
                              "should_hedge": v.get("should_hedge"), "should_be_confident": v.get("should_be_confident"),
                              "steering": v.get("steering")}
        g = row["mean_gain_pp"]
        row["gain_per_100usd"] = (g / usd * 100) if (g is not None and usd) else None
        hrs = c.get("attended_hours")
        row["gain_per_attended_hour"] = (g / hrs) if (g is not None and hrs) else None
        mins = s.get("eval_minutes") or 0
        em = editor_minutes(a) if a in ("fable", "opus", "astra", "kimi") else None
        tr = json.load(open(os.path.join(FT, "train_stats.json"))).get("minutes_total", 0) if (a == "finetune" and os.path.exists(os.path.join(FT, "train_stats.json"))) else 0
        row["mac_hours"] = round((mins + (em or 0) + tr) / 60, 1) if mins else None
        base = arms.get("base")
        if base and row["severe_rate"] is not None and base.get("severe_rate") is not None:
            row["severe_avoided_pp"] = 100 * (base["severe_rate"] - row["severe_rate"])
            row["severe_avoided_per_100usd"] = (row["severe_avoided_pp"] / usd * 100) if usd else None
        out["arms"][a] = row
    # break-even against the frontier arm: queries at which a local arm's build cost is paid back by its lower per-query cost
    fr = out["arms"].get("frontier")
    if fr and fr["per_1k_queries_usd"]:
        for a, row in out["arms"].items():
            if a != "frontier" and row["build_usd"] is not None and row["per_1k_queries_usd"] is not None:
                dq = (fr["per_1k_queries_usd"] - row["per_1k_queries_usd"]) / 1000
                row["breakeven_queries_vs_frontier"] = (row["build_usd"] / dq) if dq > 0 else None   # 0 when there is no build cost
    # markdown
    L = ["# The full test — every lever on the same held-out sets", "",
         "| | " + " | ".join(out["arms"][a]["label"] for a in cols) + " |", "|---|" + "---|" * len(cols)]

    def cell(a, s):
        r = out["arms"][a]["sets"].get(s)
        if not r:
            return "—"
        vb = r.get("vs_base")
        return f"{100 * r['acc']:.0f}%" + (f" ({vb['diff_pp']:+.0f}±{vb['se_pp']:.0f})" if vb else "")
    for s, lab in SETS.items():
        L.append(f"| {lab} | " + " | ".join(cell(a, s) for a in cols) + " |")
    k = max(len(out["arms"][a]["sets"]) for a in cols)
    L.append(f"| **mean over the {k} measured sets** | " + " | ".join((f"**{100 * out['arms'][a]['mean_acc']:.1f}%**" + (f" ({out['arms'][a]['mean_gain_pp']:+.1f})" if out['arms'][a]['mean_gain_pp'] is not None else "")) if out['arms'][a]['mean_acc'] is not None else "—" for a in cols) + " |")
    L.append("| severe-error rate (eligibility + fabrication) | " + " | ".join(f"{100 * out['arms'][a]['severe_rate']:.0f}%" if out['arms'][a]['severe_rate'] is not None else "—" for a in cols) + " |")
    L.append("| steering gap, generation (pp) | " + " | ".join(f"{out['arms'][a]['sets']['heldout_steering']['steering']['gap_pp']:+.0f}" if out['arms'][a]['sets'].get('heldout_steering') else "—" for a in cols) + " |")
    L.append("| steering index, teacher-forced (pp) | " + " | ".join(f"{out['arms'][a]['steering_tf']:+.1f}" if out['arms'][a]['steering_tf'] is not None else "—" for a in cols) + " |")
    L.append("| build cost | " + " | ".join(f"${out['arms'][a]['build_usd']:.0f}" if out['arms'][a]['build_usd'] is not None else "—" for a in cols) + " |")
    L.append("| attended hours (assumed) | " + " | ".join(f"{out['arms'][a]['attended_hours']}" for a in cols) + " |")
    L.append("| per 1,000 queries | " + " | ".join(f"${out['arms'][a]['per_1k_queries_usd']:.2f}" if out['arms'][a]['per_1k_queries_usd'] is not None else "—" for a in cols) + " |")
    L.append("| Mac time, measured (h) | " + " | ".join(f"{out['arms'][a]['mac_hours']}" if out['arms'][a].get('mac_hours') is not None else "—" for a in cols) + " |")
    L.append("| mean gain per attended hour | " + " | ".join(f"{out['arms'][a]['gain_per_attended_hour']:+.1f} pp" if out['arms'][a].get('gain_per_attended_hour') is not None else "—" for a in cols) + " |")
    L.append("| mean gain per $100 build | " + " | ".join(f"{out['arms'][a]['gain_per_100usd']:+.1f} pp" if out['arms'][a].get('gain_per_100usd') is not None else "—" for a in cols) + " |")
    L.append("| severe errors avoided per $100 build | " + " | ".join(f"{out['arms'][a]['severe_avoided_per_100usd']:+.1f} pp" if out['arms'][a].get('severe_avoided_per_100usd') is not None else "—" for a in cols) + " |")
    L.append("| break-even vs frontier (queries) | " + " | ".join(f"{out['arms'][a]['breakeven_queries_vs_frontier']:,.0f}" if out['arms'][a].get('breakeven_queries_vs_frontier') is not None else "—" for a in cols) + " |")
    L.append("| refresh when a fact changes | " + " | ".join(out["arms"][a]["refresh"] or "" for a in cols) + " |")
    L += ["", "Cells: held-out accuracy, then the paired gain over the base arm in pp ± SE (same prompts, same seeds, n samples per item). "
          "Severe errors: a stated eligibility error or an invented specific. Build cost is measured from logs where a run exists; attended hours "
          "and the local operating cost are assumptions in costs.json."]
    md = "\n".join(L)
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "full_test.md"), "w").write(md + "\n")
    json.dump(out, open(os.path.join(RES, "full_test.json"), "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
