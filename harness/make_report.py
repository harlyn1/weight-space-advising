"""Render results/full_test.json (from compare_arms.py) as one self-contained page: results/full_test.html.
Arms that have not run yet are shown as pending with their cost estimates from costs.json, so the page is the same
before and after the runs; cells fill in as arms finish.

    python3 make_report.py [--out results/full_test.html]
"""
import argparse
import html
import json
import os
import time

WSP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(WSP, "results")
ORDER = ["base", "astra_published", "fable", "opus", "astra", "kimi", "sonnet", "astra80", "astra_xhigh", "context", "finetune", "frontier", "fable_context", "opus_context", "astra_context", "kimi_context", "sonnet_context", "astra80_context", "astra_xhigh_context"]
SETS = [("heldout", "facts + scenarios", 30), ("heldout_gpa", "academic level", 16), ("heldout_eligibility", "eligibility", 22),
        ("heldout_fabrication", "fabrication", 14), ("heldout_calibration", "calibration", 21), ("heldout_steering", "steering", 36)]
LEVER_CLASS = {"none": "none", "weight editing (published)": "edit", "weight editing": "edit", "context injection": "ctx",
               "fine-tuning": "ft", "frontier model + context": "frontier", "weight editing + context": "combo"}
LEVER_NAME = {"none": "no change", "weight editing (published)": "weight editing", "weight editing": "weight editing",
              "context injection": "context injection", "fine-tuning": "fine-tuning", "frontier model + context": "frontier + context",
              "weight editing + context": "edit + context"}
SHORT = {"base": "as is", "astra_published": "published loop fix", "fable": "Fable 5.1", "opus": "Opus 5.5", "astra": "GPT-6 Astra", "kimi": "Kimi K3",
         "context": "context", "finetune": "LoRA", "frontier": "Sonnet 5.5 via API + ctx", "fable_context": "Fable + ctx",
         "opus_context": "Opus + ctx", "astra_context": "Astra + ctx", "kimi_context": "Kimi + ctx", "sonnet": "Sonnet edit", "astra80": "Astra 80p", "astra_xhigh": "Astra xhigh",
         "sonnet_context": "Sonnet edit + ctx", "astra80_context": "Astra 80p + ctx", "astra_xhigh_context": "Astra xhigh + ctx"}
E = html.escape


def pct(x, d=0):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def load():
    data = json.load(open(os.path.join(RES, "full_test.json"))) if os.path.exists(os.path.join(RES, "full_test.json")) else {"arms": {}, "order": []}
    costs = json.load(open(os.path.join(WSP, "costs.json")))
    arms = {}
    for a in ORDER:
        if a in data["arms"]:
            arms[a] = {**data["arms"][a], "status": "measured"}
        elif a in costs["arms"]:
            est = costs.get("estimates", {}).get(a, {})
            arms[a] = {"label": costs["arms"][a]["label"], "lever": costs["arms"][a]["lever"], "status": "pending", "sets": {},
                       "refresh": costs["arms"][a]["refresh"], "attended_hours": costs["arms"][a]["attended_hours"],
                       "per_1k_queries_usd": costs["arms"][a]["per_1k_queries_usd"], "est": est}
    return arms, costs


def delta_cell(r):
    if not r:
        return '<td class="cell pending"><span>pending</span></td>'
    vb = r.get("vs_base")
    if not vb:
        return f'<td class="cell"><b>{pct(r["acc"])}</b><small>base</small></td>'
    d, se = vb["diff_pp"], vb["se_pp"] or 0
    cls = "up" if d > se else "down" if d < -se else "flat"
    return f'<td class="cell"><b>{pct(r["acc"])}</b><small class="d {cls}">{d:+.0f} <span class="se">±{se:.0f}</span></small></td>'


def table_accuracy(arms):
    cols = list(arms)
    h = ['<div class="scroll"><table class="grid"><thead><tr><th class="rowhead">held-out set <small>(items)</small></th>']
    for a in cols:
        r = arms[a]
        h.append(f'<th class="arm {LEVER_CLASS[r["lever"]]}"><span class="lever">{E(LEVER_NAME[r["lever"]])}</span><span class="name">{E(SHORT[a])}</span>'
                 + ('' if r["status"] == "measured" else '<span class="tag">pending</span>')
                 + (f'<span class="tag n">n={r["n_samples"]}</span>' if r.get("n_samples") else '') + '</th>')
    h.append('</tr></thead><tbody>')
    for key, lab, n in SETS:
        h.append(f'<tr><th class="rowhead">{E(lab)} <small>({n})</small></th>')
        for a in cols:
            h.append(delta_cell(arms[a]["sets"].get(key)))
        h.append('</tr>')
        if key == "heldout_calibration":
            for fam, flab in (("should_hedge", "↳ should hedge or route (11)"), ("should_be_confident", "↳ settled facts (10)")):
                h.append(f'<tr class="sub"><th class="rowhead">{E(flab)}</th>')
                for a in cols:
                    r = arms[a]["sets"].get(key)
                    q = r.get(fam) if r else None
                    h.append(delta_cell(q) if q else '<td class="cell pending"><span>pending</span></td>')
                h.append('</tr>')
        if key == "heldout_steering":
            h.append('<tr class="sub"><th class="rowhead">↳ steering gap, generation <small>(marginalized − privileged, pp; 0 is fair)</small></th>')
            for a in cols:
                r = arms[a]["sets"].get(key)
                st = r.get("steering") if r else None
                h.append(f'<td class="cell"><b>{st["gap_pp"]:+.0f}</b></td>' if st else '<td class="cell pending"><span>pending</span></td>')
            h.append('</tr><tr class="sub"><th class="rowhead">↳ steering index, teacher-forced <small>(ΔP selective, 6 pairs, pp)</small></th>')
            for a in cols:
                t = arms[a].get("steering_tf")
                h.append(f'<td class="cell"><b>{t:+.1f}</b></td>' if t is not None else '<td class="cell pending"><span>' + ("n/a" if a == "frontier" else "pending") + '</span></td>')
            h.append('</tr>')
    h.append('<tr class="total"><th class="rowhead">mean over measured sets</th>')
    for a in cols:
        r = arms[a]
        if r["status"] != "measured" or r.get("mean_acc") is None:
            h.append('<td class="cell pending"><span>pending</span></td>')
        else:
            g = r.get("mean_gain_pp")
            h.append(f'<td class="cell"><b>{pct(r["mean_acc"], 1)}</b>' + (f'<small class="d {"up" if g > 0 else "down" if g < 0 else "flat"}">{g:+.1f}</small>' if g is not None else '<small>base</small>') + '</td>')
    h.append('</tr><tr class="total"><th class="rowhead">severe-error rate <small>(stated eligibility error or invented specific; lower is better)</small></th>')
    for a in cols:
        r = arms[a]
        h.append(f'<td class="cell"><b>{pct(r["severe_rate"])}</b></td>' if r.get("severe_rate") is not None else '<td class="cell pending"><span>pending</span></td>')
    h.append('</tr></tbody></table></div>')
    return "".join(h)


def money(x):
    return "—" if x is None else f"${x:,.0f}" if x >= 10 else f"${x:,.2f}"


def table_cost(arms):
    cols = list(arms)
    rows = []

    def est_range(a, key, unit=""):
        e = arms[a].get("est", {}).get(key)
        if not e:
            return "—"
        lo, hi = e
        return f"<span class='est'>{unit}{lo:g}–{unit}{hi:g} est.</span>" if lo != hi else f"<span class='est'>{unit}{lo:g} est.</span>"
    rows.append(("build cost <small>(measured from logs where a run exists)</small>",
                 [money(arms[a]["build_usd"]) + (f"<small class='note'>{E(arms[a]['build_note'])}</small>" if arms[a].get("build_note") else "")
                  if arms[a]["status"] == "measured" else est_range(a, "build_usd", "$") for a in cols]))
    rows.append(("Mac time <small>(editor run + training + evaluation)</small>",
                 [est_range(a, "mac_hours").replace("est.", "h est.") if arms[a]["status"] != "measured" else
                  (f"{arms[a]['mac_hours']} h" if arms[a].get("mac_hours") is not None else "<span class='est'>not logged</span>") for a in cols]))
    rows.append(("attended hours <small>(assumed)</small>", [f"{arms[a]['attended_hours']}" if arms[a].get("attended_hours") is not None else "—" for a in cols]))
    rows.append(("operating cost per 1,000 queries", [money(arms[a]["per_1k_queries_usd"]) if arms[a].get("per_1k_queries_usd") is not None else
                                                      ("<span class='est'>from the eval's tokens</span>" if a == "frontier" else "—") for a in cols]))
    rows.append(("mean gain per attended hour", [f"{arms[a]['gain_per_attended_hour']:+.1f} pp" if arms[a].get("gain_per_attended_hour") is not None else
                                                ("<span class='pend'>pending</span>" if arms[a]["status"] != "measured" else "—") for a in cols]))
    rows.append(("mean gain per $100 of build", [f"{arms[a]['gain_per_100usd']:+.1f} pp" if arms[a].get("gain_per_100usd") is not None else
                                               ("<span class='est'>build is $0</span>" if arms[a]["status"] == "measured" and not arms[a].get("build_usd") and arms[a].get("mean_gain_pp") is not None else "<span class='pend'>pending</span>" if arms[a]["status"] != "measured" else "—") for a in cols]))
    rows.append(("severe errors avoided per $100 of build", [f"{arms[a]['severe_avoided_per_100usd']:+.1f} pp" if arms[a].get("severe_avoided_per_100usd") is not None else
                                                           ("<span class='pend'>pending</span>" if arms[a]["status"] != "measured" else "—") for a in cols]))
    rows.append(("break-even vs the frontier arm <small>(queries until the build pays back)</small>",
                 [("0 <small class='note'>no build cost</small>" if arms[a].get("breakeven_queries_vs_frontier") == 0 else f"{arms[a]['breakeven_queries_vs_frontier']:,.0f}")
                  if arms[a].get("breakeven_queries_vs_frontier") is not None else ("<span class='pend'>needs the frontier arm</span>" if a != "frontier" else "—") for a in cols]))
    rows.append(("refresh when a fact changes", [f"<small>{E(arms[a]['refresh'] or '')}</small>" for a in cols]))
    h = ['<div class="scroll"><table class="grid cost"><thead><tr><th class="rowhead">cost and return</th>']
    for a in cols:
        r = arms[a]
        h.append(f'<th class="arm {LEVER_CLASS[r["lever"]]}"><span class="lever">{E(LEVER_NAME[r["lever"]])}</span><span class="name">{E(SHORT[a])}</span></th>')
    h.append('</tr></thead><tbody>')
    for lab, cells in rows:
        h.append(f'<tr><th class="rowhead">{lab}</th>' + "".join(f'<td class="cell">{c}</td>' for c in cells) + '</tr>')
    h.append('</tbody></table></div>')
    return "".join(h)


def bars_svg(arms, key, title, unit_lo, unit_hi, better="higher", fmt=lambda v: f"{100 * v:.0f}%"):
    """horizontal bars, one per arm, colored by lever; pending arms as dashed outlines at the base value."""
    cols = list(arms)
    base = arms.get("base", {}).get(key)
    W, rowh, left, right, top = 760, 30, 190, 70, 26
    H = top + rowh * len(cols) + 34
    x0, x1 = left, W - right
    sx = lambda v: x0 + (x1 - x0) * (v - unit_lo) / (unit_hi - unit_lo)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{E(title)}"><title>{E(title)}</title>']
    for t in [unit_lo + (unit_hi - unit_lo) * i / 4 for i in range(5)]:
        s.append(f'<line x1="{sx(t):.1f}" y1="{top - 6}" x2="{sx(t):.1f}" y2="{H - 30}" class="grid"/>'
                 f'<text x="{sx(t):.1f}" y="{H - 12}" class="tick" text-anchor="middle">{fmt(t)}</text>')
    if base is not None:
        s.append(f'<line x1="{sx(base):.1f}" y1="{top - 10}" x2="{sx(base):.1f}" y2="{H - 30}" class="ref"/>'
                 f'<text x="{sx(base) + 4:.1f}" y="{top - 12}" class="reflab">as is {fmt(base)}</text>')
    for i, a in enumerate(cols):
        y = top + rowh * i
        r = arms[a]
        cls = LEVER_CLASS[r["lever"]]
        s.append(f'<text x="{left - 10}" y="{y + 19}" class="lab" text-anchor="end">{E(SHORT[a])}</text>')
        v = r.get(key)
        if r["status"] == "measured" and v is not None:
            w = max(0.0, sx(v) - x0)
            s.append(f'<rect x="{x0}" y="{y + 6}" width="{w:.1f}" height="18" rx="3" class="bar {cls}"><title>{E(r["label"])}: {fmt(v)}</title></rect>'
                     f'<text x="{x0 + w + 6:.1f}" y="{y + 19}" class="val">{fmt(v)}</text>')
        else:
            w = max(0.0, (sx(base) if base is not None else x0 + 60) - x0)
            s.append(f'<rect x="{x0}" y="{y + 6}" width="{w:.1f}" height="18" rx="3" class="bar pending {cls}"/>'
                     f'<text x="{x0 + w + 6:.1f}" y="{y + 19}" class="val pend">pending</text>')
    s.append(f'<text x="{x0}" y="{H - 12}" class="axis" text-anchor="start" dy="0"></text></svg>')
    return "".join(s)


def scatter_svg(arms):
    """mean gain over the base (y) against build cost (x); pending arms as dashed cost ranges in a strip under the axis."""
    cols = [a for a in arms if a != "base"]
    pend = [a for a in cols if not (arms[a]["status"] == "measured" and arms[a].get("mean_gain_pp") is not None) and arms[a].get("est", {}).get("build_usd")]
    W, left, right, top = 760, 60, 30, 24
    lane_h = 16
    strip = 30 + lane_h * len(pend)              # the pending strip under the tick labels
    bottom = 24 + strip + 34
    H = 260 + bottom
    x0, x1, y0, y1 = left, W - right, H - bottom, top
    xmax = 260
    gains = [arms[a]["mean_gain_pp"] for a in cols if arms[a].get("mean_gain_pp") is not None]
    ymin, ymax = min(-15, min(gains) - 5 if gains else -15), max(25, max(gains) + 5 if gains else 25)
    sx = lambda v: x0 + (x1 - x0) * min(v, xmax) / xmax
    sy = lambda v: y0 - (y0 - y1) * (v - ymin) / (ymax - ymin)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="mean gain against build cost"><title>mean gain over the model as is, against build cost</title>']
    for t in range(0, xmax + 1, 50):
        s.append(f'<line x1="{sx(t):.1f}" y1="{y1}" x2="{sx(t):.1f}" y2="{y0}" class="grid"/><text x="{sx(t):.1f}" y="{y0 + 16}" class="tick" text-anchor="middle">${t}</text>')
    for t in range(int(ymin // 10 * 10), int(ymax) + 1, 10):
        if ymin <= t <= ymax:
            s.append(f'<line x1="{x0}" y1="{sy(t):.1f}" x2="{x1}" y2="{sy(t):.1f}" class="grid"/><text x="{x0 - 8}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t:+d}</text>')
    s.append(f'<line x1="{x0}" y1="{sy(0):.1f}" x2="{x1}" y2="{sy(0):.1f}" class="ref"/>')
    s.append(f'<text x="14" y="{(y0 + y1) / 2:.1f}" class="axis" text-anchor="middle" transform="rotate(-90 14 {(y0 + y1) / 2:.1f})">mean gain, pp</text>')
    for a in cols:
        r = arms[a]
        cls = LEVER_CLASS[r["lever"]]
        if r["status"] == "measured" and r.get("mean_gain_pp") is not None:
            x = sx(r.get("build_usd") or 0)
            y = sy(r["mean_gain_pp"])
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="dot {cls}"><title>{E(r["label"])}: {r["mean_gain_pp"]:+.1f} pp for {money(r.get("build_usd") or 0)}</title></circle>'
                     f'<text x="{x + 11:.1f}" y="{y + 4:.1f}" class="lab">{E(SHORT[a])}</text>')
    # the pending strip: each arm's estimated build cost as a dashed range, one lane each
    ys = y0 + 36
    if pend:
        s.append(f'<text x="{x0}" y="{ys}" class="pend" font-size="11">pending arms, estimated build cost (their height is what the run decides):</text>')
    for i, a in enumerate(pend):
        r = arms[a]
        cls = LEVER_CLASS[r["lever"]]
        lo, hi = r["est"]["build_usd"]
        yy = ys + 12 + lane_h * i + 6
        xa, xb = sx(lo), (sx(hi) if hi > lo else sx(lo) + 10)
        s.append(f'<line x1="{xa:.1f}" y1="{yy}" x2="{xb:.1f}" y2="{yy}" class="range {cls}"><title>{E(r["label"])}: ${lo:g}–${hi:g} estimated</title></line>'
                 f'<text x="{xb + 8:.1f}" y="{yy + 4}" class="lab" font-size="12">{E(SHORT[a])} <tspan class="pend">${lo:g}{("–$" + format(hi, "g")) if hi > lo else ""}</tspan></text>')
    s.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{H - 10}" class="axis" text-anchor="middle">build cost, USD (editor spend, curation, training; the context file and reuse of the published edit are $0)</text>')
    s.append('</svg>')
    return "".join(s)


def page(arms, costs):
    measured = [a for a in arms if arms[a]["status"] == "measured"]
    pending = [a for a in arms if arms[a]["status"] != "measured"]
    n_note = ", ".join(f"{SHORT[a]} n={arms[a]['n_samples']}" for a in measured if arms[a].get("n_samples"))
    prelim = any((arms[a].get("n_samples") or 2) < 2 for a in measured)
    gen = time.strftime("%Y-%m-%d %H:%M")
    legend = "".join(f'<span class="key"><i class="sw {c}"></i>{E(n)}</span>' for c, n in
                     [("none", "no change"), ("edit", "weight editing"), ("ctx", "context injection"), ("ft", "fine-tuning"), ("frontier", "frontier + context"), ("combo", "edit + context")])
    return f"""<title>Advising Levers Full Test</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column (max 1120px); tables and charts are the content and scroll in their own containers at phone width */
:root {{
  --bg:#f5f7fa; --surface:#ffffff; --fg:#18212b; --muted:#5d6977; --line:#d6dce4; --accent:#1f5f99; --accent-soft:#e6eef8;
  --edit:#2F6DB5; --ctx:#C9771E; --ft:#7A4FB0; --frontier:#1F9A6E; --none:#6b7280;
  --good:#1F9A6E; --bad:#C4452D; --pending:#9a7b10; --pending-bg:#fff7dc;
  --display:'Archivo', system-ui, sans-serif; --body:'Source Sans 3', system-ui, sans-serif; --mono:'JetBrains Mono', ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#121519; --surface:#1a1f26; --fg:#e7eaef; --muted:#9aa5b2; --line:#2d353f; --accent:#7fb0e6; --accent-soft:#1d2a3a;
  --edit:#5B93DD; --ctx:#C27D2A; --ft:#A079D2; --frontier:#34AA7F; --none:#8b95a3;
  --good:#34AA7F; --bad:#E06A52; --pending:#D8A93A; --pending-bg:#2a2410; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg:#121519; --surface:#1a1f26; --fg:#e7eaef; --muted:#9aa5b2; --line:#2d353f; --accent:#7fb0e6; --accent-soft:#1d2a3a;
  --edit:#5B93DD; --ctx:#C27D2A; --ft:#A079D2; --frontier:#34AA7F; --none:#8b95a3;
  --good:#34AA7F; --bad:#E06A52; --pending:#D8A93A; --pending-bg:#2a2410; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 16px; line-height: 1.5;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 0 48px; }}
.wrap {{ max-width: 1120px; margin: 0 auto; display: grid; gap: 40px; }}
header {{ padding-top: 40px; display: grid; gap: 10px; }}
.eyebrow {{ font-family: var(--mono); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font-family: var(--display); font-weight: 700; font-size: clamp(30px, 4vw, 44px); line-height: 1.05; margin: 0; text-wrap: balance; letter-spacing: -.01em; }}
h2 {{ font-family: var(--display); font-weight: 600; font-size: 22px; margin: 0 0 6px; text-wrap: balance; }}
p {{ max-width: 68ch; margin: 0; }}
.lede {{ font-size: 18px; color: var(--fg); }}
.status {{ display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; font-size: 14px; color: var(--muted); }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); font-family: var(--mono); font-size: 12px; color: var(--fg); }}
.chip.pend {{ border-style: dashed; color: var(--pending); border-color: var(--pending); background: var(--pending-bg); }}
section {{ display: grid; gap: 14px; }}
.scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
table.grid {{ border-collapse: separate; border-spacing: 0; width: 100%; min-width: 980px; font-variant-numeric: tabular-nums; font-size: 14px; }}
table.grid th, table.grid td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; text-align: left; }}
table.grid thead th {{ position: sticky; top: 0; background: var(--surface); z-index: 1; }}
th.rowhead {{ font-weight: 600; min-width: 220px; max-width: 300px; }}
th.rowhead small, td small.note {{ display: block; font-weight: 400; color: var(--muted); font-size: 12px; }}
th.arm {{ border-top: 4px solid var(--none); min-width: 96px; }}
th.arm .lever {{ display: block; font-family: var(--mono); font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }}
th.arm .name {{ display: block; font-family: var(--display); font-weight: 600; font-size: 14px; line-height: 1.2; margin-top: 2px; }}
th.arm .tag {{ display: inline-block; margin-top: 4px; font-family: var(--mono); font-size: 10.5px; color: var(--pending); border: 1px dashed var(--pending); border-radius: 4px; padding: 0 5px; }}
th.arm .tag.n {{ color: var(--muted); border-color: var(--line); border-style: solid; margin-left: 4px; }}
th.arm.edit, th.arm.combo {{ border-top-color: var(--edit); }} th.arm.ctx {{ border-top-color: var(--ctx); }} th.arm.ft {{ border-top-color: var(--ft); }} th.arm.frontier {{ border-top-color: var(--frontier); }}
th.arm.combo {{ border-top-style: double; }}
td.cell b {{ font-family: var(--mono); font-weight: 500; font-size: 14px; }}
td.cell small {{ display: block; font-family: var(--mono); font-size: 11.5px; color: var(--muted); }}
td.cell small.d.up {{ color: var(--good); }} td.cell small.d.down {{ color: var(--bad); }} td.cell small.d.flat {{ color: var(--muted); }}
td.cell .se {{ opacity: .7; }}
td.cell.pending span, span.pend {{ font-family: var(--mono); font-size: 11.5px; color: var(--pending); border: 1px dashed var(--pending); border-radius: 4px; padding: 1px 6px; white-space: nowrap; }}
span.est {{ font-family: var(--mono); font-size: 12px; color: var(--muted); }}
tr.sub th.rowhead {{ font-weight: 400; color: var(--muted); padding-left: 22px; }}
tr.total th, tr.total td {{ background: var(--accent-soft); }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
figure {{ margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 14px 8px; min-width: 0; }}
figcaption {{ font-size: 13px; color: var(--muted); padding: 4px 2px 6px; }}
figcaption b {{ color: var(--fg); font-family: var(--display); font-weight: 600; font-size: 14px; display: block; }}
svg {{ display: block; max-width: 100%; height: auto; font-family: var(--body); }}
svg .grid {{ stroke: var(--line); stroke-width: 1; }} svg .ref {{ stroke: var(--fg); stroke-width: 1; stroke-dasharray: 3 3; opacity: .7; }}
svg .tick, svg .axis {{ fill: var(--muted); font-size: 11px; font-family: var(--mono); }} svg .axis {{ font-family: var(--body); font-size: 12px; }}
svg .lab {{ fill: var(--fg); font-size: 13px; }} svg .val {{ fill: var(--fg); font-size: 12px; font-family: var(--mono); }} svg .reflab {{ fill: var(--muted); font-size: 11px; font-family: var(--mono); }}
svg .pend {{ fill: var(--pending); font-family: var(--mono); }}
svg .bar {{ fill: var(--none); }} svg .bar.edit, svg .bar.combo, svg .dot.edit, svg .dot.combo {{ fill: var(--edit); }} svg .bar.ctx, svg .dot.ctx {{ fill: var(--ctx); }} svg .bar.ft, svg .dot.ft {{ fill: var(--ft); }} svg .bar.frontier, svg .dot.frontier {{ fill: var(--frontier); }}
svg .bar.pending {{ fill: none; stroke: var(--pending); stroke-dasharray: 4 3; stroke-width: 1.2; }}
svg .dot {{ stroke: var(--surface); stroke-width: 2; }}
svg .range {{ stroke-width: 6; stroke-linecap: round; stroke-dasharray: 2 6; stroke: var(--pending); }}
svg .range.edit, svg .range.combo {{ stroke: var(--edit); }} svg .range.ctx {{ stroke: var(--ctx); }} svg .range.ft {{ stroke: var(--ft); }} svg .range.frontier {{ stroke: var(--frontier); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--muted); }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }} .sw {{ width: 12px; height: 12px; border-radius: 3px; background: var(--none); display: inline-block; }}
.sw.edit {{ background: var(--edit); }} .sw.ctx {{ background: var(--ctx); }} .sw.ft {{ background: var(--ft); }} .sw.frontier {{ background: var(--frontier); }}
.sw.combo {{ background: repeating-linear-gradient(45deg, var(--edit) 0 3px, var(--ctx) 3px 6px); }}
.rules {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
.rule {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; min-width: 0; }}
.rule h3 {{ margin: 0 0 6px; font-family: var(--display); font-size: 15px; font-weight: 600; }}
.rule p {{ font-size: 14px; color: var(--muted); }}
.notes {{ font-size: 14px; color: var(--muted); display: grid; gap: 6px; }}
.notes code {{ font-family: var(--mono); font-size: 12.5px; color: var(--fg); }}
footer {{ font-size: 12.5px; color: var(--muted); font-family: var(--mono); }}
@media (max-width: 560px) {{ body {{ font-size: 15px; }} .lede {{ font-size: 16px; }} th.rowhead {{ min-width: 170px; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">Weight Programming v1 · Qwen3.5-0.8B as a college advisor · one held-out test, every lever</div>
  <h1>Which lever improves the advisor, and what each point costs</h1>
  <p class="lede">Every arm answers the same 139 held-out prompts with the same seeds, graded by the same rules: the six families below, a severe-error rate for the two that can hurt a student, the demographic-steering gap, and the build and operating cost of getting there.</p>
  <div class="status">
    <span class="chip">{len(measured)} arm{'s' if len(measured) != 1 else ''} measured</span>
    <span class="chip pend">{len(pending)} pending</span>
    {'<span>' + E(n_note) + ('; n=1 figures are preliminary, the full evaluation uses n=2' if prelim else '') + '</span>' if n_note else ''}
    <span>generated {gen}</span>
  </div>
  <div class="legend">{legend}</div>
</header>

<section>
  <h2>Accuracy by family, and the paired gain over the model as is</h2>
  <p>Each cell is the arm's held-out accuracy; below it, the gain over the base arm in percentage points with its standard error (paired by prompt and seed). Green and red mean the gain is larger than one standard error; grey means it is inside the noise.</p>
  {table_accuracy(arms)}
</section>

<section>
  <h2>Two numbers per arm</h2>
  <div class="charts">
    <figure>{bars_svg(arms, "mean_acc", "mean held-out accuracy over the measured families", 0.0, 1.0)}<figcaption><b>Mean held-out accuracy</b> Higher is better; the dashed line is the model as is. Pending arms are drawn hollow at the base value.</figcaption></figure>
    <figure>{bars_svg(arms, "severe_rate", "severe-error rate on the eligibility and fabrication sets", 0.0, 0.5)}<figcaption><b>Severe-error rate</b> Share of eligibility and fabrication answers that state an eligibility error or invent a specific. Lower is better.</figcaption></figure>
  </div>
</section>

<section>
  <h2>Cost, return and what it takes to keep it current</h2>
  <p>Build cost is read from the run logs (editor API spend per turn, curation tokens, training minutes at an electricity rate). Attended hours and the local model's operating cost are assumptions kept in <code>costs.json</code>. The frontier arm has no build cost and pays per query, so the break-even row says how many queries a local arm needs before its build has paid for itself.</p>
  {table_cost(arms)}
  <div class="charts">
    <figure>{scatter_svg(arms)}<figcaption><b>Gain against build cost</b> Measured arms are dots. Pending arms are shown as their estimated cost range along the bottom; their height is what the run decides.</figcaption></figure>
  </div>
</section>

<section>
  <h2>How to read the finished table</h2>
  <div class="rules">
    <div class="rule"><h3>Capacity or lever?</h3><p>If Sonnet 5.5 + context sits far above context on the small model, the deficit is capacity and the lever is the base model, whatever the small-model arms do.</p></div>
    <div class="rule"><h3>Knowledge or obstacle?</h3><p>If context moves eligibility and fabrication and none of the three editors does, the deficit is knowledge and the paper's line holds: weight editing removes obstacles, it does not add facts.</p></div>
    <div class="rule"><h3>Does weight editing earn a place?</h3><p>If an editor's arm moves academic level or calibration without raising the severe-error rate, and the edit + context arm keeps both gains, the levers stack and the edit belongs in the deployment.</p></div>
    <div class="rule"><h3>Which editor?</h3><p>Fable 5.1, Opus 5.5, GPT-6 Astra and Kimi K3 get the same brief, budget and starting point; their rows differ only by what each found, and their transcripts show where the reasoning diverged. Kimi's row says whether the editor has to be frontier-priced.</p></div>
  </div>
</section>

<section>
  <div class="notes">
    <span><b>Protocol.</b> Chat template, thinking off, non-thinking sampling (T 1.0, top-p 1.0, top-k 20, presence 2.0), at most 1,024 new tokens, n samples per item with seeds from the item id, the same for every arm; context arms get <code>contexts/advising.md</code> (about 1,300 tokens) as the system prompt; the frontier arm uses the API's default sampling with the same prompt and the same n.</span>
    <span><b>Sets.</b> facts + scenarios 30 · academic level 16 · eligibility 22 · fabrication 14 · calibration 21 (11 should hedge or route, 10 settled facts) · steering 36 (3 academic profiles × 12 demographic sentences; the correct answer is the same for all 12). Grading is by regex checks validated on the base model's practice outputs; the same grader noise applies to every arm. The teacher-forced steering index scores log P("I'd recommend applying to selective universities") after six new questions with twelve demographic sentences each.</span>
    <span><b>Arms.</b> base: the model as is · published loop fix: the paper's edit, reused · Fable 5.1 / Opus 5.5 / GPT-6 Astra / Kimi K3 (open-weight, a quarter of Fable's price): that editor's 40-pass edit on the same brief, starting from the loop fix · context: the model as is + the advising context · LoRA: a fine-tune on answers written by Sonnet 5.5 with the context and kept only when they pass the graders · Sonnet 5.5 + context: the capacity ceiling · edit + ctx: an editor's edit with the context.</span>
    <span><b>Files.</b> <code>results/full_test.json</code> is the source of this page; <code>work/eval_&lt;arm&gt;/summary.md</code> has each arm's detail and <code>runs/full_&lt;arm&gt;/</code> the editor transcripts.</span>
  </div>
</section>
<footer>Weight Programming v1 · make_report.py · {gen}</footer>
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "full_test.html"))
    args = ap.parse_args()
    arms, costs = load()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(page(arms, costs))
    print(f"wrote {args.out} ({sum(1 for a in arms if arms[a]['status'] == 'measured')} measured, {sum(1 for a in arms if arms[a]['status'] != 'measured')} pending)")


if __name__ == "__main__":
    main()
