"""Arm D, step 1: curate an SFT set for the advising fine-tune (the DHSS pattern: a frontier model writes the answers with the
same advising context, the rule graders keep only the ones that pass; held-out items are never used).

    cd ~/qwen-0.8b/wsp && ANTHROPIC_API_KEY=... .venv/bin/python finetune/curate.py            # --model claude-sonnet-5-5 --tries 2
Writes finetune/sft.jsonl ({"id", "source", "messages": [user, assistant]}) and finetune/sft_stats.json (kept / dropped, tokens, cost).
Sources: facts (50), scenarios (20), gpa_values (34), eligibility (28), fabrication (16), calibration (28), steering (36) — the
practice splits only; rubric items (ungraded) are included with the context-following answer unfiltered (20).
"""
import argparse
import glob
import json
import os
import sys
import time

WSP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, WSP)
PRACTICE = os.path.join(WSP, "practice")
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (1.0, 5.0), "claude-opus-5-5": (4.0, 20.0)}
SOURCES = ["facts", "scenarios", "gpa_values", "eligibility", "fabrication", "calibration", "steering", "rubric"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--context", default=os.path.join(WSP, "contexts", "advising.md"))
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(WSP, "finetune", "sft.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N items")
    args = ap.parse_args()
    import envfile
    envfile.load(os.path.join(WSP, ".env"))
    import anthropic
    import server as S
    client = anthropic.Anthropic()
    system = open(args.context).read().strip()
    items = []
    for src in SOURCES:
        p = os.path.join(PRACTICE, src + ".jsonl")
        if os.path.exists(p):
            items += [json.loads(l) for l in open(p) if l.strip()]
    items = [it for it in items if not it["id"].startswith("ho_")]
    if args.limit:
        items = items[:args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out)}
    out = open(args.out, "a")
    kept, dropped, tin, tout = 0, [], 0, 0
    for it in items:
        if it["id"] in done:
            continue
        ok_text = None
        for t in range(args.tries):
            for attempt in range(5):
                try:
                    resp = client.messages.create(model=args.model, max_tokens=700,              # no sampling knobs in this API; a retry is a fresh sample
                                                  system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                                                  messages=[{"role": "user", "content": it["problem"]}])   # the context is cached across the items
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  API error {type(e).__name__}: {str(e)[:100]} -> retry", flush=True)
                    time.sleep(15 * (attempt + 1))
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            tin += resp.usage.input_tokens + (resp.usage.cache_creation_input_tokens or 0) + (resp.usage.cache_read_input_tokens or 0)
            tout += resp.usage.output_tokens
            correct, _ = S.grade(it, text)
            if correct is None or correct == 1:      # ungraded (rubric) items are kept as written
                ok_text = text
                break
        if ok_text is None:
            dropped.append(it["id"])
            print(f"  dropped {it['id']} (failed the grader {args.tries}x)", flush=True)
            continue
        out.write(json.dumps({"id": it["id"], "source": it["source"], "messages": [{"role": "user", "content": it["problem"]},
                                                                                    {"role": "assistant", "content": ok_text}]}) + "\n")
        out.flush()
        kept += 1
        print(f"  kept {it['id']} ({kept} so far)", flush=True)
    pi, po = PRICES.get(args.model, (3.0, 15.0))
    stats = {"model": args.model, "kept": kept + len(done), "dropped": dropped, "tokens_in": tin, "tokens_out": tout,
             "cost_usd": round((tin * pi + tout * po) / 1e6, 2), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(stats, open(args.out[:-6] + "_stats.json" if args.out.endswith(".jsonl") else args.out + "_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
