"""The editor loop: a frontier model in the editor's seat (the paper's harness, with the model chosen by --model).

    cd ~/qwen-0.8b/wsp && ANTHROPIC_API_KEY=... .venv/bin/python editor_api.py --task tasks/q2_full.md --run full_fable
    --model claude-fable-5-1 (default) | claude-opus-5-5 (ANTHROPIC_API_KEY) | gpt-6-astra (OPENAI_API_KEY; the paper's editor)
            | accounts/fireworks/models/kimi-k3 (FIREWORKS_API_KEY; Fireworks' OpenAI-compatible endpoint; any Fireworks model id works)
            | kimi-k3 (MOONSHOT_API_KEY; Moonshot's own endpoint, reasoning_effort low|high|max)

Protocol, as in lm_understand-logs/exp8_math: one conversation; each turn the editor replies with an explanation and
exactly one ```python or ```forward block (or, at the end, no block); the harness writes it to requests/, waits for the
server's result, and sends the output back as the next user message. Every 40 turns the conversation is restarted
with a handoff message (the task, the editor's notes.md and its current edit). Everything is logged under
runs/<run>/turns/tNNN/{prompt.txt, answer.md} plus tool_calls.jsonl and final_answer.md, like the paper's logs.

server.py must be running in another terminal.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

WSP = os.path.dirname(os.path.abspath(__file__))
REQ, WORK, RUNS = (os.path.join(WSP, d) for d in ("requests", "work", "runs"))
EXPLAIN = ("EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 to "
           "8 sentences): what the last result showed, what you conclude from it, what you will do now and why, and what "
           "outcome would confirm or refute your current hypothesis. In your final answer, also summarize the path that "
           "led to your edits.")
# list prices, USD per million tokens: (input, cache write, cache read, output). Anthropic: Fable 5.1 and Opus 5.5 ($4/$20;
# cache write 1.25x and cache read 0.1x of input, the standard multipliers). OpenAI gpt-6-astra: $10/$50 (OpenRouter and
# press listings, Sep 2026); cached-input price not published here, assumed 0.1x; reasoning tokens are billed as output.
PRICES = {"claude-fable-5-1": (10.0, 12.5, 0.25, 50.0), "claude-opus-5-5": (4.0, 5.0, 0.4, 20.0), "claude-sonnet-5-5": (3.0, 3.75, 0.3, 15.0), "gpt-6-astra": (10.0, 0.0, 1.0, 50.0),
          "kimi-k3": (2.5, 0.0, 0.29, 14.0),   # Kimi K3 at Moonshot (pricepertoken, Nov 2026)
          "accounts/fireworks/models/kimi-k3": (3.0, 0.0, 0.30, 15.0)}   # Kimi K3 at Fireworks (fireworks.ai, Nov 2026)


def provider(model):
    if model.startswith(("accounts/fireworks", "fireworks/")):
        return "fireworks"
    if model.startswith("kimi"):
        return "moonshot"
    return "openai" if model.startswith(("gpt", "o1", "o3", "o4")) else "anthropic"


def next_k(prefix, ext):
    ks = [int(m.group(1)) for p in glob.glob(f"{REQ}/{prefix}_*")
          if (m := re.fullmatch(prefix + r"_(\d+)" + re.escape(ext), os.path.basename(p)))]
    return max(ks, default=0) + 1


SERVER_DEAD = "[the forward-pass server is not running]"
PARENT_GONE = "[this editor's run (run_compare.sh) is gone]"
LONG_CALL_SECONDS = float(os.environ.get("WSP_API_TIMEOUT", "2400"))
LAST_RAW = None                                                         # the last provider message as returned (OpenAI-compatible providers), for the turn log
# Kimi K3 (Fireworks/Moonshot) is tool-trained: without tools it ends its turn after announcing what it will do, so the same two
# actions are also offered as functions; a call is converted into the ```python / ```forward block the harness runs.
TOOLS = [
    {"type": "function", "function": {"name": "python", "description": "Run one Python cell in the persistent kernel (identical to a "
     "```python block in your reply; variables persist between cells; the output comes back as the next message).",
     "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "the Python code to run"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "forward", "description": "Request one forward pass (identical to a ```forward block): the "
     "argument is the JSON request described in the task, as text. It counts against the pass budget.",
     "parameters": {"type": "object", "properties": {"request": {"type": "string", "description": "the JSON request"}}, "required": ["request"]}}},
]   # per-call ceiling for the OpenAI-compatible providers (40 min)


def server_alive():
    """the server writes its pid to work/server.pid; alive if that process exists. Any doubt counts as alive."""
    try:
        p = os.path.join(WORK, "server.pid")
        if os.path.exists(p):
            pid = int(open(p).read().strip())
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
        return subprocess.run(["pgrep", "-f", "server.py --budget"], capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001
        return True


def server_dead():
    """two checks 20 s apart must both fail: a single reading is not trusted."""
    if server_alive():
        return False
    time.sleep(20)
    return not server_alive()


def parent_gone():
    """True once the process that started this editor (run_compare.sh's shell) is gone: on macOS an orphan is re-parented to
    launchd (pid 1). An editor without its run would otherwise keep driving whatever server is up, for hours and dollars."""
    try:
        return os.getppid() == 1
    except Exception:  # noqa: BLE001
        return False


def snapshot_partial(run_dir):
    """copy this arm's current edits, notes and pass counter into runs/<run>/ so a resume starts from the latest state even if
    the run's shell never reaches its own snapshot step (the machine slept, the window was closed, the shell was killed)."""
    try:
        if os.path.exists(os.path.join(run_dir, "final_answer.md")):
            return
        tmp, dst = os.path.join(run_dir, "edited_partial.tmp"), os.path.join(run_dir, "edited_partial")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for p in glob.glob(os.path.join(WSP, "edited", "*.npy")):
            shutil.copy(p, tmp)
        shutil.rmtree(dst, ignore_errors=True)
        os.rename(tmp, dst)
        for src, name in ((os.path.join(WORK, "notes.md"), "notes_partial.md"), (os.path.join(WSP, "state.json"), "state_partial.json")):
            if os.path.exists(src):
                shutil.copy(src, os.path.join(run_dir, name))
    except Exception as e:  # noqa: BLE001
        print(f"[snapshot of the partial state failed: {type(e).__name__}: {str(e)[:120]}]", flush=True)


def wait_for(path, timeout):
    """poll for the server's answer; give up early if the server process has died or this editor's run is gone."""
    t0, last_check = time.time(), time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            return None
        if parent_gone():
            return PARENT_GONE
        if time.time() - last_check > 60:
            last_check = time.time()
            if server_dead():
                return SERVER_DEAD
        time.sleep(2)
    time.sleep(0.5)
    return open(path).read()


def run_block(kind, body):
    if parent_gone():
        return "none", PARENT_GONE
    if server_dead():
        return "none", SERVER_DEAD
    if kind == "python":
        k = next_k("code", ".py")
        open(f"{REQ}/code_{k}.py", "w").write(body)
        lim = int(os.environ.get("WSP_CODE_SECONDS", "120")) + 60
        out = wait_for(f"{REQ}/code_{k}.out.txt", timeout=lim)
        return f"code_{k}", out if out is not None else f"[the kernel did not answer within {lim} s]"
    k = next_k("fwd", ".json")
    try:
        json.loads(body)
    except Exception as e:  # noqa: BLE001
        return f"fwd_{k}", f"[forward block is not valid JSON: {e}; the pass was not run and did not count]"
    open(f"{REQ}/fwd_{k}.json", "w").write(body)
    out = wait_for(f"{REQ}/fwd_{k}.done.json", timeout=6 * 3600)
    return f"fwd_{k}", out if out is not None else "[the forward pass did not finish within 6 hours]"


BLOCK = re.compile(r"```(python|forward)\s*\n(.*?)```", re.S)


def extract_block(text):
    blocks = BLOCK.findall(text)
    if not blocks:
        return None
    return blocks[-1][0], blocks[-1][1]


def state():
    """the server's pass counter. Before the server has answered anything in a fresh arm the file may not exist yet: that is
    an unknown budget, never an exhausted one (a budget of 0 would end the run on the first reply without a code block)."""
    p = os.path.join(WSP, "state.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:  # noqa: BLE001  (a write in progress)
            time.sleep(1)
            return json.load(open(p))
    return {"budget": None, "passes_used": 0}


def budget_exhausted(s):
    return s.get("budget") is not None and s.get("passes_used", 0) >= s["budget"]


def work_listing(limit=120):
    """what is in WORK, so a restarted editor does not search for its files."""
    names = sorted(os.listdir(WORK)) if os.path.isdir(WORK) else []
    keep = [n for n in names if not n.startswith(("gen_ev_", "fwd_ev_", "eval_", "invalid", "preview", "state_"))]
    shown = keep[:limit]
    return ", ".join(shown) + (f" ... and {len(keep) - limit} more" if len(keep) > limit else "")


def handoff_message(task, notes, interrupted=False):
    s = state()
    sys.path.insert(0, WSP)
    import kernel as K
    return (f"{task}\n\n---\nHANDOFF. This is a fresh conversation continuing your earlier work (the previous one was "
            f"restarted). Forward passes used: {s['passes_used']} of {s['budget'] if s.get('budget') is not None else '?'}. Edited tensors currently in place: "
            f"{K.edited() or 'none'}. Your notes (WORK/notes.md):\n\n{notes or '(empty)'}\n\n"
            + ("The kernel's variables are gone (the server was restarted), but every file you saved is in WORK. " if interrupted
               else "The kernel's variables were kept. ")
            + f"Contents of WORK (your folder; forward-pass results are WORK/fwd_<k>/ and WORK/gen_<k>*.jsonl, request files are "
            f"not in WORK): {work_listing()}\n\nDo not search the filesystem for files: everything of yours is in WORK or edited/. "
            f"Kernel cells are limited to {os.environ.get('WSP_CODE_SECONDS', '120')} s each. Continue from your notes.")


class Editor:
    """One API call per turn. The conversation is kept provider-neutral: user turns are strings; assistant turns keep the
    provider's own content (Anthropic blocks, so thinking signatures round-trip; plain text for OpenAI)."""

    def __init__(self, model, effort, max_tokens):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.prov = provider(model)
        if self.prov == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        elif self.prov == "fireworks":
            from openai import OpenAI
            # a reasoning model's turn can run for many minutes (Kimi K3 thinks at length): the SDK's default 10-minute
            # timeout would abandon the call and its automatic retries would bill the same reasoning twice
            self.client = OpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1",
                                 timeout=LONG_CALL_SECONDS, max_retries=1)
            self.model = model if model.startswith("accounts/") else "accounts/fireworks/models/" + model.split("/", 1)[1]
        elif self.prov == "moonshot":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["MOONSHOT_API_KEY"], base_url="https://api.moonshot.ai/v1", timeout=LONG_CALL_SECONDS, max_retries=1)
            if self.effort not in ("low", "high", "max"):          # K3 accepts low | high | max
                self.effort = {"medium": "low", "xhigh": "max"}.get(self.effort, "high")
        else:
            from openai import OpenAI
            self.client = OpenAI(timeout=LONG_CALL_SECONDS, max_retries=1)
        self.price = PRICES.get(model, PRICES["claude-fable-5-1"])

    def _too_long(self, e):
        """a rejected output ceiling: halve it (down to 8k) and tell the caller to retry."""
        msg = str(e).lower()
        if ("max_tokens" in msg or "max_completion_tokens" in msg or "output tokens" in msg) and self.max_tokens > 8000:
            self.max_tokens = max(8000, self.max_tokens // 2)
            print(f"output ceiling rejected by {self.model}: retrying with max_tokens={self.max_tokens}", flush=True)
            return True
        return False

    def call(self, messages):
        """returns (text, usage dict, cost usd, stop reason, the assistant message to append)."""
        if self.prov == "anthropic":
            for _ in range(4):
                kw = dict(model=self.model, max_tokens=self.max_tokens, messages=messages, cache_control={"type": "ephemeral"})
                try:
                    # streamed: a max-effort turn can run past the SDK's 10-minute limit for non-streaming calls
                    with self.client.messages.stream(**kw, output_config={"effort": self.effort}) as stream:
                        resp = stream.get_final_message()
                    break
                except Exception as e:  # noqa: BLE001
                    if "effort" in str(e).lower() and "streaming" not in str(e).lower() and self.effort != "high":   # effort level not accepted
                        print(f"effort '{self.effort}' rejected by {self.model}: using 'high'", flush=True)
                        self.effort = "high"
                    elif not self._too_long(e):
                        raise
            text = "".join(b.text for b in resp.content if b.type == "text")
            u = resp.usage.model_dump()
            inp, cw, cr, out = (u.get("input_tokens", 0), u.get("cache_creation_input_tokens") or 0,
                                u.get("cache_read_input_tokens") or 0, u.get("output_tokens", 0))
            msg = {"role": "assistant", "content": [b.model_dump() for b in resp.content]}
            stop = resp.stop_reason
        else:
            conv = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                     else "".join(b.get("text", "") for b in m["content"] if b.get("type") == "text")} for m in messages]
            for _ in range(6):
                kw = dict(model=self.model, messages=conv, max_completion_tokens=self.max_tokens)
                if self.prov in ("fireworks", "moonshot") and not getattr(self, "no_tools", False):
                    kw.update(tools=TOOLS, tool_choice="auto")
                try:
                    resp = (self.client.chat.completions.create(**kw, reasoning_effort=self.effort) if self.effort != "none"
                            else self.client.chat.completions.create(**kw))
                    break
                except Exception as e:  # noqa: BLE001
                    msg = str(e).lower()
                    if self._too_long(e):
                        continue
                    if "tool" in msg and "tools" in kw:
                        print(f"tools rejected by {self.model}: continuing with markdown blocks only ({str(e)[:120]})", flush=True)
                        self.no_tools = True
                        continue
                    if "effort" in msg and os.environ.get("WSP_EFFORT_STRICT"):
                        # an arm that exists to test a specific effort level must not silently run at another one
                        print(f"reasoning_effort '{self.effort}' rejected by {self.model} and WSP_EFFORT_STRICT is set: stopping this arm ({str(e)[:160]})", flush=True)
                        sys.exit(3)
                    if ("effort" in msg or "reasoning" in msg or "unknown" in msg or "unsupported" in msg or "invalid" in msg) and self.effort != "none":
                        # gpt-6-astra documents low/medium/high; a provider that does not take the parameter at all gets the call without it
                        nxt = "high" if self.effort not in ("high", "none") and "effort" in msg else "none"
                        print(f"reasoning_effort '{self.effort}' rejected by {self.model}: {'using high' if nxt == 'high' else 'sending the call without it'}", flush=True)
                        self.effort = nxt
                    else:
                        raise
            m = resp.choices[0].message
            text = m.content or ""
            global LAST_RAW
            try:
                LAST_RAW = m.model_dump()
            except Exception:  # noqa: BLE001
                LAST_RAW = {"content": text}
            # a tool-trained model (Kimi K3) may answer with a native function call instead of a markdown block: take the code from it
            for tc in (getattr(m, "tool_calls", None) or []):
                try:
                    fn = tc.function
                    try:
                        args = json.loads(fn.arguments or "{}")
                    except Exception:  # noqa: BLE001
                        args = {"code": fn.arguments}
                    if fn.name == "forward":
                        body = args.get("request") or args.get("json") or args
                        body = body if isinstance(body, str) else json.dumps(body)
                        text += f"\n\n```forward\n{body.strip()}\n```"
                        print("[tool call 'forward' converted to a forward block]", flush=True)
                        continue
                    code = args.get("code") or args.get("python") or args.get("source") or args.get("command") or args.get("input") or ""
                    if isinstance(code, str) and code.strip():
                        text += f"\n\n```python\n{code.rstrip()}\n```"
                        print(f"[tool call '{fn.name}' converted to a python block]", flush=True)
                except Exception:  # noqa: BLE001
                    pass
            # or the block may sit in the reasoning while the visible reply only announces it
            rc = getattr(m, "reasoning_content", None) or (LAST_RAW.get("reasoning_content") if isinstance(LAST_RAW, dict) else None) or ""
            if rc and not BLOCK.search(text) and BLOCK.search(rc):
                b = BLOCK.findall(rc)[-1]
                text += f"\n\n```{b[0]}\n{b[1].rstrip()}\n```"
                print("[code block taken from the reasoning: the visible reply had none]", flush=True)
            u = resp.usage.model_dump()
            cr = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            inp, cw, out = u.get("prompt_tokens", 0) - cr, 0, u.get("completion_tokens", 0)
            msg = {"role": "assistant", "content": text}
            stop = resp.choices[0].finish_reason
        pi, pcw, pcr, po = self.price
        cost = (inp * pi + cw * pcw + cr * pcr + out * po) / 1e6
        return text, u, cost, stop, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="the task message file (tasks/*.md)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="claude-fable-5-1")
    ap.add_argument("--effort", default="max", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max_turns", type=int, default=5000)   # the pass budget (and the spend cap) are the stopping rules, not this
    ap.add_argument("--handoff_every", type=int, default=40)
    ap.add_argument("--max_tokens", type=int, default=64000)   # hard ceiling on thinking + reply per turn
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from work/notes.md and edited/ (the paper's handoff protocol)")
    args = ap.parse_args()

    import envfile
    envfile.load(os.path.join(WSP, ".env"))        # keys from wsp/.env when not exported
    ed = Editor(args.model, args.effort, args.max_tokens)
    task = open(args.task).read().strip()
    if "EXPLAIN EVERY STEP" not in task:
        task += "\n\n" + EXPLAIN
    run_dir = os.path.join(RUNS, args.run)
    os.makedirs(os.path.join(run_dir, "turns"), exist_ok=True)
    open(os.path.join(run_dir, "message.txt"), "w").write(task)
    arm_p, now = os.path.join(run_dir, "arm.json"), time.strftime("%Y-%m-%d %H:%M:%S")
    arm = json.load(open(arm_p)) if args.resume and os.path.exists(arm_p) else {"model": args.model, "effort": args.effort, "task": args.task, "started": now}
    arm.setdefault("sessions", [])                                    # one entry per (re)start, so wall time can be summed per session
    arm["sessions"].append({"started": now, "turn_from": len(glob.glob(os.path.join(run_dir, "turns", "t*"))) + 1, "pid": os.getpid(), "ppid": os.getppid()})
    json.dump(arm, open(arm_p, "w"), indent=1)
    calls = open(os.path.join(run_dir, "tool_calls.jsonl"), "a")

    messages = [{"role": "user", "content": task}]
    turn, since_handoff, no_block_streak, total_cost = 0, 0, 0, 0.0
    done_turns = sorted(glob.glob(os.path.join(run_dir, "turns", "t*", "run.json")))
    if args.resume and done_turns:
        turn = len(glob.glob(os.path.join(run_dir, "turns", "t*")))
        total_cost = sum(json.load(open(p)).get("turn_cost_usd", 0) for p in done_turns)
        notes_p = os.path.join(WORK, "notes.md")
        notes = open(notes_p).read() if os.path.exists(notes_p) else ""
        messages = [{"role": "user", "content": handoff_message(task, notes, interrupted=True) + "\n\nThe previous session was "
                     "interrupted (the machine, the API or a harness limit stopped it) and the server was restarted; edited/ and WORK "
                     "are intact, and the forward-pass count continues from where it was."}]
        print(f"resuming {args.run} at turn {turn + 1}, ${total_cost:.2f} spent so far", flush=True)
    max_cost = float(os.environ.get("WSP_MAX_COST_USD", "0") or 0)   # optional spend cap: stop the run when exceeded
    while turn < args.max_turns:
        turn += 1
        since_handoff += 1
        tdir = os.path.join(run_dir, "turns", f"t{turn:03d}")
        os.makedirs(tdir, exist_ok=True)
        open(os.path.join(tdir, "prompt.txt"), "w").write(messages[-1]["content"] if isinstance(messages[-1]["content"], str)
                                                          else json.dumps(messages[-1]["content"]))
        t0 = time.time()
        if parent_gone():
            snapshot_partial(run_dir)
            print(f"[turn {turn}] this editor's run (run_compare.sh) is gone; stopping so no orphan keeps driving the server. "
                  "Partial state snapshotted; re-run run_compare.sh to resume.", flush=True)
            return
        res = None
        for attempt in range(6):          # transient API errors (rate limits, overload): wait and retry
            try:
                res = ed.call(messages)
                break
            except Exception as e:  # noqa: BLE001
                wait = 30 * (2 ** attempt)
                print(f"[turn {turn}] API error: {type(e).__name__}: {str(e)[:200]} -> retry in {wait}s", flush=True)
                time.sleep(wait)
        if res is None:
            snapshot_partial(run_dir)
            print("giving up after repeated API errors; the run can be resumed later from work/notes.md", flush=True)
            return
        text, u, cost, stop, msg = res
        open(os.path.join(tdir, "answer.md"), "w").write(text)
        total_cost += cost
        if LAST_RAW is not None:
            try:
                json.dump(LAST_RAW, open(os.path.join(tdir, "raw_message.json"), "w"), indent=1, default=str)
            except Exception:  # noqa: BLE001
                pass
        json.dump({"model": args.model, "effort": ed.effort, "usage": u, "turn_cost_usd": round(cost, 4),
                   "run_cost_usd_so_far": round(total_cost, 2), "stop_reason": stop,
                   "seconds": round(time.time() - t0)}, open(os.path.join(tdir, "run.json"), "w"), indent=1)
        print(f"[turn {turn}] usage {json.dumps({k: v for k, v in u.items() if isinstance(v, int)})}, "
              f"${cost:.2f} this turn, ${total_cost:.2f} so far", flush=True)
        messages.append(msg)
        if max_cost and total_cost >= max_cost:
            open(os.path.join(run_dir, "final_answer.md"), "w").write(text + f"\n\n[run stopped by the spend cap of ${max_cost:.0f} after turn {turn}]")
            print(f"spend cap ${max_cost:.0f} reached (${total_cost:.2f}); stopping. Edits so far are in edited/.", flush=True)
            return
        print(f"[turn {turn}] {len(text)} chars, stop={stop}", flush=True)

        blk = extract_block(text)
        s = state()
        if blk is None:
            no_block_streak += 1
            if budget_exhausted(s):                       # only once the passes are spent is a reply without a block the final answer
                open(os.path.join(run_dir, "final_answer.md"), "w").write(text)
                print("final answer written; run complete", flush=True)
                return
            if no_block_streak >= 12:
                snapshot_partial(run_dir)
                print("twelve replies in a row without a code block while forward passes remain; stopping (the arm did not finish). "
                      "Transcripts in the turns folder; raw provider messages in raw_message.json.", flush=True)
                return
            nudge = ("No code block found, so nothing was run. Reply with exactly one ```python or ```forward block at the end of "
                     "your message (the session continues until all forward passes are used).")
            if no_block_streak >= 2:
                nudge += (" Your reply is executed only if it contains the fenced block itself (or calls the python/forward "
                          "function): write the code inside ```python ... ``` in the visible reply, not only a plan of it. "
                          f"({no_block_streak} replies without a block so far.)")
            messages.append({"role": "user", "content": nudge})
            continue
        no_block_streak = 0
        kind, body = blk
        if "EXPLAIN EVERY STEP" in task and len(text.split("```")[0].strip()) < 80:
            messages.append({"role": "user", "content": "Your block was not run: explain every step in plain English "
                             "before the block (3 to 8 sentences), then resend it."})
            continue
        name, out = run_block(kind, body)
        calls.write(json.dumps({"turn": turn, "kind": kind, "name": name, "code": body, "output": out}) + "\n")
        calls.flush()
        if out == SERVER_DEAD:
            snapshot_partial(run_dir)
            print("the forward-pass server is not running; stopping. Re-run run_compare.sh to resume this arm (notes and edits are kept).", flush=True)
            return
        if out == PARENT_GONE:
            snapshot_partial(run_dir)
            print("this editor's run (run_compare.sh) is gone; stopping so no orphan keeps driving the server. Re-run run_compare.sh to resume.", flush=True)
            return
        if kind == "forward":
            snapshot_partial(run_dir)      # after every pass the resumable snapshot is at most one pass old
        s = state()
        budget_line = f"\n\n[forward passes used: {s['passes_used']} of {s['budget'] if s.get('budget') is not None else '?'}]"
        nxt = f"OUTPUT of {name}:\n{out}{budget_line}"
        if budget_exhausted(s):
            nxt += ("\n\nAll forward passes are used. Make your final choice: the tensors in edited/ when you finish are "
                    "submitted. Reply without a code block, saying what you changed, the evidence for it and why you "
                    "expect it to help.")
        if since_handoff >= args.handoff_every:
            notes_p = os.path.join(WORK, "notes.md")
            notes = open(notes_p).read() if os.path.exists(notes_p) else ""
            messages = [{"role": "user", "content": handoff_message(task, notes) + "\n\n" + nxt}]
            since_handoff = 0
            print(f"[handoff at turn {turn}]", flush=True)
        else:
            messages.append({"role": "user", "content": nxt})


if __name__ == "__main__":
    main()
