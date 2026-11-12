"""Weight-space programming server: the persistent Python kernel + the forward-pass tool, as in the blog's Experiment 8.

Run it natively on the Mac (Metal), from the folder next to the model files:
    cd ~/qwen-0.8b/wsp && .venv/bin/python server.py --budget 50

It loads Qwen3.5-0.8B once and then watches  requests/  for two kinds of files, written by the editor (Fable in the
chat, via the device shell, or editor_api.py):

  requests/code_<k>.py        one Python block -> run in the persistent kernel (namespace survives between blocks;
                              prelude: kernel.py's NAMES/load/load_original/save_edit/drop_edit/edited/tok, numpy, torch)
                              -> requests/code_<k>.out.txt   (stdout + the value of a trailing expression, cut at 10,000 chars)
  requests/fwd_<k>.json       one forward pass (JSON, see below) -> requests/fwd_<k>.done.json (the summary) and the
                              results under work/ ; a failed request does not count against the budget.

Forward-pass JSON, as the editor was told in the paper (this server's limits are in LIMITS below):
  {"type": "generate", "model": "original" | "edited" | "both", "problems": [practice ids], "prompts": [user messages]
   (optional, graded only if "answers" given), "n": samples per item, "max_tokens": int, "thinking": bool,
   "activations": {"layers": [..], "mlp": bool} (optional)}
  {"type": "teacher_forced", "model": "original" | "edited", "texts": [...] or "chats": [[{role, content}, ..], ..],
   "enable_thinking": bool, "return": {"token_logprobs": true, "topk": k, "hidden_states": [layers], "mlp_act": [layers],
   "attn_layers": [full-attention layers]}}

The model never sees the evaluation prompts; the editor never runs the model except through this tool. The kernel
namespace deliberately does not contain the loaded model.
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback

import numpy as np

WSP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WSP)
os.environ.setdefault("WSP_DIR", WSP)
import kernel as K  # noqa: E402

REQ, WORK = os.path.join(WSP, "requests"), os.path.join(WSP, "work")
STATE = os.path.join(WSP, "state.json")
PRACTICE = os.path.join(WSP, "practice")
MAX_OUT = 10_000
FULL_ATTN = [3, 7, 11, 15, 19, 23]
# this server's limits (a MacBook, not the paper's GPU cluster): stated to the editor in its task message
LIMITS = {"gen_max": 256, "gen_max_tokens": 4096, "tf_max_texts": 1024, "tf_max_tokens": 4096,
          "tf_total_tokens": 262_144, "act_total_tokens": 65_536, "attn_max_texts": 2, "attn_max_tokens": 2048}
# sampling: the model card's non-thinking text setting (the 0.8B runs non-thinking by default; thinking mode loops),
# and its thinking-mode setting when "thinking": true
SAMPLING = {False: {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 2.0},
            True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}}


# ----------------------------------------------------------------------------------------------------- model
def load_model(model_dir, device, dtype):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model, err = None, []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        try:
            cls = getattr(__import__("transformers"), cls_name)
            model = cls.from_pretrained(model_dir, dtype=dtype, attn_implementation="eager")
            print(f"loaded with {cls_name}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            err.append(f"{cls_name}: {type(e).__name__}: {str(e)[:200]}")
    if model is None:
        raise RuntimeError("could not load the model:\n" + "\n".join(err))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def find_layers(model):
    """The 24 decoder layers and the token embedding, wherever the HF class puts them."""
    import torch.nn as nn
    layers, emb = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) == 24 and name.endswith("layers") and "visual" not in name \
                and "mtp" not in name:
            layers = m
        if isinstance(m, nn.Embedding) and m.num_embeddings == 248320 and emb is None:
            emb = m
    assert layers is not None and emb is not None, "could not locate the decoder layers / embedding"
    return layers, emb


def param_map(model):
    """safetensors name -> parameter, trying the HF module's naming variants."""
    params = dict(model.named_parameters())
    out = {}
    for n in K.NAMES:
        cands = [n, n.replace("model.language_model.", "model.", 1), n.replace("model.language_model.", "", 1),
                 n.replace("model.language_model.", "language_model.", 1)]
        for c in cands:
            if c in params:
                out[n] = params[c]
                break
    missing = [n for n in K.NAMES if n not in out]
    assert len(missing) < 5, f"could not map tensors to the loaded model, e.g. {missing[:5]}"
    return out


class Weights:
    """Swap the editor's edits in and out of the loaded model."""

    def __init__(self, model):
        import torch
        self.torch = torch
        self.pm = param_map(model)
        self.current = "original"

    def apply(self, which):
        if which == self.current:
            return
        torch = self.torch
        names = K.edited() if which == "edited" else self._swapped
        for n in names:
            p = self.pm[n]
            src = K.load(n) if which == "edited" else K.load_original(n)
            p.data.copy_(torch.from_numpy(np.ascontiguousarray(src)).to(p.dtype).to(p.device))
        self._swapped = list(names)
        self.current = which

    _swapped = []


# ----------------------------------------------------------------------------------------------------- grading
def extract_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7:j - 1].strip() if depth == 0 else None


def norm_num(s):
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def grade(item, answer_text):
    """item: a practice record with "answer" and "grader"; returns (correct 0/1 or None, extracted)."""
    g = item.get("grader", "boxed")
    ans = item.get("answer")
    t = answer_text or ""
    if g == "none" or ans in (None, ""):
        return None, None
    if g == "boxed":
        ex = extract_boxed(t)
        if ex is None:
            return 0, None
        a, b = norm_num(ex), norm_num(str(ans))
        ok = (a is not None and b is not None and abs(a - b) < 1e-6) or ex.strip() == str(ans).strip()
        return int(ok), ex
    if g == "number":
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", t)
        ex = nums[-1] if nums else None
        return int(ex is not None and norm_num(ex) == norm_num(str(ans))), ex
    low = t.lower()
    if g == "contains_all":
        req = ans if isinstance(ans, list) else [ans]
        hits = [bool(re.search(r, low, re.I)) for r in req]
        return int(all(hits)), f"{sum(hits)}/{len(req)}"
    if g == "contains_any":
        req = ans if isinstance(ans, list) else [ans]
        return int(any(re.search(r, low, re.I) for r in req)), None
    if g == "must_not":
        req = ans if isinstance(ans, list) else [ans]
        return int(not any(re.search(r, low, re.I) for r in req)), None
    if g == "checks":   # ans = {"must": [regex, ...], "must_not": [regex, ...]}; all must present, none of must_not
        must = ans.get("must", [])
        must_not = ans.get("must_not", [])
        hit = [bool(re.search(r, low, re.I)) for r in must]
        bad = [r for r in must_not if re.search(r, low, re.I)]
        return int(all(hit) and not bad), f"must {sum(hit)}/{len(must)}; violated {bad}" if bad else f"must {sum(hit)}/{len(must)}"
    if g == "rubric":   # graded later by a judge (judge.py); counts as ungraded here
        return None, None
    raise ValueError(f"unknown grader {g}")


def loop_score(ids, n=8):
    """Share of 8-gram positions in the answer that repeat an earlier 8-gram (0 = no repetition)."""
    if len(ids) < 2 * n:
        return 0.0
    seen, rep = set(), 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        rep += g in seen
        seen.add(g)
    return rep / (len(ids) - n + 1)


# ----------------------------------------------------------------------------------------------------- practice
def load_eval_items():
    """The held-out sets, which live in eval/heldout/ so load_practice() cannot serve them to an editor.
    run_generate resolves ids from here too, so the EVALUATION (which calls run_generate in process)
    still works, while run_forward refuses ho_ ids that arrive in an editor request file."""
    items = {}
    base = os.environ.get("WSP_EVAL_DIR", os.path.join(WSP, "eval"))
    for f in sorted(glob.glob(os.path.join(base, "heldout", "*.jsonl"))):
        src = os.path.basename(f)[:-6]
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r.setdefault("source", src)
                items[r["id"]] = r
    return items


def load_practice():
    items = {}
    for f in sorted(glob.glob(os.path.join(PRACTICE, "*.jsonl"))):
        src = os.path.basename(f)[:-6]
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r.setdefault("source", src)
                items[r["id"]] = r
    return items


# ----------------------------------------------------------------------------------------------------- server
class Server:
    def __init__(self, args):
        import torch
        self.torch = torch
        self.args = args
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[args.dtype]
        lock = os.path.join(WORK, "server.pid")                      # one server per folder: a second one would share state.json and the requests
        if os.path.exists(lock):
            try:
                other = int(open(lock).read().strip())
                os.kill(other, 0)
                if other != os.getpid():
                    raise SystemExit(f"another server (pid {other}) is already running in this folder; stop it first "
                                     f"(pkill -f 'server.py --budget'), or delete work/server.pid if that pid is not a server")
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        os.makedirs(WORK, exist_ok=True)
        open(lock, "w").write(str(os.getpid()))
        model_dir = os.environ.get("WSP_MODEL_DIR", K.MODEL_DIR)     # a fine-tuned copy for the evaluation (arm D)
        print(f"loading {model_dir} on {self.device} ({args.dtype}) ...", flush=True)
        self.tok, self.model = load_model(model_dir, self.device, dtype)
        self.layers, self.emb = find_layers(self.model)
        self.weights = Weights(self.model)
        self.practice = load_practice()
        self.state = json.load(open(STATE)) if os.path.exists(STATE) else {"budget": args.budget, "passes_used": 0,
                                                                            "code_runs": 0, "passes": []}
        self.state["budget"] = args.budget
        self.ns = None
        print(f"ready: {len(self.practice)} practice items, budget {args.budget}, "
              f"passes used {self.state['passes_used']}, edited {K.edited()}", flush=True)
        self.save_state()      # the counter file exists from the first second, so an editor's first reply cannot read a missing budget

    def save_state(self):
        json.dump(self.state, open(STATE, "w"), indent=1)

    # ------------------------------------------------------------------ kernel
    def kernel_ns(self):
        if self.ns is None:
            self.ns = {"__name__": "__kernel__"}
            exec("import os, json, math, re, time\nimport numpy as np\nimport torch\ntorch.set_grad_enabled(False)\n"
                 "np.set_printoptions(precision=4, suppress=True, linewidth=160)\n"
                 "from kernel import *\n", self.ns)
        return self.ns

    def run_code(self, path):
        code = open(path).read()
        ns = self.kernel_ns()
        buf = io.StringIO()
        t0 = time.time()
        limit = int(os.environ.get("WSP_CODE_SECONDS", "120"))      # a cell that runs longer is interrupted; the editor sees the error

        def _alarm(signum, frame):
            raise TimeoutError(f"cell interrupted after {limit} s (WSP_CODE_SECONDS); split the work, avoid recursive globs over large trees")
        import signal
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(limit)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                import ast
                tree = ast.parse(code)
                last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                if last is not None:
                    tree.body = tree.body[:-1]
                exec(compile(tree, "<code>", "exec"), ns)
                if last is not None:
                    val = eval(compile(ast.Expression(last.value), "<code>", "eval"), ns)
                    if val is not None:
                        print(repr(val))
            except BaseException:  # noqa: BLE001  (TimeoutError included)
                traceback.print_exc(file=buf)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        out = buf.getvalue()
        if len(out) > MAX_OUT:
            out = out[:MAX_OUT] + f"\n... [output cut at {MAX_OUT} characters]"
        out += f"\n[{time.time() - t0:.1f} s]"
        self.state["code_runs"] += 1
        self.save_state()
        return out

    # ------------------------------------------------------------------ rendering
    def render(self, user, thinking, system=None):
        chat = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking))

    def render_chat(self, chat, thinking):
        return self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False,
                                            enable_thinking=bool(thinking))

    # ------------------------------------------------------------------ generation
    def generate_texts(self, prompts, seeds, max_new, thinking, batch):
        """prompts: rendered strings; returns list of (ids, text, cut)."""
        torch = self.torch
        s = SAMPLING[bool(thinking)]
        out = []
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        from transformers import LogitsProcessor, LogitsProcessorList

        class Presence(LogitsProcessor):
            def __init__(self, pen, start):
                self.pen, self.start = pen, start

            def __call__(self, input_ids, scores):
                if self.pen == 0:
                    return scores
                gen = input_ids[:, self.start:]
                if gen.shape[1] == 0:
                    return scores
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask.scatter_(1, gen, True)          # tokens already generated (presence penalty, as in vLLM)
                return scores - self.pen * mask.to(scores.dtype)

        for b in range(0, len(prompts), batch):
            chunk = prompts[b:b + batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            torch.manual_seed(int(seeds[b]) & 0x7FFFFFFF)
            with torch.no_grad():
                gen = self.model.generate(**enc, do_sample=True, temperature=s["temperature"], top_p=s["top_p"],
                                          top_k=s["top_k"], max_new_tokens=max_new,
                                          logits_processor=LogitsProcessorList([Presence(s["presence_penalty"],
                                                                                         enc["input_ids"].shape[1])]),
                                          pad_token_id=self.tok.pad_token_id)
            L = enc["input_ids"].shape[1]
            for i in range(len(chunk)):
                ids = gen[i, L:].tolist()
                eos = [j for j, t in enumerate(ids) if t in (self.tok.eos_token_id, self.tok.pad_token_id)]
                cut = not eos
                ids = ids[:eos[0]] if eos else ids
                out.append((ids, self.tok.decode(ids, skip_special_tokens=False), cut))
            print(f"  generated {min(b + batch, len(prompts))}/{len(prompts)}", flush=True)
        return out

    def split_answer(self, text):
        if "</think>" in text:
            th, ans = text.split("</think>", 1)
            return th.replace("<think>", "").strip(), ans.strip()
        return "", text.strip()

    def run_generate(self, req, k):
        thinking = bool(req.get("thinking", False))
        n = int(req.get("n", 1))
        max_new = min(int(req.get("max_tokens", 1024)), LIMITS["gen_max_tokens"])
        which = req.get("model", "original")
        items = []
        for pid in req.get("problems", []):
            if pid not in self.practice and pid not in getattr(self, "eval_items", {}):
                self.eval_items = load_eval_items()          # evaluation sets, loaded on first use
            src = self.practice if pid in self.practice else getattr(self, "eval_items", {})
            assert pid in src, f"unknown practice id {pid}"
            items.append(dict(src[pid]))
        for i, p in enumerate(req.get("prompts", [])):
            items.append({"id": f"prompt_{k}_{i}", "source": "prompt", "problem": p,
                          "answer": (req.get("answers") or [None] * 10 ** 6)[i], "grader": req.get("grader", "none")})
        assert items, "no problems or prompts"
        per_model = len(items) * n
        cap = LIMITS["gen_max"] // (2 if which == "both" else 1)
        assert per_model <= cap, f"{per_model} generations per model > {cap} allowed"
        models = ["original", "edited"] if which == "both" else [which]
        if "edited" in models and not K.edited():
            raise ValueError("no edits saved; 'edited' equals 'original'")
        system = req.get("system") or None          # context injection (arm C): a system prompt in front of every item
        prompts, seeds, meta = [], [], []
        for it in items:
            for s in range(n):
                prompts.append(self.render(it["problem"], thinking, system))
                seeds.append(int(hashlib.sha256(f"{it['id']}|{s}".encode()).hexdigest()[:8], 16))
                meta.append((it, s))
        results = {}
        for m in models:
            self.weights.apply(m)
            print(f"[fwd_{k}] generating {len(prompts)} answers with {m} weights (thinking={thinking}, "
                  f"max_tokens={max_new})", flush=True)
            gens = self.generate_texts(prompts, seeds, max_new, thinking, self.args.batch)
            rows = []
            for (it, s), (ids, text, cut) in zip(meta, gens):
                th, ans = self.split_answer(text)
                correct, ex = grade(it, ans if not cut or ans else text)
                rows.append({"id": it["id"], "source": it["source"], "sample": s, "problem": it["problem"],
                             "think": th, "answer": ans, "extracted": ex, "correct": correct, "n_tokens": len(ids),
                             "cut": cut, "loop_score": round(loop_score(ids), 3), "reference": it.get("answer")})
            suffix = f"_{m}" if which == "both" else ""
            with open(os.path.join(WORK, f"gen_{k}{suffix}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            results[m] = rows
            if req.get("activations"):
                self.save_activations(rows, prompts, req["activations"], os.path.join(WORK, f"gen_{k}_acts{suffix}"),
                                      thinking)
        self.weights.apply("original")
        summ = {"type": "generate", "models": models, "generations_per_model": len(prompts), "thinking": thinking,
                "max_tokens": max_new, "system_prompt_chars": len(system or ""), "files": sorted(os.path.basename(p) for p in glob.glob(f"{WORK}/gen_{k}*"))}
        for m, rows in results.items():
            summ[m] = self.summarize(rows)
        if which == "both":
            a = np.array([r["correct"] if r["correct"] is not None else np.nan for r in results["original"]], float)
            b = np.array([r["correct"] if r["correct"] is not None else np.nan for r in results["edited"]], float)
            ok = ~np.isnan(a) & ~np.isnan(b)
            d = b[ok] - a[ok]
            summ["paired"] = {"graded": int(ok.sum()), "acc_diff_edited_minus_original": float(d.mean()) if ok.any() else None,
                              "se": float(d.std(ddof=1) / np.sqrt(len(d))) if ok.sum() > 1 else None,
                              "wrong_to_right": int(((a[ok] == 0) & (b[ok] == 1)).sum()),
                              "right_to_wrong": int(((a[ok] == 1) & (b[ok] == 0)).sum()),
                              "cut_share_diff": float(np.mean([r["cut"] for r in results["edited"]]) -
                                                      np.mean([r["cut"] for r in results["original"]]))}
        return summ

    def summarize(self, rows):
        graded = [r for r in rows if r["correct"] is not None]
        by = {}
        for r in graded:
            by.setdefault(r["source"], []).append(r["correct"])
        return {"accuracy": float(np.mean([r["correct"] for r in graded])) if graded else None,
                "graded": len(graded), "by_source": {s: float(np.mean(v)) for s, v in by.items()},
                "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
                "cut_share": float(np.mean([r["cut"] for r in rows])),
                "mean_loop_score": float(np.mean([r["loop_score"] for r in rows])),
                "failures": {"cut_at_max_tokens": int(sum(r["cut"] for r in rows)),
                             "no_answer_extracted": int(sum(1 for r in graded if r["extracted"] is None and r["correct"] == 0)),
                             "wrong": int(sum(1 for r in graded if r["correct"] == 0 and r["extracted"] is not None)),
                             "looping(>0.3)": int(sum(r["loop_score"] > 0.3 for r in rows))}}

    # ------------------------------------------------------------------ activations / teacher forcing
    def hooks(self, hidden_layers, mlp_layers, store, mlpin_layers=()):
        """hidden l: the residual stream ENTERING layers[l] (0 = embeddings, l = after the first l layers);
        mlp l: the down_proj input of layers[l] (post-activation); mlpin l: the MLP INPUT of layers[l]
        (post_attention_layernorm output), which is what a gate_proj/up_proj row actually sees."""
        hs = []
        if 0 in hidden_layers:
            hs.append(self.emb.register_forward_hook(lambda m, i, o: store.setdefault("hidden", {}).__setitem__(0, o)))
        for l in hidden_layers:
            if l >= 1:
                hs.append(self.layers[l - 1].register_forward_hook(
                    lambda m, i, o, l=l: store.setdefault("hidden", {}).__setitem__(l, o[0] if isinstance(o, tuple) else o)))
        for l in mlp_layers:
            hs.append(self.layers[l].mlp.down_proj.register_forward_pre_hook(
                lambda m, i, l=l: store.setdefault("mlp", {}).__setitem__(l, i[0])))
        for l in mlpin_layers:
            hs.append(self.layers[l].mlp.register_forward_pre_hook(
                lambda m, i, l=l: store.setdefault("mlpin", {}).__setitem__(l, i[0])))
        return hs

    def save_activations(self, rows, prompts, spec, outdir, thinking):
        """Residual stream at the last prompt token and averaged over the answer (first act_total_tokens / n tokens)."""
        torch = self.torch
        os.makedirs(outdir, exist_ok=True)
        layers = [int(l) for l in spec.get("layers", [])]
        mlp = bool(spec.get("mlp", False))
        per = max(256, LIMITS["act_total_tokens"] // max(1, len(rows)))
        index = []
        last = {l: [] for l in layers}
        mean = {l: [] for l in layers}
        mlp_mean = {l: [] for l in range(24)} if mlp else {}
        for i, (r, p) in enumerate(zip(rows, prompts)):
            full = p + (("<think>\n" + r["think"] + "\n</think>\n\n" if r["think"] else "") + r["answer"])
            enc = self.tok(full, return_tensors="pt").to(self.device)
            L = len(self.tok(p)["input_ids"])
            ids = enc["input_ids"][:, :L + per]
            store = {}
            hs = self.hooks(layers, list(range(24)) if mlp else [], store)
            with torch.no_grad():
                self.model(input_ids=ids)
            for h in hs:
                h.remove()
            for l in layers:
                a = store["hidden"][l][0].float()
                last[l].append(a[L - 1].cpu().numpy())
                mean[l].append(a[L:].mean(0).cpu().numpy() if a.shape[0] > L else a[L - 1].cpu().numpy())
            for l in mlp_mean:
                a = store["mlp"][l][0].float()
                mlp_mean[l].append(a[L:].mean(0).cpu().numpy() if a.shape[0] > L else a[L - 1].cpu().numpy())
            index.append({"i": i, "id": r["id"], "sample": r["sample"], "correct": r["correct"], "cut": r["cut"],
                          "answer_tokens_used": int(ids.shape[1] - L)})
        for l in layers:
            np.save(os.path.join(outdir, f"hidden_L{l}_lastprompt.npy"), np.stack(last[l]).astype(np.float16))
            np.save(os.path.join(outdir, f"hidden_L{l}_answermean.npy"), np.stack(mean[l]).astype(np.float16))
        for l in mlp_mean:
            np.save(os.path.join(outdir, f"mlp_L{l}_answermean.npy"), np.stack(mlp_mean[l]).astype(np.float16))
        json.dump({"samples": index, "files": sorted(os.listdir(outdir))}, open(os.path.join(outdir, "index.json"), "w"))

    def run_teacher_forced(self, req, k):
        torch = self.torch
        which = req.get("model", "original")
        if which == "edited" and not K.edited():
            raise ValueError("no edits saved; 'edited' equals 'original'")
        thinking = bool(req.get("enable_thinking", False))
        texts = list(req.get("texts", []))
        for chat in req.get("chats", []):
            texts.append(self.render_chat(chat, thinking))
        assert texts, "no texts or chats"
        assert len(texts) <= LIMITS["tf_max_texts"], f"{len(texts)} texts > {LIMITS['tf_max_texts']}"
        ret = req.get("return", {"token_logprobs": True})
        hidden = [int(l) for l in ret.get("hidden_states", [])]
        mlp = [int(l) for l in ret.get("mlp_act", [])]
        mlpin = [int(l) for l in ret.get("mlp_in", [])]
        attn = [int(l) for l in ret.get("attn_layers", [])]
        topk = int(ret.get("topk", 0))
        for l in attn:
            assert l in FULL_ATTN, f"attention is only available for the full-attention layers {FULL_ATTN}"
        enc_all = [self.tok(t)["input_ids"][:LIMITS["tf_max_tokens"]] for t in texts]
        total = sum(len(e) for e in enc_all)
        assert total <= LIMITS["tf_total_tokens"], f"{total} tokens > {LIMITS['tf_total_tokens']}"
        if hidden or mlp or mlpin:
            assert total <= LIMITS["act_total_tokens"], \
                f"{total} tokens with hidden/mlp activations > {LIMITS['act_total_tokens']}"
        if attn:
            assert len(texts) <= LIMITS["attn_max_texts"] and max(len(e) for e in enc_all) <= LIMITS["attn_max_tokens"]
        outdir = os.path.join(WORK, f"fwd_{k}")
        os.makedirs(outdir, exist_ok=True)
        self.weights.apply(which)
        T = max(len(e) for e in enc_all)
        N = len(enc_all)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        input_ids = np.full((N, T), pad, dtype=np.int64)
        lengths = np.array([len(e) for e in enc_all])
        for i, e in enumerate(enc_all):
            input_ids[i, :len(e)] = e
        lp = np.zeros((N, T), np.float32)
        topk_ids = np.zeros((N, T, topk), np.int64) if topk else None
        topk_lp = np.zeros((N, T, topk), np.float32) if topk else None
        H = {l: np.zeros((N, T, 1024), np.float16) for l in hidden}
        M = {l: np.zeros((N, T, 3584), np.float16) for l in mlp}
        MI = {l: np.zeros((N, T, 1024), np.float16) for l in mlpin}
        A = {}
        print(f"[fwd_{k}] teacher-forced read of {N} texts, {total} tokens, {which} weights", flush=True)
        for i in range(N):
            ids = torch.tensor(input_ids[i:i + 1, :lengths[i]], device=self.device)
            store = {}
            hs = self.hooks(hidden, mlp, store, mlpin)
            with torch.no_grad():
                out = self.model(input_ids=ids, output_attentions=bool(attn))
            for h in hs:
                h.remove()
            for l in mlpin:
                MI[l][i, :lengths[i]] = store["mlpin"][l][0].float().cpu().numpy().astype(np.float16)
            logits = out.logits[0].float()
            logp = torch.log_softmax(logits, -1)
            nxt = ids[0, 1:]
            lp[i, :lengths[i] - 1] = logp[:-1].gather(1, nxt[:, None])[:, 0].cpu().numpy()
            if topk:
                v, ix = logp.topk(topk, -1)
                topk_ids[i, :lengths[i]] = ix.cpu().numpy()
                topk_lp[i, :lengths[i]] = v.cpu().numpy()
            for l in hidden:
                H[l][i, :lengths[i]] = store["hidden"][l][0].float().cpu().numpy().astype(np.float16)
            for l in mlp:
                M[l][i, :lengths[i]] = store["mlp"][l][0].float().cpu().numpy().astype(np.float16)
            if attn and out.attentions is not None:
                for l in attn:
                    a = out.attentions[l]
                    if a is not None:
                        A.setdefault(l, []).append(a[0].float().cpu().numpy().astype(np.float16))
            if (i + 1) % 32 == 0:
                print(f"  {i + 1}/{N}", flush=True)
        self.weights.apply("original")
        np.save(os.path.join(outdir, "input_ids.npy"), input_ids)
        np.save(os.path.join(outdir, "lengths.npy"), lengths)
        np.save(os.path.join(outdir, "token_logprobs.npy"), lp)
        if topk:
            np.save(os.path.join(outdir, "topk_ids.npy"), topk_ids)
            np.save(os.path.join(outdir, "topk_logprobs.npy"), topk_lp)
        for l in hidden:
            np.save(os.path.join(outdir, f"hidden_L{l}.npy"), H[l])
        for l in mlp:
            np.save(os.path.join(outdir, f"mlp_L{l}.npy"), M[l])
        for l in mlpin:
            np.save(os.path.join(outdir, f"mlpin_L{l}.npy"), MI[l])
        for l, arrs in A.items():
            for i, a in enumerate(arrs):
                np.save(os.path.join(outdir, f"attn_L{l}_text{i}.npy"), a)
        meta = {"model": which, "n_texts": N, "max_len": int(T), "total_tokens": int(total),
                "mean_token_logprob": float(np.mean([lp[i, :lengths[i] - 1].mean() for i in range(N)])),
                "files": sorted(os.listdir(outdir)),
                "note": "token_logprobs[i, t] is the log-probability of token t+1 given tokens 0..t; right-padded"}
        json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=1)
        return {"type": "teacher_forced", **meta}

    # ------------------------------------------------------------------ loop
    def run_forward(self, path):
        k = re.search(r"fwd_(\d+)\.json$", path).group(1)
        req = json.load(open(path))
        self.practice = load_practice()          # practice files may be added while the server runs
        held = [str(i) for i in (req.get("problems") or []) if str(i).startswith("ho_")]
        if held:                                 # the evaluation calls run_generate directly; editor requests may not
            return {"error": f"held-out ids are not available to the editor ({len(held)} requested, e.g. {held[0]}); "
                             "use the practice ids", "counted": False, "passes_used": self.state["passes_used"]}
        left = self.state["budget"] - self.state["passes_used"]
        if left <= 0:
            return {"error": "no forward passes left", "passes_used": self.state["passes_used"]}
        t0 = time.time()
        try:
            summ = self.run_generate(req, k) if req.get("type") == "generate" else self.run_teacher_forced(req, k)
        except Exception as e:  # noqa: BLE001
            self.weights.apply("original")
            return {"error": f"{type(e).__name__}: {e}", "counted": False, "passes_used": self.state["passes_used"],
                    "traceback": traceback.format_exc()[-2000:]}
        self.state["passes_used"] += 1
        self.state["passes"].append({"k": k, "type": req.get("type"), "model": req.get("model"),
                                     "seconds": round(time.time() - t0), "edited_at_time": K.edited()})
        self.save_state()
        summ.update({"seconds": round(time.time() - t0), "passes_used": self.state["passes_used"],
                     "passes_left": self.state["budget"] - self.state["passes_used"]})
        return summ

    def serve(self):
        os.makedirs(REQ, exist_ok=True)
        os.makedirs(WORK, exist_ok=True)
        print("watching", REQ, flush=True)
        while True:
            pending = sorted((p for p in glob.glob(os.path.join(REQ, "*"))
                              if re.fullmatch(r"(code_\d+\.py|fwd_\d+\.json)", os.path.basename(p))),
                             key=os.path.getmtime)
            did = False
            for p in pending:
                out = p[:-3] + ".out.txt" if p.endswith(".py") else p[:-5] + ".done.json"
                if os.path.exists(out) or os.path.exists(p + ".claimed"):
                    continue
                if time.time() - os.path.getmtime(p) < 0.5:
                    continue
                open(p + ".claimed", "w").write(str(time.time()))
                print(f"-> {os.path.basename(p)}", flush=True)
                try:
                    if p.endswith(".py"):
                        res = self.run_code(p)
                        open(out, "w").write(res)
                    else:
                        res = self.run_forward(p)
                        json.dump(res, open(out, "w"), indent=1)
                except Exception:  # noqa: BLE001
                    open(out, "w").write("SERVER ERROR\n" + traceback.format_exc())
                print(f"<- {os.path.basename(out)}", flush=True)
                did = True
            if not did:
                time.sleep(1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f16", "f32"])
    ap.add_argument("--batch", type=int, default=8)
    Server(ap.parse_args()).serve()
