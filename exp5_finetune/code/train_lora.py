"""Arm D, step 2: a LoRA fine-tune of Qwen3.5-0.8B on finetune/sft.jsonl, merged into a full copy at finetune/merged/.

    cd ~/qwen-0.8b/wsp && .venv/bin/pip install peft && .venv/bin/python finetune/train_lora.py        # ~20-60 min on an M-series Air
    WSP_MODEL_DIR=finetune/merged .venv/bin/python eval_heldout.py --arm finetune --weights original   # then evaluate it

The training text is exactly the inference format: the chat template with thinking off (prompt rendered with
add_generation_prompt=True) + the answer + <|im_end|>; loss on the answer tokens only. LoRA r=16, alpha=32 on the attention
projections (q/k/v/o of the six full-attention layers) and every MLP (gate/up/down), AdamW 1e-4, 3 epochs, 10% dev split.
If MPS fails in the backward pass, --device cpu works (slowly).
Writes finetune/train_stats.json (loss per epoch, wall time, trainable parameters).
"""
import argparse
import json
import math
import os
import random
import sys
import time

WSP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, WSP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=os.path.join(WSP, "finetune", "sft.jsonl"))
    ap.add_argument("--out", default=os.path.join(WSP, "finetune", "merged"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)      # the 248k-token vocabulary makes the logits large; the loss is computed in chunks
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    import kernel as K
    import server as S
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok, model = S.load_model(K.MODEL_DIR, device, dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.train()
    # module names differ by loader (AutoModelForCausalLM: model.layers.N.*; the multimodal class: model.language_model.layers.N.*),
    # so match by suffix and keep the vision tower out explicitly if one is present
    names = [n for n, _ in model.named_modules() if n.split(".")[-1] in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
             and "visual" not in n and "vision" not in n]
    assert names, "no attention/MLP projections found in the model"
    cfg = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, bias="none", target_modules=names)
    model = get_peft_model(model, cfg)
    print(f"LoRA on {len(names)} projections (e.g. {names[0]} ... {names[-1]})", flush=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_train:,}", flush=True)

    rows = [json.loads(l) for l in open(args.sft) if l.strip()]
    random.shuffle(rows)
    n_dev = max(1, len(rows) // 10)
    dev, train = rows[:n_dev], rows[n_dev:]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def encode(r):
        user, ans = r["messages"][0]["content"], r["messages"][1]["content"]
        prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(ans, add_special_tokens=False)["input_ids"] + [im_end]
        ids = (p_ids + a_ids)[:args.max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:args.max_len]
        return ids, labels

    train_enc, dev_enc = [encode(r) for r in train], [encode(r) for r in dev]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def batches(enc, bs, shuffle):
        order = list(range(len(enc)))
        if shuffle:
            random.shuffle(order)
        for i in range(0, len(order), bs):
            chunk = [enc[j] for j in order[i:i + bs]]
            L = max(len(c[0]) for c in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            lab = torch.full((len(chunk), L), -100, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for k, (a, b) in enumerate(chunk):
                ids[k, :len(a)] = torch.tensor(a)
                lab[k, :len(b)] = torch.tensor(b)
                att[k, :len(a)] = 1
            yield ids.to(device), att.to(device), lab.to(device)

    def free():
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    def chunked_loss(logits, lab, chunk=128):
        """mean cross-entropy over the answer tokens, computed 128 positions at a time so the float32 logits never exceed ~130 MB."""
        logits, lab = logits[:, :-1], lab[:, 1:]
        n = int((lab != -100).sum())
        tot = logits.new_zeros((), dtype=torch.float32)
        for j in range(0, logits.shape[1], chunk):
            lg = logits[:, j:j + chunk].float().reshape(-1, logits.shape[-1])
            tl = lab[:, j:j + chunk].reshape(-1)
            if int((tl != -100).sum()) == 0:
                continue
            tot = tot + torch.nn.functional.cross_entropy(lg, tl, ignore_index=-100, reduction="sum")
        return tot / max(n, 1)

    def dev_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for ids, att, lab in batches(dev_enc, args.batch, False):
                out = model(input_ids=ids, attention_mask=att)
                tot += float(chunked_loss(out.logits, lab).detach()) * ids.shape[0]
                n += ids.shape[0]
                del out
        free()
        model.train()
        return tot / max(n, 1)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_enc) / args.batch / args.accum)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total_steps // 20)) * 0.5 * (1 + math.cos(math.pi * s / max(1, total_steps))))
    t0 = time.time()
    stats = {"examples": len(train_enc), "dev_examples": len(dev_enc), "trainable_params": n_train, "epochs": [], "dev_loss_before": dev_loss(),
             "config": vars(args), "device": device}
    print(f"dev loss before: {stats['dev_loss_before']:.3f}", flush=True)
    step = 0
    for ep in range(args.epochs):
        tot, n, micro = 0.0, 0, 0
        for ids, att, lab in batches(train_enc, args.batch, True):
            out = model(input_ids=ids, attention_mask=att)
            loss = chunked_loss(out.logits, lab)
            (loss / args.accum).backward()
            tot += float(loss.detach()) * ids.shape[0]
            n += ids.shape[0]
            micro += 1
            del out, loss
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                free()
                step += 1
                if step % 5 == 0:
                    print(f"  epoch {ep + 1} step {step}/{total_steps} loss {tot / n:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if micro % args.accum:
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        dl = dev_loss()
        stats["epochs"].append({"epoch": ep + 1, "train_loss": tot / max(n, 1), "dev_loss": dl, "minutes": round((time.time() - t0) / 60, 1)})
        print(f"epoch {ep + 1}: train {tot / max(n, 1):.3f} dev {dl:.3f}", flush=True)
    model = model.merge_and_unload()
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    stats["minutes_total"] = round((time.time() - t0) / 60, 1)
    stats["merged_dir"] = args.out
    json.dump(stats, open(os.path.join(WSP, "finetune", "train_stats.json"), "w"), indent=1)
    print(f"merged model saved to {args.out} ({stats['minutes_total']} min)", flush=True)


if __name__ == "__main__":
    main()
