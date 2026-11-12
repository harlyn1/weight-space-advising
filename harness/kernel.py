"""The editor's kernel over the weights of Qwen3.5-0.8B - the prelude the blog's Experiment 8 gave the editor.

Defined here, exactly as in the published harness (lm_understand-logs/exp8_math, message.txt):
    NAMES                 all language-model tensor names (model.language_model.*; the vision tower and MTP head are
                          not part of the edited model and are hidden)
    SHAPES, DTYPES        shape and stored dtype of every tensor
    load(name)            float32 numpy array: your current edit if there is one, else the original
    load_original(name)   float32 numpy array of the original
    save_edit(name, arr)  change a tensor (refused: the frozen tensors, a wrong shape, non-finite values, and an edit
                          that only multiplies the whole tensor by a constant)
    drop_edit(name)       remove an edit
    edited()              names of the edited tensors
    tok                   the tokenizer (tokenizers.Tokenizer), or None if the package is missing
    ARCH                  a short description of the architecture (layer types, heads, widths)

Edits are stored as float32 .npy files in <wsp>/edited/<name>.npy; the forward-pass server casts them to bfloat16
and swaps them into the model for "edited" runs, so what you measure is what the evaluation sees.

Works with numpy alone (reading the safetensors file by memory map), so it runs in a small VM without torch;
import it from the server (native macOS) or run analysis scripts in the VM:
    import sys; sys.path.insert(0, "<wsp>"); from kernel import *
"""
import json
import os
import struct

import numpy as np

WSP = os.environ.get("WSP_DIR", os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("WSP_MODEL", os.path.dirname(WSP))          # the Hugging Face folder of Qwen3.5-0.8B
EDIT_DIR = os.path.join(WSP, "edited")
WORK = os.path.join(WSP, "work")

FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")
PREFIX = "model.language_model."

ARCH = ("Qwen3.5-0.8B, text part: 24 layers of width 1,024; layers 3, 7, 11, 15, 19 and 23 are full-attention layers "
        "with 8 query heads and 2 key/value heads of size 256 and a gated output (o_proj input 2048 = 8 heads x 256), "
        "the other 18 are Gated DeltaNet linear-attention layers with 16 heads of size 128; each layer has a SwiGLU MLP "
        "of width 3,584 (gate_proj, up_proj: 3584 x 1024; down_proj: 1024 x 3584); RMSNorm; vocabulary 248,320; input "
        "embedding and output layer tied (lm_head = embed_tokens).")

_DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64, "I64": np.int64, "I32": np.int32,
       "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def _safetensors_files():
    fs = sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors"))
    assert fs, f"no .safetensors in {MODEL_DIR}"
    return [os.path.join(MODEL_DIR, f) for f in fs]


def _read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


_INDEX = {}      # name -> (path, dtype, shape, start, end)
for _p in _safetensors_files():
    _h, _off = _read_header(_p)
    for _k, _v in _h.items():
        if _k == "__metadata__":
            continue
        _a, _b = _v["data_offsets"]
        _INDEX[_k] = (_p, _v["dtype"], tuple(_v["shape"]), _off + _a, _off + _b)

NAMES = sorted(k for k in _INDEX if k.startswith(PREFIX))
SHAPES = {k: _INDEX[k][2] for k in NAMES}
DTYPES = {k: _INDEX[k][1] for k in NAMES}
_MMAP = {}


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16_bits(x32):
    """float32 -> bfloat16 bits, round to nearest even (matches torch .to(bfloat16) for finite values)."""
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def _raw(name):
    path, dt, shape, s, e = _INDEX[name]
    if path not in _MMAP:
        _MMAP[path] = np.memmap(path, dtype=np.uint8, mode="r")
    buf = _MMAP[path][s:e]
    return np.frombuffer(buf, dtype=_DT[dt]).reshape(shape), dt


def load_original(name):
    """float32 numpy array of the original tensor (a copy; safe to modify)."""
    if name == "lm_head.weight":
        name = PREFIX + "embed_tokens.weight"
    assert name in _INDEX, f"unknown tensor {name}"
    arr, dt = _raw(name)
    return bf16_to_f32(arr) if dt == "BF16" else arr.astype(np.float32)


def _edit_path(name):
    return os.path.join(EDIT_DIR, name + ".npy")


def load(name):
    """float32 numpy array: your current edit if there is one, else the original."""
    p = _edit_path(name)
    return np.load(p).astype(np.float32) if os.path.exists(p) else load_original(name)


def edited():
    if not os.path.isdir(EDIT_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(EDIT_DIR) if f.endswith(".npy"))


def save_edit(name, arr):
    """Change a tensor. Refused: frozen tensors, wrong shape, non-finite values, a pure rescale of the whole tensor."""
    if name in FROZEN:
        raise ValueError(f"refused: {name} is out of scope (final norm, embeddings and output layer are frozen)")
    assert name in SHAPES, f"unknown tensor {name}"
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().float().numpy()
    except ImportError:
        pass
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    assert arr.shape == SHAPES[name], f"{name}: shape {arr.shape}, expected {SHAPES[name]}"
    if not np.isfinite(arr).all():
        raise ValueError(f"refused: {name} has non-finite values")
    orig = load_original(name)
    d = arr - orig
    if not d.any():
        drop_edit(name)
        return "no change (identical to the original); edit dropped"
    # a pure rescale: arr ~= c * orig for one scalar c
    denom = float((orig * orig).sum())
    if denom > 0:
        c = float((arr * orig).sum() / denom)
        resid = arr - c * orig
        if float(np.abs(resid).max()) <= 1e-6 * (1.0 + float(np.abs(arr).max())):
            raise ValueError(f"refused: {name} edit only multiplies the whole tensor by {c:.4g}")
    os.makedirs(EDIT_DIR, exist_ok=True)
    np.save(_edit_path(name), arr)
    rel = float(np.linalg.norm(d) / (np.linalg.norm(orig) + 1e-12))
    nz = int((bf16_bits(arr) != bf16_bits(orig)).sum())
    return f"saved {name}: relative change {rel:.4g}, {nz:,} bfloat16 entries differ"


def drop_edit(name):
    p = _edit_path(name)
    if not os.path.exists(p):
        return f"no edit for {name}"
    try:
        os.remove(p)
    except PermissionError:  # the device shell may not delete inside a connected folder: move it aside instead
        import time
        d = os.path.join(WSP, "dropped")
        os.makedirs(d, exist_ok=True)
        os.replace(p, os.path.join(d, f"{name}.{int(time.time())}.npy"))
    return f"dropped {name}"


def layer_names(i):
    return [n for n in NAMES if f".layers.{i}." in n]


def _tokenizer():
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    except Exception:
        return None


tok = _tokenizer()
os.makedirs(WORK, exist_ok=True)

__all__ = ["NAMES", "SHAPES", "DTYPES", "ARCH", "FROZEN", "WORK", "EDIT_DIR", "MODEL_DIR", "load", "load_original",
           "save_edit", "drop_edit", "edited", "layer_names", "tok", "bf16_bits", "bf16_to_f32", "np"]

if __name__ == "__main__":
    print(f"{len(NAMES)} language-model tensors in {MODEL_DIR}")
    tot = sum(int(np.prod(SHAPES[n])) for n in NAMES)
    print(f"{tot:,} weights; edited: {edited()}; tokenizer: {'ok' if tok else 'missing (pip install tokenizers)'}")
    for n in NAMES[:3] + layer_names(15):
        print(" ", n, SHAPES[n], DTYPES[n])
