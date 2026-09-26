import argparse
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def render(tok, sample, order="task-first"):
    usr = (f"{sample['task']} {sample['user_message']}".strip()
           if order == "task-first"
           else f"{sample['user_message']} {sample['task']}".strip())
    text = tok.apply_chat_template(
        [{"role": "system", "content": sample["system_message"]},
         {"role": "user", "content": usr}],
        tokenize=False, add_generation_prompt=True)
    return text, sample["system_message"]


def sys_mask_from_offsets(tok, text, sys_text):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    lo = text.find(sys_text)
    if lo < 0:
        return enc["input_ids"], [False] * len(enc["input_ids"])
    hi = lo + len(sys_text)
    return enc["input_ids"], [(b > a and a >= lo and b <= hi)
                              for a, b in enc["offset_mapping"]]


class ConflictDS(Dataset):

    def __init__(self, files, tok, order="task-first", limit=0,
                 targets=None, max_len=1024, max_target_tokens=448):
        self.items = []
        tmap = {}
        if targets:
            for r in json.load(open(targets)):
                if r.get("verified"):
                    tmap[(r["file"], r["id"])] = r["target"]
        self.n_with_target = 0
        for f in files:
            base = os.path.basename(f)
            conf = [s for s in json.load(open(f)) if s["label"] == "conflict"]
            for s in (conf[:limit] if limit else conf):
                text, sys_text = render(tok, s, order)
                ids, mask = sys_mask_from_offsets(tok, text, sys_text)
                if not any(mask):
                    continue
                ids, mask = ids[:max_len], mask[:max_len]
                focus_idx = len(ids) - 1
                labels = [-100] * len(ids)
                tgt = tmap.get((base, s["id"]))
                if targets is not None and tgt is None:
                    continue
                if tgt is not None:
                    tid = tok(tgt + (tok.eos_token or ""),
                              add_special_tokens=False)["input_ids"][:max_target_tokens]
                    ids = ids + tid
                    mask = mask + [False] * len(tid)
                    labels = labels + tid
                    self.n_with_target += 1
                self.items.append((ids, mask, labels, focus_idx))
        if not self.items:
            raise ValueError("no usable training samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, tok):
    L = max(len(ids) for ids, _, _, _ in batch)
    pad = tok.pad_token_id
    ii, am, sm, lb, fi = [], [], [], [], []
    for ids, m, lab, f in batch:
        k = L - len(ids)
        ii.append(list(ids) + [pad] * k)
        am.append([1] * len(ids) + [0] * k)
        sm.append(list(m) + [False] * k)
        lb.append(list(lab) + [-100] * k)
        fi.append(f)
    return {"input_ids": torch.tensor(ii),
            "attention_mask": torch.tensor(am),
            "labels": torch.tensor(lb),
            "sys_mask": torch.tensor(sm),
            "focus_idx": torch.tensor(fi)}


def _minmax(d):
    v = np.array(list(d.values()), dtype=np.float64)
    lo, hi = v.min(), v.max()
    rng = (hi - lo) if hi > lo else 1.0
    return {k: (x - lo) / rng for k, x in d.items()}


def score_heads(normal, conflict, mode="paper", a=0.4, b=0.3, g=0.3):
    mag, dir_, dist, repo = {}, {}, {}, {}
    for k in normal:
        if k not in conflict:
            continue
        T = min(len(x) for x in normal[k] + conflict[k])
        n = np.stack([r[:T] for r in normal[k]])
        c = np.stack([r[:T] for r in conflict[k]])
        m = min(n.shape[0], c.shape[0])
        n, c = n[:m], c[:m]
        if n.size == 0:
            continue
        mag[k] = float(np.abs(n - c).sum() / m)
        nf, cf = n.ravel(), c.ravel()
        dir_[k] = float(1.0 - float(nf @ cf) /
                        (np.linalg.norm(nf) * np.linalg.norm(cf) + 1e-12))
        p = n / np.clip(n.sum(-1, keepdims=True), 1e-12, None)
        q = c / np.clip(c.sum(-1, keepdims=True), 1e-12, None)
        kl = (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1) \
           + (q * (np.log(q + 1e-12) - np.log(p + 1e-12))).sum(-1)
        dist[k] = float(kl.mean())
        frob = float(np.linalg.norm(n - c))
        shift = float(np.mean(np.abs(n.mean(1) - c.mean(1))))
        e = np.exp(n - n.max(-1, keepdims=True)); e /= e.sum(-1, keepdims=True)
        fq = np.exp(c - c.max(-1, keepdims=True)); fq /= fq.sum(-1, keepdims=True)
        repo[k] = 0.4 * frob + 0.3 * shift + 0.3 * float(
            (e * (np.log(e + 1e-6) - np.log(fq + 1e-6))).sum() / e.shape[0])
    if mode == "repo":
        return repo
    M, D, G = _minmax(mag), _minmax(dir_), _minmax(dist)
    return {k: a * M[k] + b * D[k] + g * G[k] for k in M}


@torch.no_grad()
def last_row_attention(model, tok, sample, order):
    text, _ = render(tok, sample, order)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model(**enc, output_attentions=True)
    last = enc["input_ids"].shape[1] - 1
    rows = {}
    for l, A in enumerate(out.attentions):
        r = A[0, :, last, :].float().cpu().numpy()
        for h in range(r.shape[0]):
            rows[f"L{l}_H{h}"] = r[h]
    return rows


def detect_heads(files, model, tok, topk, order, mode, a, b, g, probe_per_file):
    normal, conflict = defaultdict(list), defaultdict(list)
    for f in files:
        grp = defaultdict(dict)
        for s in json.load(open(f)):
            grp[s["id"].replace("_normal", "").replace("_conflict", "")][s["label"]] = s
        for bid in list(grp)[:probe_per_file]:
            for lab, bucket in (("normal", normal), ("conflict", conflict)):
                if lab in grp[bid]:
                    for k, v in last_row_attention(model, tok, grp[bid][lab], order).items():
                        bucket[k].append(v)
    ranked = sorted(score_heads(normal, conflict, mode, a, b, g).items(),
                    key=lambda kv: kv[1], reverse=True)
    return [(k, float(v)) for k, v in ranked[:topk]], ranked


MODULE_SETS = {"qk": ("q_proj", "k_proj"),
               "qkv": ("q_proj", "k_proj", "v_proj"),
               "qkvo": ("q_proj", "k_proj", "v_proj", "o_proj")}


def get_lora_targets(model, layers, which="qk"):
    names = set(n for n, _ in model.named_modules())
    out = []
    for i in layers:
        hit = False
        for p in MODULE_SETS[which]:
            cand = f"model.layers.{i}.self_attn.{p}"
            if cand in names:
                out.append(cand); hit = True
        if not hit:
            for n in names:
                if f".{i}." in n and n.split(".")[-1] in {"qkv_proj", "c_attn",
                                                          "query_key_value"}:
                    out.append(n)
    if not out:
        raise ValueError(f"no attention projections found for layers={layers}")
    return sorted(set(out))


def focus_loss(attns, sys_mask, focus_idx, heads):
    by_layer = defaultdict(list)
    for tag, _ in heads:
        by_layer[int(tag.split("_")[0][1:])].append(int(tag.split("_H")[1]))

    dev = sys_mask.device
    B = sys_mask.size(0)
    bidx = torch.arange(B, device=dev)
    fidx = focus_idx.to(dev)
    valid = sys_mask.any(-1)
    if not valid.any():
        return torch.zeros([], dtype=torch.float32, device=dev)

    ratios = []
    m = sys_mask.unsqueeze(1).to(torch.float32)
    for l, hs in by_layer.items():
        A = attns[l].permute(0, 2, 1, 3)[bidx, fidx]
        A = A[:, hs, :].to(torch.float32)
        ratios.append((A * m).sum(-1) / A.sum(-1).clamp_min(1e-6))
    R = torch.cat(ratios, dim=1)
    v = valid.to(torch.float32).unsqueeze(1)
    return -(R * v).sum() / v.sum().clamp_min(1.0) / R.size(1)


def main():
    ap = argparse.ArgumentParser("FocalLoRA (fixed, + LM loss)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--targets_json", default="")
    ap.add_argument("--lm_loss", type=int, default=1)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_modules", choices=list(MODULE_SETS), default="qk")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_focus", type=float, default=1.0)
    ap.add_argument("--score", choices=["paper", "repo"], default="paper")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--order", choices=["task-first", "instr-first"], default="task-first")
    ap.add_argument("--probe_per_file", type=int, default=10)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--load_4bit", type=int, default=0)
    ap.add_argument("--include_permutation", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*_instruction.json")))
    if not args.include_permutation:
        files = [f for f in files if "permutation_" not in os.path.basename(f)]
    print(f"[fl] {len(files)} constraint files")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map={"": 0}, trust_remote_code=True, attn_implementation="eager")
    if args.load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kw)
    model.config.use_cache = False

    print("[fl] identifying focal heads ...")
    heads, ranked = detect_heads(files, model, tok, args.topk, args.order,
                                 args.score, args.alpha, args.beta, args.gamma,
                                 args.probe_per_file)
    print("[fl] focal heads:", heads)
    json.dump({"score_mode": args.score, "topk": args.topk, "focal_heads": heads,
               "all_ranked": ranked[:200]},
              open(os.path.join(args.output_dir, "focal_heads.json"), "w"), indent=2)

    layers = sorted({int(t.split("_")[0][1:]) for t, _ in heads})
    targets = get_lora_targets(model, layers, args.lora_modules)
    print(f"[fl] LoRA on {len(targets)} modules ({args.lora_modules}) across layers {layers}")

    if args.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, bias="none",
        target_modules=targets, task_type="CAUSAL_LM"))
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"[fl] trainable {tr:,} / {tot:,} = {100 * tr / tot:.4f}%")

    ds = ConflictDS(files, tok, args.order, args.train_limit,
                    args.targets_json or None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok))
    print(f"[fl] {len(ds)} prompts ({ds.n_with_target} with gold target), "
          f"{len(dl)} micro-steps/epoch")
    if args.lm_loss and ds.n_with_target == 0:
        raise SystemExit("--lm_loss 1 needs --targets_json with verified targets")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = max(1, args.epochs * len(dl) // args.grad_accum)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    model.train()
    hist = []
    for ep in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {ep + 1}/{args.epochs}")
        for it, batch in enumerate(pbar):
            sys_mask = batch.pop("sys_mask").to(model.device)
            focus_idx = batch.pop("focus_idx")
            labels = batch.pop("labels").to(model.device)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch, labels=labels if args.lm_loss else None,
                        output_attentions=True)
            lf = focus_loss(out.attentions, sys_mask, focus_idx, heads)
            lm = out.loss if args.lm_loss else torch.zeros_like(lf)
            loss = lm + args.lambda_focus * lf
            (loss / args.grad_accum).backward()
            if (it + 1) % args.grad_accum == 0 or it + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            hist.append({"lm": float(lm.detach()), "focus": float(lf.detach())})
            pbar.set_postfix(lm=f"{float(lm.detach()):.3f}",
                             sys_ratio=f"{-float(lf.detach()):.3f}")
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    json.dump({"args": vars(args), "n_steps": len(hist),
               "first20": hist[:20], "last20": hist[-20:]},
              open(os.path.join(args.output_dir, "train_log.json"), "w"), indent=2)
    print(f"[fl] saved adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
