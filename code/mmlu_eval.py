import argparse, json, os, random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

SYS = "Answer the multiple-choice question with a single letter: A, B, C, or D."


def build(tok, q, choices, no_system=False):
    body = q.strip() + "\n"
    for L, c in zip("ABCD", choices):
        body += f"{L}. {c}\n"
    body += "Answer:"
    msgs = ([{"role": "user", "content": body}] if no_system
            else [{"role": "system", "content": SYS}, {"role": "user", "content": body}])
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def letter_ids(tok):
    ids = {}
    for L in "ABCD":
        cand = []
        for form in (L, " " + L):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1:
                cand.append(t[0])
        if not cand:
            cand = [tok.encode(L, add_special_tokens=False)[0]]
        ids[L] = cand
    return ids


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default="")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_system", type=int, default=0)
    args = ap.parse_args()

    import glob
    hits = glob.glob("/mnt/hf-cache/hub/datasets--cais--mmlu/snapshots/*/all/test-*.parquet")
    if hits:
        print(f"[mmlu] loading cached parquet {hits[0]}")
        ds = load_dataset("parquet", data_files=hits[0], split="train")
    else:
        ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    random.Random(args.seed).shuffle(idx)
    idx = sorted(idx[:args.n]) if args.n else idx
    items = [ds[i] for i in idx]
    print(f"[mmlu] {len(items)} items from {len(ds)} (seed {args.seed})")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, attn_implementation="sdpa")
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload()
        print(f"[mmlu] merged LoRA from {args.lora_path}")
    model.eval()

    LID = letter_ids(tok)
    prompts = [build(tok, it["question"], it["choices"], args.no_system) for it in items]
    gold = [int(it["answer"]) for it in items]
    subj = [it["subject"] for it in items]

    preds = []
    for i in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[i:i + args.batch_size], return_tensors="pt",
                  padding=True, truncation=True, max_length=2048).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        for b in range(logits.size(0)):
            scores = [max(logits[b, t].item() for t in LID[L]) for L in "ABCD"]
            preds.append(int(max(range(4), key=lambda k: scores[k])))
        if (i // args.batch_size) % 10 == 0:
            print(f"  [{i + logits.size(0)}/{len(prompts)}]", flush=True)

    acc = 100.0 * sum(p == g for p, g in zip(preds, gold)) / len(gold)
    per = {}
    for s, p, g in zip(subj, preds, gold):
        per.setdefault(s, []).append(p == g)
    per = {k: 100.0 * sum(v) / len(v) for k, v in sorted(per.items())}
    dist = {L: preds.count(i) for i, L in enumerate("ABCD")}

    json.dump({"args": vars(args), "accuracy": acc, "n": len(gold),
               "pred_distribution": dist, "per_subject": per},
              open(args.out_json, "w"), indent=2)
    print(f"\n[mmlu] accuracy = {acc:.2f}%  (n={len(gold)})")
    print(f"[mmlu] prediction distribution {dist}")
    print(f"[mmlu] wrote {args.out_json}")


if __name__ == "__main__":
    main()
