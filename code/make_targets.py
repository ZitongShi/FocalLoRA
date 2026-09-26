import argparse
import glob
import json
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import verifiers


def prompt_for(tok, s, retry=False):
    usr = s["task"] if not retry else (
        f"{s['task']}\n\n(Constraint you must follow: {s['system_message']})")
    return tok.apply_chat_template(
        [{"role": "system", "content": s["system_message"]},
         {"role": "user", "content": usr}],
        tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def gen(model, tok, prompts, max_new_tokens, bs):
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(g[j, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
        print(f"  [{i + len(chunk)}/{len(prompts)}]", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=24)
    args = ap.parse_args()

    samples = []
    for f in sorted(glob.glob(os.path.join(args.data_dir, "*_instruction.json"))):
        conf = [s for s in json.load(open(f)) if s["label"] == "conflict"]
        for s in (conf[:args.limit] if args.limit else conf):
            s["_file"] = os.path.basename(f)
            samples.append(s)
    print(f"[tgt] {len(samples)} conflict samples")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, attn_implementation="sdpa").eval()

    t0 = time.time()
    outs = gen(model, tok, [prompt_for(tok, s) for s in samples],
               args.max_new_tokens, args.batch_size)
    ok = [verifiers.check(s["system_message"], o) for s, o in zip(samples, outs)]
    print(f"[tgt] pass@1 = {100 * sum(ok) / len(ok):.2f}%  ({time.time() - t0:.0f}s)")

    bad = [i for i, v in enumerate(ok) if not v]
    if bad:
        outs2 = gen(model, tok, [prompt_for(tok, samples[i], retry=True) for i in bad],
                    args.max_new_tokens, args.batch_size)
        for i, o in zip(bad, outs2):
            if verifiers.check(samples[i]["system_message"], o):
                outs[i], ok[i] = o, True
        print(f"[tgt] pass@2 = {100 * sum(ok) / len(ok):.2f}%")

    recs = [{"id": s["id"], "file": s["_file"], "system_message": s["system_message"],
             "user_message": s["user_message"], "task": s["task"],
             "target": o, "verified": bool(v)}
            for s, o, v in zip(samples, outs, ok)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(recs, open(args.out_json, "w"), indent=2, ensure_ascii=False)

    by_file = {}
    for r in recs:
        by_file.setdefault(r["file"], []).append(r["verified"])
    print("\n=== verified-target rate by constraint ===")
    for f in sorted(by_file):
        v = by_file[f]
        print(f"  {f:46s} {100 * sum(v) / len(v):6.2f}%  (n={len(v)})")
    print(f"\n[tgt] kept {sum(r['verified'] for r in recs)}/{len(recs)} -> {args.out_json}")


if __name__ == "__main__":
    main()
