import argparse
import glob
import json
import os
import re
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import verifiers

ORDINARY_TMPL = "{sys}\n{usr}\n"


def build_prompt(tok, sample, fmt, order):
    sys_msg = sample["system_message"]
    task, um = sample["task"], sample["user_message"]
    usr = f"{task} {um}".strip() if order == "task-first" else f"{um} {task}".strip()
    if fmt == "template":
        return tok.apply_chat_template(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": usr}],
            tokenize=False, add_generation_prompt=True)
    return ORDINARY_TMPL.format(sys=sys_msg, usr=usr)


def load(model_path, lora_path, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, device_map={"": 0},
        trust_remote_code=True, attn_implementation="sdpa")
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print(f"[eval] merged LoRA from {lora_path}")
    model.eval()
    return model, tok


@torch.inference_mode()
def generate(model, tok, prompts, max_new_tokens, batch_size):
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new = gen[j, enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
        print(f"  [{i + len(chunk)}/{len(prompts)}]", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default="")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--fmt", choices=["template", "ordinary"], default="template")
    ap.add_argument("--order", choices=["task-first", "instr-first"], default="task-first")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--include_permutation", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*_instruction.json")))
    if not args.include_permutation:
        files = [f for f in files if "permutation_" not in os.path.basename(f)]
    samples = []
    for f in files:
        conf = [s for s in json.load(open(f)) if s["label"] == "conflict"]
        conf = conf[args.offset:]
        conf = conf[:args.limit] if args.limit else conf
        for s in conf:
            s["_file"] = os.path.basename(f)
            samples.append(s)
    print(f"[eval] {len(files)} files, {len(samples)} conflict samples")

    model, tok = load(args.model_path, args.lora_path)
    prompts = [build_prompt(tok, s, args.fmt, args.order) for s in samples]
    t0 = time.time()
    outs = generate(model, tok, prompts, args.max_new_tokens, args.batch_size)
    print(f"[eval] generation took {time.time() - t0:.0f}s")

    per_file, records = {}, []
    for s, o in zip(samples, outs):
        key, _ = verifiers.get_checker(s["system_message"])
        ok = verifiers.check(s["system_message"], o)
        per_file.setdefault(s["_file"], []).append(ok)
        records.append({"id": s["id"], "file": s["_file"], "constraint": key,
                        "compliant": ok, "output": o})

    rows = {f: 100.0 * sum(v) / len(v) for f, v in per_file.items()}
    overall = 100.0 * sum(r["compliant"] for r in records) / len(records)
    report = {"args": vars(args), "overall_compliance": overall,
              "per_file": rows, "n": len(records)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump({"report": report, "records": records},
              open(args.out_json, "w"), indent=2, ensure_ascii=False)

    print("\n=== compliance by constraint file ===")
    for f in sorted(rows):
        print(f"  {f:46s} {rows[f]:6.2f}%  (n={len(per_file[f])})")
    print(f"\n=== OVERALL system-instruction compliance: {overall:.2f}% "
          f"(n={len(records)}) ===")
    print(f"[eval] wrote {args.out_json}")


if __name__ == "__main__":
    main()
