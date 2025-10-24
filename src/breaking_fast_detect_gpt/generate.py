# src/breaking_fast_detect_gpt/generate.py
import os, json, argparse, random, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def _load_model(name, dtype=None, device_map=None):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        low_cpu_mem_usage=True,
        device_map=device_map if device_map else ("auto" if torch.cuda.is_available() else None)
    ).eval()
    return tok, model

def _trim_to_prefix(tok, text, n_tokens):
    ids = tok.encode(text, add_special_tokens=False)
    ids = ids[:n_tokens] if len(ids) >= n_tokens else ids
    return tok.decode(ids)

def _continue(tok, model, prefix, max_new_tokens, temperature, top_p):
    inp = tok(prefix, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
        )
    s = tok.decode(out[0], skip_special_tokens=True)
    if s.startswith(prefix): s = s[len(prefix):].lstrip()
    return s or prefix

def _save(out_path, originals, sampleds):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"original": originals, "sampled": sampleds}, f, ensure_ascii=False)
    print(f"[saved] {out_path}  (#orig={len(originals)} #sampled={len(sampleds)})")

def build_xsum(args, tok, model):
    ds = load_dataset("xsum", split="test")
    orig, samp = [], []
    for i in range(args.n):
        t = str(ds[i]["summary"]).strip()
        if not t: continue
        pref = _trim_to_prefix(tok, t, args.prefix_tokens)
        gen  = _continue(tok, model, pref, args.max_new_tokens, args.temperature, args.top_p)
        orig.append(t); samp.append(gen)
    _save(f"{args.out_dir}/xsum_{args.source_model_tag}.json", orig, samp)

def build_squad(args, tok, model):
    ds = load_dataset("squad", split="validation")
    orig, samp = [], []
    for i in range(args.n):
        t = str(ds[i]["context"]).strip()
        if not t: continue
        pref = _trim_to_prefix(tok, t, args.prefix_tokens)
        gen  = _continue(tok, model, pref, args.max_new_tokens, args.temperature, args.top_p)
        orig.append(t); samp.append(gen)
    _save(f"{args.out_dir}/squad_{args.source_model_tag}.json", orig, samp)

def build_wp(args, tok, model):
    ds = load_dataset("writing_prompts", "train")["train"]
    orig, samp = [], []
    i = 0
    for ex in ds:
        if i >= args.n: break
        t = (ex.get("story") or ex.get("prompt") or "").strip()
        if not t: continue
        pref = _trim_to_prefix(tok, t, args.prefix_tokens)
        gen  = _continue(tok, model, pref, args.max_new_tokens, args.temperature, args.top_p)
        orig.append(t); samp.append(gen); i += 1
    _save(f"{args.out_dir}/writingprompts_{args.source_model_tag}.json", orig, samp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling_model_name", default="EleutherAI/gpt-neo-2.7B")
    ap.add_argument("--out_dir", required=True, help="e.g. /.../fast-detect-gpt/exp_gpt3to4/data")
    ap.add_argument("--source_model_tag", default="gpt-3.5-turbo")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prefix_tokens", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--datasets", nargs="+", default=["xsum","squad","writingprompts"])
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    tok, model = _load_model(args.sampling_model_name)
    os.makedirs(args.out_dir, exist_ok=True)

    if "xsum" in args.datasets:            build_xsum(args, tok, model)
    if "squad" in args.datasets:           build_squad(args, tok, model)
    if "writingprompts" in args.datasets:  build_wp(args, tok, model)

if __name__ == "__main__":
    main()
