# src/breaking_fast_detect_gpt/score.py
import os, argparse, subprocess, shlex, sys

def run(cmd):
    print(">>", cmd)
    r = subprocess.run(shlex.split(cmd), check=True)
    return r.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdg_dir", required=True, help="path to upstream fast-detect-gpt repo")
    ap.add_argument("--results_dir", default=None, help="override output dir inside fdg (defaults to exp_attack/results)")
    ap.add_argument("--datasets", nargs="+", default=["xsum","squad","writingprompts"])
    ap.add_argument("--dataset_tag", default="gpt-3.5-turbo", help="suffix of dataset json files")
    ap.add_argument("--scoring_model_name", default="gpt-neo-2.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    py = sys.executable
    results_root = args.results_dir or f"{args.fdg_dir}/exp_attack/results"
    os.makedirs(results_root, exist_ok=True)

    for ds in args.datasets:
        dataset_file = f"{args.fdg_dir}/exp_gpt3to4/data/{ds}_{args.dataset_tag}.json"
        out_prefix   = f"{results_root}/{ds}_{args.dataset_tag}"
        cmd = (
          f'{py} {args.fdg_dir}/scripts/baselines.py '
          f'--scoring_model_name {args.scoring_model_name} '
          f'--dataset {ds} '
          f'--dataset_file {dataset_file} '
          f'--output_file {out_prefix} '
          f'--device {args.device} '
          + (f'--cache_dir "{args.cache_dir}"' if args.cache_dir else "")
        )
        run(cmd)

if __name__ == "__main__":
    main()
