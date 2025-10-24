# src/breaking_fast_detect_gpt/evaluate.py
import os, json, argparse, pandas as pd

def collect(results_dir, datasets, tag, criteria=("likelihood","rank","logrank","entropy")):
    rows=[]
    for ds in datasets:
        pref = os.path.join(results_dir, f"{ds}_{tag}")
        for c in criteria:
            path = f"{pref}.{c}.json"
            if not os.path.exists(path): 
                print(f"[skip] {path} not found"); continue
            with open(path, "r") as f:
                J = json.load(f)
            rows.append({
                "dataset": ds, "criterion": c,
                "roc_auc": J.get("roc_auc"),
                "pr_auc":  J.get("pr_auc"),
                "n_pos":   J.get("n_pos"),
                "n_neg":   J.get("n_neg"),
                "source":  os.path.basename(path),
            })
    return pd.DataFrame(rows).sort_values(["dataset","criterion"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdg_results_dir", required=True, help="e.g. /.../fast-detect-gpt/exp_attack/results")
    ap.add_argument("--out_csv", required=True, help="e.g. /.../Breaking-Fast-DetectGPT/results/metrics/baselines_metrics.csv")
    ap.add_argument("--datasets", nargs="+", default=["xsum","squad","writingprompts"])
    ap.add_argument("--dataset_tag", default="gpt-3.5-turbo")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = collect(args.fdg_results_dir, args.datasets, args.dataset_tag)
    df.to_csv(args.out_csv, index=False)
    print("wrote:", args.out_csv)
    print(df)

if __name__ == "__main__":
    main()
