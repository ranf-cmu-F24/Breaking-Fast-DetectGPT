# Breaking-Fast-DetectGPT
## Introduction
## How to Run
### Run in Google Colab (suggested, ColabPro recommended)
Move the [`attack_fast_detect.ipynb`](attack_fast_detect.ipynb) to your own Google Colab environment and run directly.
## Datasets and model
### Raw data
We used XSum dataset and gpt-neo-2.7B model from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt/tree/main/exp_main/data)
```
dataset = "xsum"
source_model = "gpt-neo-2.7B"
```
### Attacked data
We applied three attacks on the raw data, including Synonym, Strategic, and Paraphrase. The corresponding output can be found in [`data/attacks`](data/attacks/) folder.
## Results
### Baseline result
[baseline result](results/baseline/exp_gpt3to4/xsum_gpt-3.5-turbo.gpt-neo-2.7B_gpt-neo-2.7B.sampling_discrepancy.json)
### Our results
1. [AUROC metrics of attacked data](results/attacks/)
2. [Illustrative graphs of AUROC metrics](results/plots/)
3. Semantic Analysis results:
    * [Synonym attack](results/plots/semantic_analysis_paraphrase.png)
    * [Strategic attack](results/plots/semantic_analysis_strategic.png)
    * [Paraphrase attack](results/plots/semantic_analysis_paraphrase.png)
