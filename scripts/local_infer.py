# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import numpy as np
import torch
import os
import glob
import argparse
import json
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from scipy.stats import norm

# Considering balanced classification that p(D0) equals to p(D1), we have
# p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        if args.sampling_model_name != args.scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()

        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'

        # Fallback for KeyError
        if key not in distrib_params:
            print(f"Warning: Key '{key}' not in distrib_params. Using 'gpt-neo-2.7B_gpt-neo-2.7B' as fallback.")
            key = 'gpt-neo-2.7B_gpt-neo-2.7B'

        self.classifier = distrib_params[key]

    # compute conditional probability curvature
    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        if labels.size(1) == 0: # Handle empty or single-token text
            return float('nan'), 0
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    # compute probability
    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        if np.isnan(crit):
            return float('nan'), crit, ntoken
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken

# --- NEW FLEXIBLE MAIN BLOCK ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- Text argument is now OPTIONAL ---
    parser.add_argument('--text', type=str, default=None, help='(Optional) Text to be analyzed.')
    parser.add_argument('--sampling_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    # --- EDIT THIS VARIABLE TO TEST YOUR TEXT ---

    default_text_to_analyze = """
    i am human
"""

    # Check if --text argument was provided
    if args.text is not None:
      text_to_analyze = args.text
      print("Using text provided from command line.")
    else:
      text_to_analyze = default_text_to_analyze
      print("No --text argument found. Using default text from script.")

    # 1. Initialize the detector
    print("Initializing detector...")
    detector = FastDetectGPT(args)
    print("Detector initialized.")

    # 2. Estimate the probability
    print(f"\nAnalyzing text: '{text_to_analyze.strip()[:100]}...'")
    prob, crit, ntokens = detector.compute_prob(text_to_analyze)

    # 3. Print the result
    print(f'\n--- Result ---')
    if np.isnan(crit):
        print(f'Could not analyze text. It might be too short or invalid.')
    else:
        print(f'Fast-DetectGPT criterion is {crit:.4f}')
        print(f'Probability of being machine-generated: {prob * 100:.0f}%')
