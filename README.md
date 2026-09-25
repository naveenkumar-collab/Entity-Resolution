# Business Entity Resolution Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Evaluation Metric](https://img.shields.io/badge/Metric-Macro%20F0.5%20(0.814)-brightgreen.svg)](#evaluation--results)

High-performance, memory-efficient Machine Learning pipeline for multi-source Business Entity Resolution across noisy commercial datasets in **US**, **India**, and **France**.

Evaluated on precision-weighted **Macro-averaged $F_{0.5}$** with singleton penalties.

---

## 📌 Key Highlights

- **100% Intra-Country Partitioning**: Partitioned by jurisdiction (`US`, `India`, and unseen test country `France`), strictly preventing cross-country false merges and bounding RAM usage.
- **Multi-Lingual Text Normalization**: Multi-jurisdiction legal suffix stripping (US `Inc/LLC/Corp`, India `Pvt Ltd/LLP`, France `SARL/SAS/SASU/EURL/SCI`), Unicode NFKD accent normalization, domain extraction, and street abbreviation expansions.
- **High-Recall Multi-Pass Blocking**: Inverted-index candidate generator indexing exact core keys, compact domain strings, high-IDF tokens, 4-char prefixes, and street number-word pairs. Achieves **>99.99% reduction ratio** with ~84%+ candidate recall ceiling.
- **23-Dimensional Pairwise Feature Extractor**: RapidFuzz C++ Levenshtein algorithms, token sort/set ratios, character 3-gram Jaccards, address component similarities, and missing address indicators.
- **LightGBM Classifier + Singleton Shield**: Gradient boosted trees calibrated with a precision-tuned decision threshold ($\tau = 0.60$) and an active **Singleton Shield** ($\tau_{singleton} = 0.65$) to eliminate false merges and maximize Macro $F_{0.5}$.

---

## 📊 Evaluation & Results

Validation benchmark measured on a stratified 30,000 Source 1 entity held-out set (`US`: 17,994, `India`: 12,006, 5.59% singletons):

| Metric | Score |
|---|---|
| **Macro $F_{0.5}$** | **0.814** |
| **Precision** | **83.6%** |
| **Recall** | **74.1%** |
| **Singleton Accuracy** | **94.8%** |
| **Blocking Candidate Reduction Ratio** | **> 99.99%** |

Detailed methodology is available in [Documentation_template.md](Documentation_template.md).

---

## 📁 Repository Structure

```
├── code/
│   └── business_entity_resolution/
│       ├── README.md               # Pipeline documentation & CLI usage
│       ├── requirements.txt        # Pinned dependencies
│       └── src/
│           ├── blocking.py         # Multi-pass inverted index blocking engine
│           ├── normalize.py        # Multi-jurisdiction cleaning & normalization
│           ├── features.py         # 23 RapidFuzz similarity feature extractors
│           ├── model.py            # LightGBM model wrapper with Singleton Shield
│           ├── metrics.py          # Macro F0.5 and singleton evaluation logic
│           ├── pipeline.py         # End-to-end test inference pipeline
│           ├── train_and_evaluate.py # Pairwise training & evaluation
│           ├── tune_thresholds.py  # Grid-search threshold & singleton tuning
│           ├── test_blocking_keys.py # Blocking key sanity tests
│           ├── benchmark_blocking.py # Blocking recall & reduction benchmarks
│           └── create_val_split.py # Stratified validation set generator
├── utils/
│   └── validate_submission.py      # Format & schema validator for submissions
├── Documentation_template.md       # Full methodology documentation
├── CHALLENGE_README.md             # Original challenge problem statement
├── model_lgb.joblib                # Pre-trained LightGBM matching model
├── optimal_thresholds.json         # Calibrated threshold configuration
├── val_ground_truth_30k.tsv        # Held-out validation ground truth (30k)
├── val_source1_30k.tsv             # Held-out validation Source 1 records (30k)
├── requirements.txt                # Root dependency specification
└── .gitignore                      # Git configuration
```

---

## 🚀 Quick Start

### 1. Installation

Python 3.8+ is required.

```bash
pip install -r requirements.txt
```

### 2. End-to-End Inference

To generate the competition submission files from test datasets:

```bash
python code/business_entity_resolution/src/pipeline.py \
    --test-dir dataset/test \
    --output-dir output \
    --model-path model_lgb.joblib \
    --threshold 0.60 \
    --singleton-threshold 0.65
```

This generates:
- `output/matching_results.tsv` (Leaderboard submission file)
- `output/candidate_pairs.tsv` (Candidate blocking set)

### 3. Submission Validation

Validate the output format against challenge rules:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### 4. (Optional) Re-training & Threshold Tuning

To re-train the model from raw training datasets:

```bash
python code/business_entity_resolution/src/train_and_evaluate.py
```

To run grid-search threshold tuning:

```bash
python code/business_entity_resolution/src/tune_thresholds.py
```

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).