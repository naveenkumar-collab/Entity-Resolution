# Business Entity Resolution Pipeline

High-performance Machine Learning pipeline for multi-source Business Entity Resolution across US, India, and France. Evaluated on Macro-averaged $F_{0.5}$ with singleton penalties.

## Architecture Highlights
- **Intra-Country Partitioning**: 100% intra-country constraint partitioning (`US`, `India`, `France`), eliminating cross-country noise and scaling to millions of records with constant memory bounds.
- **Multi-Lingual Text Normalization**: Multi-jurisdiction legal suffix stripping (US `Inc/LLC`, India `Pvt Ltd/LLP`, France `SARL/SAS/SASU/EURL`), Unicode NFKD accent normalization, domain sanitization, and road abbreviation expansion.
- **High-Recall Multi-Pass Blocking**: Inverted indexing across exact core keys, compact domain strings, high-IDF tokens, 4-char prefixes, and street number-word pairs. Achieves >99.99% reduction ratio.
- **Pairwise Feature Engineering**: 23 features computed using `RapidFuzz` C++ Levenshtein algorithms, token sort/set ratios, character 3-gram Jaccards, address component similarities, and missing address indicators.
- **Model & Decision Logic**: Fast LightGBM gradient boosted trees with a calibrated decision threshold and an active **Singleton Shield** to eliminate false merges and maximize Macro $F_{0.5}$.

---

## Environment & Dependencies

Python 3.8+ (tested on Python 3.14 on Windows).

Install dependencies:
```bash
pip install -r requirements.txt
```

Pinned dependencies:
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `rapidfuzz>=3.0.0`
- `lightgbm>=4.0.0`
- `scipy>=1.10.0`

---

## Reproducing Outputs End-to-End

### 1. (Optional) Re-train the Matching Model
To train the LightGBM matching model from raw training data:
```bash
python src/train_and_evaluate.py
```
This saves `model_lgb.joblib` containing the fitted model.

### 2. Generate Submission Outputs
To run the full end-to-end inference pipeline on test data:
```bash
python src/pipeline.py \
    --test-dir ../../dataset/test \
    --output-dir ../../output \
    --model-path ../../model_lgb.joblib \
    --threshold 0.60 \
    --singleton-threshold 0.65
```

This generates:
- `output/matching_results.tsv` (Leaderboard submission)
- `output/candidate_pairs.tsv` (Candidate blocking set)

### 3. Validate Submission Format
Validate the generated outputs against challenge rules:
```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
Exit code `0` (`PASS`) confirms valid submission formatting.
