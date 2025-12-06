# MemLeak — Membership Inference Toolkit

MemLeak is a production-ready Python toolkit and Streamlit app that detects **membership inference attacks** against HuggingFace models. It loads any transformer checkpoint, runs three complementary attacks, quantifies per-sample leak probability, and surfaces privacy risk with visual dashboards plus mitigation guidance.

## Features
- Load HuggingFace models and tokenizers (local path or hub id)
- Three attacks: loss-based, confidence-based, metric-based (combined features)
- Per-example membership probability, confidence, and ranking
- Privacy risk score (0–100) and mitigation recommendations
- Streamlit UI for uploads (model id/path + dataset) and one-click analysis
- Plotly dashboards: leak probability scatter, risk heatmap, ROC-like curves
- Example scripts for GPT-2, BERT, and a medical-model case study

## Why Membership Inference Matters
Models can overfit and memorize sensitive training points. Adversaries test whether a candidate sample was part of training by comparing model behavior on train vs. non-train distributions. Lower loss / higher confidence usually signals membership. MemLeak operationalizes these signals to measure leak risk and suggest mitigations.

## Quickstart
```bash
git clone https://github.com/your-org/memleak
cd memleak
pip install -e .
streamlit run app.py
```

## Python API
```python
from memleak import MembershipTester

train = ["Alice has diabetes", "The cat sat on the mat"]
test = ["Bob plays guitar", "This is unseen"]

tester = MembershipTester(model="bert-base-uncased", max_length=128)
report = tester.run_attacks(train, test)

print(report.summary.head())
report.visualize()  # opens Plotly figures
report.recommendations()  # mitigation hints
```

## Running the Web App
```bash
streamlit run app.py
# On HuggingFace Spaces: point the Space to app.py
```

## Examples
- `examples/test_gpt2.py`
- `examples/test_bert.py`
- `examples/medical_model_case_study.py`

## Attacks Implemented
- **Loss-based:** compare per-sample loss distributions between train/test; lower loss → higher membership probability.
- **Confidence-based:** uses max softmax probability; higher confidence → higher membership probability.
- **Metric-based:** combines loss, entropy, margin, and optional gradient proxies via a simple learned-style scorer.

## Mitigation Guidance
- Differential privacy (ε targets per risk level)
- Regularization tweaks (dropout, weight decay)
- Data augmentation suggestions
- Vulnerable layer hints from confidence/entropy patterns

## Tests
```bash
pytest -q
```

## Citation
If you use MemLeak in research, cite:
```
@software{memleak2025,
  title  = {MemLeak: Membership Inference Detection Toolkit},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/your-org/memleak}
}
```

