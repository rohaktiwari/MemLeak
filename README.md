<<<<<<< Current (Your changes)
# 🔐 MemLeak Research Framework

MemLeak is a research-grade framework for benchmarking Membership Inference Attacks (MIA) and Defenses.

## Features
- **Modular Architecture**: Separate modules for `attacks`, `defenses`, `models`, and `datasets`.
- **Reproducible Experiments**: YAML config-driven experiments.
- **State-of-the-Art Defenses**: DP-SGD (Opacus), Label Smoothing, Temperature Scaling.
- **Benchmarking**: Automated ROC generation and metric logging.

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## Running Experiments
To run a specific experiment configuration:
```bash
python run_experiment.py --config configs/cifar10/shadow.yaml
```

## Structure
```
memleak/
  attacks/      # MIA implementations (Threshold, Shadow)
  defenses/     # Training and Post-hoc defenses
  models/       # Architectures
  engine.py     # Training loop
experiments/    # Analysis scripts
configs/        # Experiment configurations
docs/           # Documentation and Paper
```

## Research Analysis
To analyze the relationship between Generalization Gap and MIA:
```bash
# Script coming soon (or usage manual via config variations)
```

## Author
Rohak Tiwari | Virginia Tech
=======
# 🔐 MemLeak

**MemLeak is a Python toolkit that detects membership inference vulnerabilities in machine learning models—where attackers can determine if specific data was used during training. This is critical for healthcare AI (patient record leaks), financial models (transaction exposure), and any system trained on private data. The tool works with any HuggingFace transformer (GPT-2, BERT, custom models), runs three types of privacy attacks (loss-based, confidence-based, metric-based), and generates interactive dashboards showing per-sample leak probabilities, privacy risk scores (0-100), and actionable mitigation recommendations like differential privacy parameters and regularization adjustments. Built during research on AI safety at Virginia Tech, MemLeak operationalizes foundational work by Shokri et al. and Carlini et al. on unintended memorization in neural networks, providing both a Python API and Streamlit web interface for auditing production models.**

## Quick Start
```bash
git clone https://github.com/rohaktiwari/memleak
cd memleak
pip install -e .
```
```python
from memleak import MembershipTester

tester = MembershipTester("gpt2")
report = tester.run_attacks(train_data=["..."], test_data=["..."])
print(f"Risk: {report.risk_score}/100")
report.visualize()
report.recommendations()
```

Web interface: `streamlit run app.py`

Examples: `python examples/test_gpt2.py`, `python examples/test_bert.py`, `python examples/medical_model_case_study.py`

---

**Author:** Rohak Tiwari | CS @ Virginia Tech | [Google Scholar](https://scholar.google.com/citations?user=WRFLGrwAAAAJ&hl=en) | rohaktiwari@vt.edu

**License:** MIT
>>>>>>> Incoming (Background Agent changes)
