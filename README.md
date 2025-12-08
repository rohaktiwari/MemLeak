# 🔐 MemLeak Research Framework

MemLeak is a research-grade framework for benchmarking Membership Inference Attacks (MIA) and defenses across vision and text datasets.

## Features
- Modular architecture: `attacks`, `defenses`, `models`, `datasets`, `training`, `evaluation`, `plotting`.
- Reproducible, config-driven experiments (YAML).
- Classic + SOTA attacks: confidence, loss, shadow (Shokri-style), meta-classifier.
- Defenses: DP-SGD (Opacus), label smoothing, dropout/weight decay, temperature scaling.
- Benchmark outputs: ROC curves, score histograms, mean ± std metrics, privacy–utility tradeoff plots.

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## Running Experiments
```bash
python run_experiment.py --config configs/cifar10/shadow.yaml
python run_experiment.py --config configs/ag_news/shadow.yaml
python run_experiment.py --config configs/imdb/shadow.yaml
```
Results are saved under `results/<dataset>/<attack-set>/<timestamp>/` with metrics JSON, plots, and logs.

## Structure
```
memleak/
  attacks/       # MIA implementations (confidence, loss, shadow, meta)
  defenses/      # DP-SGD hook, regularization helpers
  models/        # CNN/ResNet + Transformer builders, temperature scaling
  data/          # CIFAR-10, AG News, IMDB/SST-2 loaders
  training/      # Trainer with seed control and logging
  evaluation/    # AUC/PR/F1/acc aggregation
  plotting/      # ROC, histograms, privacy–utility plots
configs/         # YAML experiment configs
docs/            # Mini-paper and repo structure
run_experiment.py
```

## Research Notes
- Generalization gap captured per seed to study correlation with attack AUC.
- Random baseline included for comparison.
- Use provided configs as templates for ablations (e.g., dropout/weight decay sweeps, DP-SGD noise).

## Legacy API
The original `MembershipTester` remains for quick audits; the new pipeline is recommended for research-grade benchmarking.

## License
MIT
