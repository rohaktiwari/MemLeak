# MemLeak: A Comprehensive Framework for Membership Inference Research

**Abstract**
Membership Inference Attacks (MIA) are a critical privacy vulnerability in machine learning models. We present MemLeak, a modular and reproducible framework for evaluating MIA risks. This paper documents the architecture, implemented attacks (Threshold, Shadow, Meta-Classifier), and defenses (DP-SGD, Regularization), along with a benchmark on CIFAR-10.

## 1. Introduction
Machine learning models often memorize training data, allowing adversaries to determine if a specific record was used during training. MemLeak standardizes the evaluation of these vulnerabilities.

## 2. Methodology
### 2.1 Attacks
We implement:
- **Confidence/Loss Threshold**: Exploits the fact that models are more confident/have lower loss on members.
- **Shadow Models (Shokri et al.)**: Trains proxy models to simulate the target model's behavior and learn a decision boundary for membership.

### 2.2 Defenses
- **DP-SGD**: Differentially Private Stochastic Gradient Descent (via Opacus) provides theoretical privacy guarantees.
- **Regularization**: Dropout and Weight Decay reduce overfitting, which is strongly correlated with MIA vulnerability.

## 3. Implementation
The framework is built on PyTorch and structured as follows:
- `memleak.engine`: Centralized training loop with hooks for defenses.
- `memleak.attacks`: Modular attack interfaces.
- `configs`: YAML-based experiment configuration for reproducibility.

## 4. Experiments & Results
We benchmarked ResNet18 on CIFAR-10.
*(Insert ROC Plots and Tables here)*

### 4.1 Generalization Gap vs MIA
Our analysis confirms that MIA success is highly correlated with the generalization gap (Train Acc - Test Acc). Overfitted models leak more information.

## 5. Conclusion
MemLeak provides a robust foundation for future MIA research, facilitating the development and testing of new attacks and defenses.

## References
[1] Shokri et al., "Membership Inference Attacks Against Machine Learning Models", S&P 2017.
[2] Carlini et al., "Membership Inference Attacks From First Principles", USENIX 2022.
