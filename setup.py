from setuptools import find_packages, setup

setup(
    name="memleak",
    version="0.2.0",
    description="Research Framework for Membership Inference Attacks & Defenses",
    author="Rohak Tiwari",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "scikit-learn>=1.2.2",
        "pandas>=2.0.0",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "opacus>=1.4.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "scipy>=1.10.0",
    ],
)
