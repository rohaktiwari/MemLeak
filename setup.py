from setuptools import find_packages, setup


setup(
    name="memleak",
    version="0.1.0",
    description="Membership inference detection toolkit and Streamlit app",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.37.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.18.0",
        "matplotlib>=3.8.0",
        "streamlit>=1.28.0",
        "tqdm>=4.66.0",
        "huggingface_hub>=0.19.0",
    ],
)

