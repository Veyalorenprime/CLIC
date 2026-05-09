"""Setup script for CLIC package"""

from setuptools import setup, find_packages

setup(
    name="clic",
    version="0.1.0",
    author="Yahya El Fataoui",
    description="Causal Latent Circuits for Interpretable OOD Detection",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)