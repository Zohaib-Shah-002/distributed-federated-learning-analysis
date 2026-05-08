# Performance Evaluation of Federated Learning Algorithms under IID and Non-IID Data Distributions

## Overview

This repository contains the complete implementation and experimental framework for the research project:

**“Performance Evaluation of Federated Learning Algorithms under IID and Non-IID Data Distributions”**

The project presents a systematic empirical evaluation of four federated learning optimization algorithms:

* Federated Averaging (FedAvg)
* Federated Proximal (FedProx)
* SCAFFOLD
* FedNova

The evaluation is conducted under multiple heterogeneous federated learning environments using:

* IID partitioning
* Label-skew Non-IID partitioning
* Dirichlet-based Non-IID partitioning

Experiments are performed on:

* MNIST
* CIFAR-10

The framework additionally integrates:

* Earth Mover’s Distance (EMD) heterogeneity quantification
* Hyperparameter sensitivity analysis
* Statistical significance testing
* Convergence analysis
* Visualization and benchmarking tools

---

# Key Features

* Complete federated learning framework implemented from scratch using PyTorch
* No external federated learning libraries (Flower, PySyft, FedML) used
* Support for IID and multiple Non-IID partitioning strategies
* Implementation of FedAvg, FedProx, SCAFFOLD, and FedNova
* Multi-dataset benchmarking (MNIST and CIFAR-10)
* Hyperparameter sensitivity analysis
* Statistical validation using:

  * Welch’s t-test
  * Bonferroni correction
  * Cohen’s d effect size analysis
* Automated visualization pipeline
* Reproducible experiments using fixed random seeds

---

# Repository Structure

```text
├── Federated_Learning_Benchmark.ipynb
├── requirements.txt
└── README.md
```

The complete implementation, experiments, visualization pipeline, and statistical analysis are contained within a single Jupyter Notebook file:

```text
Federated_Learning_Benchmark.ipynb
```

Datasets are automatically downloaded using PyTorch dataset utilities when the notebook is executed with an active internet connection.

---

# System Requirements

## Hardware

* Apple M-series processor (recommended)
* NVIDIA CUDA GPU (optional)
* Minimum 16 GB RAM

## Software

* Python 3.10+
* PyTorch 2.8.0
* NumPy
* SciPy
* Matplotlib
* Pandas
* Seaborn

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/Zohaib-Shah-002/federated-learning-benchmark.git
cd federated-learning-benchmark
```

## 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Notebook

Open the Jupyter Notebook:

```bash
jupyter notebook Federated_Learning_Benchmark.ipynb
```

or using JupyterLab:

```bash
jupyter lab Federated_Learning_Benchmark.ipynb
```

The notebook contains:

* Dataset loading
* IID and Non-IID partitioning
* FedAvg, FedProx, SCAFFOLD, and FedNova implementations
* Training and aggregation pipeline
* Hyperparameter sensitivity analysis
* Statistical significance analysis
* Automated visualization generation

Datasets are downloaded automatically during execution using PyTorch dataset loaders.

---
# Dataset Links

## MNIST
http://yann.lecun.com/exdb/mnist/

## CIFAR-10
https://www.cs.toronto.edu/~kriz/cifar.html

# Experimental Configuration

| Parameter              | Value        |
| ---------------------- | ------------ |
| Number of Clients      | K = 5        |
| Participation Fraction | C = 0.6      |
| Communication Rounds   | T = 50       |
| Local Epochs           | E = 1        |
| Batch Size             | 64           |
| Learning Rate          | 0.01         |
| Momentum               | 0.9          |
| Weight Decay           | 1e-4         |
| Random Seeds           | {42, 52, 62} |

---

# Partitioning Strategies

The framework supports three partitioning strategies:

## IID Partitioning

Uniform class distribution across all clients.

## Label-Skew Non-IID Partitioning

Each client receives samples from only two classes.

## Dirichlet Non-IID Partitioning

Client distributions generated using Dirichlet concentration parameter:

```text
α ∈ {0.1, 0.5, 1.0}
```

---

# Evaluation Metrics

The framework evaluates:

* Global test accuracy
* Global test loss
* Convergence speed
* Training stability
* Statistical significance

Statistical validation includes:

* Welch’s t-test
* Bonferroni correction
* Cohen’s d effect size analysis

---

# Main Findings

* Federated optimization performs effectively under IID environments.
* Severe degradation occurs under label-skew Non-IID conditions.
* All evaluated algorithms converge to similar performance ranges under strong heterogeneity.
* SCAFFOLD exhibits task-dependent behavior across datasets.
* Dirichlet heterogeneity causes substantially smaller degradation compared to label-skew partitioning.
* A structural Non-IID accuracy barrier exists under class-exclusion heterogeneity.

---

# Reproducibility

All experiments are fully reproducible.

The framework fixes:

* Python random seed
* NumPy random seed
* PyTorch random seed

Experiments are repeated across three seeds:

```text
{42, 52, 62}
```

Results are reported as:

```text
mean ± standard deviation
```

---

# Research Paper

The complete IEEE-style research paper associated with this repository includes:

* Research background
* Problem formulation
* Related work
* Federated system design
* Experimental methodology
* Results and analysis
* Conclusions and contributions

---

# Citation

If you use this repository, please cite:

```text
Performance Evaluation of Federated Learning Algorithms under IID and Non-IID Data Distributions
```

---

# Authors

Zohaib Ali Shah

Distributed Computing Project

Department of Computer Engineering

Federated Learning Benchmark Framework

---

# License

This project is intended for academic and research purposes only.
