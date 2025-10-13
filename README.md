# Temporal Graph Neural Networks for Real-Time Financial Fraud Detection

**Team Name:** GNN-erds  
**Course:** DSL501 - Machine Learning Project  
**Team Members:**
- Kunal Sewal (12341270)
- Kesav Patneedi (12341130)

## 📋 Project Overview

This project explores the application of Temporal Graph Neural Networks (TGNNs) for detecting fraudulent transactions in financial networks. By modeling how transactions evolve over time, we aim to capture subtle fraud patterns that static methods may miss.

### Key Objectives
- Compare static GNN baselines vs temporal GNNs for fraud detection
- Evaluate on multiple financial fraud benchmarks
- Demonstrate scalability from small to large-scale datasets

## 🗂️ Repository Structure

```
Financial-Fraud-Detection/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md              # Dataset download instructions
│   └── processed/             # Preprocessed data (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_temporal_models.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── models.py              # Model implementations
│   ├── train.py               # Training utilities
│   └── evaluate.py            # Evaluation metrics
├── results/
│   ├── baseline_results.md
│   └── figures/
└── configs/
    └── config.yaml            # Experiment configurations
```

## 📊 Datasets

We experiment with three financial fraud datasets of varying scales:

### 1. Ethereum Fraud Dataset (Kaggle)
- **Source:** [Kaggle - Ethereum Fraud Detection](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)
- **Scale:** Small-scale, suitable for rapid prototyping
- **Features:** Transaction graph with binary fraud labels
- **Status:** ✅ Ready to use

### 2. DGraph (NeurIPS 2022)
- **Source:** [NeurIPS 2022 Dataset Track](https://dgraph.xinye.com/)
- **Scale:** ~3M nodes, ~4M edges
- **Features:** Dynamic financial transaction network with labeled fraudster nodes
- **Status:** 🔄 To be downloaded

### 3. FiGraph (WWW 2025)
- **Source:** WWW 2025 Conference
- **Scale:** ~730K companies, ~1M edges, 9 yearly snapshots
- **Features:** Heterogeneous financial network with anomaly labels
- **Status:** 🔄 To be obtained

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU with ≥16GB VRAM (recommended)
- 32GB system RAM (for large datasets)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/KunalSewal/Financial-Fraud-Detection.git
cd Financial-Fraud-Detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Phase 1: Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Phase 2: Baseline Models
```bash
jupyter notebook notebooks/02_baseline_models.ipynb
```

### Phase 3: Temporal Models (Coming Soon)
```bash
jupyter notebook notebooks/03_temporal_models.ipynb
```

## 🔬 Models

### Baseline Models
- **MLP Classifier:** Simple feedforward network on node features
- **GraphSAGE:** Static graph neural network baseline

### Temporal Models
- **TGN (Rossi et al., ICML 2020):** Temporal Graph Network with memory modules
- **TGN-ATT (Kim et al., AAAI 2024):** Enhanced TGN for anomaly detection
- **MPTGNN (Saldaña-Ulloa et al., 2024):** Multi-path temporal GNN
- **TGAT (Xu et al., ICLR 2020):** Temporal Graph Attention Network
- **DyRep (Trivedi et al., ICLR 2019):** Dynamic representation learning
- **EvolveGCN:** Evolving graph convolutional networks

## 📈 Evaluation Metrics

- Precision, Recall, F1-Score
- ROC-AUC
- Average Precision (AP)
- Confusion Matrix
- Training/Validation Loss Curves

## 🎯 Current Progress

- [x] Project setup and repository structure
- [x] Dataset identification and download instructions
- [ ] Data exploration and preprocessing
- [ ] MLP baseline implementation
- [ ] GraphSAGE baseline implementation
- [ ] TGN implementation
- [ ] Full model comparison
- [ ] Final report and demo

## 📚 References

1. Kim, Y., et al. (2024). Temporal Graph Networks for Graph Anomaly Detection in Financial Networks. AAAI 2024. [arXiv:2404.00060](https://arxiv.org/abs/2404.00060)

2. Saldaña-Ulloa, D., et al. (2024). A Temporal Graph Network Algorithm for Detecting Fraudulent Transactions. Algorithms, 17(12), 552. [DOI:10.3390/a17120552](https://doi.org/10.3390/a17120552)

3. Wang, Z., et al. (2025). FiGraph: A Large-Scale Dynamic Financial Graph Benchmark. WWW 2025.

4. Huang, Q., et al. (2022). DGraph: A Large-Scale Financial Transaction Dataset. NeurIPS 2022.

5. Xu, D., et al. (2020). Inductive Representation Learning on Temporal Graphs (TGAT). ICLR 2020. [arXiv:2002.07962](https://arxiv.org/abs/2002.07962)

6. Trivedi, R., et al. (2019). DyRep: Learning Representations over Dynamic Graphs. ICLR 2019. [arXiv:1905.09936](https://arxiv.org/abs/1905.09936)

## 📝 License

This project is for academic purposes as part of DSL501 coursework.

## 🤝 Contributing

This is a course project. For collaboration, please contact the team members.

## 📧 Contact

- Kunal Sewal: kunal.sewal@example.edu
- Kesav Patneedi: kesav.patneedi@example.edu

---

**Last Updated:** October 2025
