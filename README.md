# Temporal Graph Neural Networks for Real-Time Financial Fraud Detection

**Team Name:** GNN-erds  
**Course:** DSL501 - Machine Learning Project  
**Team Members:**
- Kunal Sewal (12341270)
- Kesav Patneedi (12341130)

## 📋 Project Overview

This project implements and compares Temporal Graph Neural Networks (TGNNs) for detecting fraudulent transactions in financial networks. We've built a complete end-to-end system including model training, evaluation, and a production-ready real-time dashboard.

### Key Achievements
- ✅ Implemented and trained 5 models: Baseline GNN, TGAT, TGN, and 2 ensemble methods
- ✅ Achieved **74.78% AUC** with Weighted Ensemble (35% TGN + 65% TGAT)
- ✅ Built full-stack fraud detection dashboard with Next.js + FastAPI
- ✅ Real-time transaction monitoring and graph visualization
- ✅ Comprehensive model analytics and performance comparison

## 🗂️ Repository Structure

```
Financial-Fraud-Detection/
├── README.md                  # Main project documentation
├── requirements.txt           # Python dependencies
├── api/                       # FastAPI backend
│   ├── main.py               # API server with fraud detection endpoints
│   └── requirements.txt      # API-specific dependencies
├── dashboard/                 # Next.js frontend
│   ├── app/                  # Next.js 13+ app directory
│   │   ├── page.tsx         # Dashboard homepage
│   │   ├── analytics/       # Model analytics page
│   │   ├── graph/           # Network visualization
│   │   ├── monitoring/      # Live transaction monitoring
│   │   ├── experiments/     # Training history
│   │   ├── alerts/          # Fraud alerts
│   │   ├── security/        # System security status
│   │   └── settings/        # Configuration
│   ├── components/          # React components
│   ├── lib/                 # API client and utilities
│   ├── package.json
│   └── tsconfig.json
├── data/                      # Dataset and preprocessing
│   ├── ibm/                  # IBM fraud detection dataset
│   │   └── ibm_fraud_29k_nonfraud_60k.csv
│   ├── preprocessing/        # Data processing scripts
│   └── processed/            # Processed graph data
├── src/                       # Core ML code
│   ├── models/               # Model implementations
│   │   ├── tgn.py           # Temporal Graph Network
│   │   ├── tgat.py          # Temporal Graph Attention
│   │   └── mptgnn.py        # Multi-path TGNN
│   ├── data_utils.py        # Dataset utilities
│   └── evaluate.py          # Evaluation metrics
├── scripts/                   # Training and analysis scripts
│   ├── train_tgn_fraud.py
│   ├── train_tgat_fraud.py
│   ├── compare_models.py
│   └── README.md
├── saved_models/              # Trained model checkpoints
│   ├── tgn_fraud_best.pt
│   └── tgat_fraud_best.pt
├── checkpoints/               # Training checkpoints
├── results/                   # Experimental results
│   ├── Final_Results.md      # Summary of all model results
│   └── figures/              # Plots and visualizations
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_temporal_models.ipynb
├── configs/                   # Configuration files
│   └── config.yaml
└── docs/                      # Documentation
    ├── DASHBOARD_SETUP.md
    └── TRAINING_GUIDE.md
```

## 📊 Dataset

### IBM Credit Card Fraud Detection Dataset
- **Source:** IBM Transactions Dataset
- **Scale:** 89,757 transactions, 1,527 users
- **Fraud Ratio:** 33.15% (29,757 fraud, 60,000 non-fraud)
- **Time Period:** 2019 credit card transactions
- **Features:** 
  - User demographics and behavior
  - Transaction amounts and timestamps
  - Merchant information (MCC codes, states)
  - Temporal patterns (hour, day of week)
- **Graph Construction:**
  - **Nodes:** 1,527 users (filtered for ≥10 transactions)
  - **Edges:** 857,732 temporal edges (users active on same day)
  - **Node Features:** 10 aggregated features per user
  - **Fraud Labels:** User-level (87.56% fraud users) and transaction-level
- **Status:** ✅ Loaded and preprocessed

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 18+ (for dashboard)
- CUDA-capable GPU (optional, for faster training)
- 8GB RAM minimum

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/KunalSewal/Financial-Fraud-Detection.git
cd Financial-Fraud-Detection
```

2. **Create Python virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install dashboard dependencies:**
```bash
cd dashboard
npm install
cd ..
```

## 🚀 Quick Start

### 1. Start the Backend API
```bash
cd api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will:
- Load the IBM dataset (89,757 transactions)
- Build the temporal graph (1,527 nodes, 857,732 edges)
- Register 5 trained models
- Serve endpoints at `http://localhost:8000`

### 2. Start the Dashboard
```bash
cd dashboard
npm run dev
```

Open `http://localhost:3000` to access the dashboard.

### 3. Explore the System

**Dashboard Features:**
- 📊 **Analytics:** ROC curves, confusion matrices, model comparison
- 🔴 **Live Monitoring:** Simulated transaction stream with fraud detection
- 🕸️ **Graph Visualization:** Interactive 2D network with fraud communities
- 🧪 **Experiments:** Training history and performance metrics
- 🔒 **Security:** System health and API status

### 4. Train Models (Optional)

```bash
# Train TGN model
python scripts/train_tgn_fraud.py

# Train TGAT model
python scripts/train_tgat_fraud.py

# Compare all models
python scripts/compare_models.py
```

## 🔬 Models & Results

### Implemented Models

1. **Baseline GNN**
   - Static graph neural network
   - Results: 69.10% AUC, 67.52% Accuracy

2. **TGAT (Temporal Graph Attention Network)**
   - Attention-based temporal aggregation
   - Results: 68.23% AUC, 71.68% Accuracy, 31.35% Recall

3. **TGN (Temporal Graph Network)**
   - Memory module for temporal patterns
   - Results: 68.41% AUC, 71.64% Accuracy, 26.97% Recall

4. **Weighted Ensemble** ⭐ **BEST MODEL**
   - 35% TGN + 65% TGAT
   - Results: **74.78% AUC**, 71.98% Accuracy, 27.65% Recall

5. **Voting Ensemble**
   - Majority voting across models
   - Results: 66.49% AUC, 72.42% Accuracy, 27.16% Recall

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|----------|
| Baseline GNN | 67.52% | 70.99% | 3.52% | 6.70% | 69.10% |
| TGAT | 71.68% | 69.26% | 31.35% | 42.06% | 68.23% |
| TGN | 71.64% | 70.20% | 26.97% | 39.55% | 68.41% |
| **Weighted Ensemble** | **71.98%** | **69.44%** | **27.65%** | **39.55%** | **74.78%** |
| Voting Ensemble | 72.42% | 72.36% | 27.16% | 39.49% | 66.49% |

See `results/Final_Results.md` for detailed metrics.

## 📈 Key Insights

- **Ensemble methods outperform individual models** - Weighted ensemble achieves 74.78% AUC
- **Low recall across all models** (27-31%) - Challenging imbalanced dataset
- **High precision** (69-72%) - Models are conservative in fraud predictions
- **Temporal models show improvement** over static GNN baseline

## 🎯 Project Status

### Completed ✅
- [x] Dataset acquisition and preprocessing (IBM fraud dataset)
- [x] Temporal graph construction (857K edges from 89K transactions)
- [x] Baseline GNN implementation and training
- [x] TGAT implementation and training
- [x] TGN implementation and training
- [x] Ensemble methods (weighted + voting)
- [x] Full model comparison and evaluation
- [x] FastAPI backend with fraud detection endpoints
- [x] Next.js dashboard with real-time monitoring
- [x] Interactive graph visualization (2D force-directed)
- [x] Model analytics and performance comparison
- [x] Complete documentation and README

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

- Kunal Sewal: kunalsewal@gmail.com
- Kesav Patneedi: kesavpatneedi@gmail.com

---

**Last Updated:** 14th Nov 2025
