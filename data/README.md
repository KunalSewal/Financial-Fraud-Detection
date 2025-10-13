# Dataset Information

This directory contains instructions and scripts for downloading and preprocessing the datasets used in this project.

## 📁 Directory Structure

```
data/
├── README.md (this file)
├── raw/                    # Raw downloaded data (gitignored)
├── processed/              # Preprocessed data ready for training (gitignored)
└── scripts/
    ├── download_ethereum.py
    ├── download_dgraph.py
    └── preprocess_data.py
```

## 📊 Datasets Overview

### 1. Ethereum Fraud Detection Dataset

**Source:** Kaggle  
**Size:** ~50 MB  
**License:** CC0: Public Domain  
**Description:** Transaction graph from Ethereum blockchain with binary fraud labels.

#### Download Instructions:

**Option 1: Manual Download**
1. Visit: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
2. Download the dataset
3. Extract to `data/raw/ethereum/`

**Option 2: Using Kaggle API**
```bash
# Install Kaggle CLI
pip install kaggle

# Setup Kaggle credentials (get from kaggle.com/account)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d vagifa/ethereum-frauddetection-dataset
unzip ethereum-frauddetection-dataset.zip -d data/raw/ethereum/
```

**Expected Files:**
- `transaction_dataset.csv` - Transaction records
- `transaction_node_features.csv` - Node features (if available)

---

### 2. DGraph Dataset (NeurIPS 2022)

**Source:** NeurIPS 2022 Dataset Track  
**Size:** ~2-3 GB  
**License:** Check official source  
**Description:** Large-scale dynamic financial transaction network with ~3M nodes and ~4M edges.

#### Download Instructions:

1. Visit: https://dgraph.xinye.com/
2. Register and request dataset access
3. Download the dataset files
4. Place in `data/raw/dgraph/`

**Expected Files:**
- `edges.csv` - Edge list with timestamps
- `nodes.csv` - Node features and labels
- `metadata.json` - Dataset metadata

**Note:** Due to size and potential licensing restrictions, this dataset is not included in the repository.

---

### 3. FiGraph Dataset (WWW 2025)

**Source:** WWW 2025 Conference  
**Size:** ~1-2 GB  
**License:** Research use only  
**Description:** Heterogeneous financial network with ~730K companies across 9 yearly snapshots.

#### Download Instructions:

⚠️ **Status:** Dataset to be released with WWW 2025 proceedings

1. Check WWW 2025 conference website for data release
2. Alternative: Contact authors directly via paper reference
3. Place downloaded files in `data/raw/figraph/`

**Expected Files:**
- `snapshots/` - 9 yearly graph snapshots
- `companies.csv` - Company node features
- `relationships.csv` - Edge relationships
- `labels.csv` - Anomaly labels

---

## 🔄 Data Preprocessing

After downloading raw data, run preprocessing scripts:

```bash
# Preprocess Ethereum dataset
python data/scripts/preprocess_data.py --dataset ethereum --input data/raw/ethereum/ --output data/processed/ethereum/

# Preprocess DGraph (when available)
python data/scripts/preprocess_data.py --dataset dgraph --input data/raw/dgraph/ --output data/processed/dgraph/

# Preprocess FiGraph (when available)
python data/scripts/preprocess_data.py --dataset figraph --input data/raw/figraph/ --output data/processed/figraph/
```

## 📝 Data Format

All preprocessed datasets will be converted to PyTorch Geometric format:

```python
Data(
    x=node_features,           # Node feature matrix [num_nodes, num_features]
    edge_index=edge_index,     # Edge connectivity [2, num_edges]
    edge_attr=edge_features,   # Edge features [num_edges, num_edge_features]
    y=labels,                  # Node labels [num_nodes]
    edge_time=timestamps       # Edge timestamps [num_edges]
)
```

## 💾 Storage Requirements

| Dataset | Raw Size | Processed Size | Total |
|---------|----------|----------------|-------|
| Ethereum | ~50 MB | ~100 MB | ~150 MB |
| DGraph | ~2-3 GB | ~4-5 GB | ~7 GB |
| FiGraph | ~1-2 GB | ~2-3 GB | ~4 GB |

**Recommended:** At least 15 GB free disk space

## 🔒 Data Privacy and Ethics

- All datasets contain anonymized transaction data
- No personally identifiable information (PII) is included
- Datasets are used solely for academic research purposes
- Comply with each dataset's license and terms of use

## 🐛 Troubleshooting

**Issue: Kaggle API authentication fails**
- Ensure `kaggle.json` is in the correct location
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`

**Issue: Out of memory during preprocessing**
- Process data in chunks
- Use data streaming methods
- Ensure sufficient RAM (32GB recommended for DGraph)

**Issue: Dataset links not working**
- Check for updated links in project README
- Contact dataset authors
- Raise an issue in our GitHub repository

## 📧 Support

For dataset-related issues:
1. Check this README
2. See main project README
3. Contact team members
4. Open a GitHub issue