# ST-GAFE-Net: Explainable Spatio-Temporal Graph Attention Network for WSN Anomaly Detection

## 📄 Paper Title
**Explainable Spatio-Temporal Graph Attention Network with Frequency-Enhanced Learning for Fault and Anomaly Detection in Wireless Sensor Networks**

---

## 🔍 Overview

This repository contains the official implementation of **ST-GAFE-Net**, a novel deep learning framework for fault and anomaly detection in Wireless Sensor Networks (WSNs).

ST-GAFE-Net integrates three complementary learning branches:
- **BiGRU** — Bidirectional Gated Recurrent Unit for temporal learning
- **GAT** — Graph Attention Network for spatial learning
- **FFT** — Fast Fourier Transform branch for frequency-domain feature extraction

A **SHAP-based XAI layer** is also integrated for explainable anomaly detection decisions.

---

## 🏗️ Model Architecture

```
Input (Raw WSN Sensor Data)
        ↓
Pre-processing (Normalization + SMOTE + FFT)
        ↓
┌──────────────┬──────────────┬──────────────┐
│  BiGRU       │     GAT      │    FFT       │
│ (Temporal)   │  (Spatial)   │ (Frequency)  │
└──────────────┴──────────────┴──────────────┘
        ↓
Feature Fusion (Concatenation)
        ↓
Classification Network (FC → BN → ReLU → Dropout)
        ↓
Output: Normal / Anomaly + SHAP Explanation
```

---

## 📊 Datasets

| Dataset | Source | Rows | Features |
|---|---|---|---|
| WSN Fault Detection | [GitHub](https://github.com/tmoulahi/Dataset-for-WSN-fault-detection) | 562,560 | 13 |
| Intel Lab Data | [MIT CSAIL](http://db.csail.mit.edu/labdata/labdata.html) | 2,219,803 | 4 |
| SWaT | [Kaggle](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system) | 50,000 | 51 |
| WSN-DS | [Kaggle](https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds) | 374,661 | 18 |

---

## 📈 Results

| Dataset | Accuracy | F1 Score | AUC-ROC | MCC |
|---|---|---|---|---|
| WSN_Fault | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Intel_Lab | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SWaT | 0.9913 | 0.8953 | 0.9986 | 0.8948 |
| WSN_DS | 0.9957 | 0.9769 | 0.9958 | 0.9746 |

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ST-GAFE-Net-WSN-Anomaly-Detection.git
cd ST-GAFE-Net-WSN-Anomaly-Detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Update dataset path
Open `ST_GAFE_Net_WSN.py` and update:
```python
BASE_PATH = r"your/dataset/path/here"
```

### 4. Run the main model
```bash
python ST_GAFE_Net_WSN.py
```

### 5. Run XAI analysis
```bash
python XAI_Analysis.py
```

---

## 📁 Repository Structure

```
ST-GAFE-Net-WSN-Anomaly-Detection/
│
├── ST_GAFE_Net_WSN.py          # Main model — training + evaluation
├── WSN_Fault_Fix.py            # WSN Fault dataset fixed loader
├── XAI_Analysis.py             # SHAP-based explainability analysis
├── requirements.txt            # Python dependencies
└── README.md                   # Project description
```

---

## 🔬 Experimental Setup

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Scheduler | StepLR (step=20, gamma=0.5) |
| CV Folds | 5 |
| Test Split | 20% |
| SMOTE | Yes (train only) |

---

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{stgafenet2026,
  title={Explainable Spatio-Temporal Graph Attention Network with 
         Frequency-Enhanced Learning for Fault and Anomaly Detection 
         in Wireless Sensor Networks},
  author={S.S. and S.D.},
  journal={Under Review},
  year={2026}
}
```

---

## 📬 Contact

For any questions or issues, please open a GitHub issue or contact the authors.

---

## 📜 License

This project is licensed under the MIT License.
