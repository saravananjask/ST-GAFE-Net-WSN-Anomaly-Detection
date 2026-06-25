# =============================================================================
# XAI Analysis — SHAP for ST-GAFE-Net WSN Anomaly Detection
# Generates XAI figures for Section 7 of the paper
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.fft import fft
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# CONFIG
# =============================================================================
BASE_PATH = r"E:\Journals\Journal 2026\June 26\New WSN 22.6.26\Dataset"
OUTPUT_PATH = r"E:\Journals\Journal 2026\June 26\New WSN 22.6.26\Results"
FIGURES_PATH = os.path.join(OUTPUT_PATH, 'Figures')
FAULT_PATH = os.path.join(BASE_PATH, 'Dataset-for-WSN-fault-detection-master')

RANDOM_STATE = 42
MAX_SAMPLES = 100000
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
XAI_SAMPLES = 100  # small sample for SHAP — avoids memory issues

os.makedirs(FIGURES_PATH, exist_ok=True)

print("=" * 60)
print("XAI Analysis — SHAP for ST-GAFE-Net")
print("=" * 60)

# =============================================================================
# MODEL DEFINITION
# =============================================================================

class BiGRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.bigru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.bigru(x)
        return self.norm(out.squeeze(1))

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.W(x)
        return self.relu(self.attn(h))

class STGAFENet(nn.Module):
    def __init__(self, input_size, hidden_size=64, gat_out=32, num_classes=2):
        super().__init__()
        self.temporal = BiGRUBlock(input_size, hidden_size)
        self.gat = GraphAttentionLayer(input_size, gat_out)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + gat_out, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(torch.cat([self.temporal(x), self.gat(x)], dim=-1))

def train_model(X_train, y_train, input_size, epochs=EPOCHS):
    model = STGAFENet(input_size=input_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")
    return model

# =============================================================================
# DATASET LOADERS
# =============================================================================

def add_fft(X):
    fft_feat = np.abs(fft(X, axis=1))[:, :X.shape[1] // 2]
    return np.hstack([X, fft_feat])

def load_and_prep(df, label_col=None):
    if label_col and label_col in df.columns:
        y = df[label_col].values.astype(int)
        df = df.drop(columns=[label_col])
    else:
        y = df['label'].values.astype(int)
        df = df.drop(columns=['label'])
    df = df.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    if len(df) > MAX_SAMPLES:
        idx = np.random.choice(len(df), MAX_SAMPLES, replace=False)
        df = df.iloc[idx]
        y = y[idx]
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    X = add_fft(X)
    return X, y, df.columns.tolist()

# Load all 4 datasets
print("\n[1] Loading all datasets...")

# WSN_Fault
fault_dfs = []
for fault_name, folder in [
    ('Out_of_bound', os.path.join(FAULT_PATH, 'Out of bound')),
    ('Stuck_at', os.path.join(FAULT_PATH, 'Stuck-at'))]:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.xlsx') and not f.startswith('~'):
                try:
                    df = pd.read_excel(os.path.join(folder, f))
                    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
                    df['label'] = 0 if f.startswith('y') else 1
                    fault_dfs.append(df)
                except: pass

for fault_name, folder in [
    ('Gain', os.path.join(FAULT_PATH, 'Gain')),
    ('Offset', os.path.join(FAULT_PATH, 'Offset'))]:
    if os.path.exists(folder):
        for beta in os.listdir(folder):
            bp = os.path.join(folder, beta)
            if os.path.isdir(bp):
                for f in os.listdir(bp):
                    if f.endswith('.xlsx') and not f.startswith('~'):
                        try:
                            df = pd.read_excel(os.path.join(bp, f))
                            df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
                            df['label'] = 0 if f.startswith('y') else 1
                            fault_dfs.append(df)
                        except: pass

wsn_fault_df = pd.concat(fault_dfs, ignore_index=True).fillna(0)
X_fault, y_fault, feat_fault = load_and_prep(wsn_fault_df)
print(f"  WSN_Fault: {X_fault.shape}")

# Intel Lab
intel_path = os.path.join(BASE_PATH, 'Intel lab data.txt', 'data.txt')
intel_df = pd.read_csv(intel_path, sep=r'\s+', header=None,
    names=['date','time','epoch','moteid','temperature','humidity','light','voltage'],
    on_bad_lines='skip')
intel_df = intel_df[['temperature','humidity','light','voltage']].apply(pd.to_numeric, errors='coerce').dropna()
intel_df['label'] = 0
intel_df.loc[(intel_df['voltage'] < 2.4) | (intel_df['temperature'] > 50) | (intel_df['temperature'] < 0), 'label'] = 1
X_intel, y_intel, feat_intel = load_and_prep(intel_df)
print(f"  Intel_Lab: {X_intel.shape}")

# SWaT
swat_df = pd.concat([
    pd.read_csv(os.path.join(BASE_PATH, 'SWAT', 'normal.csv'), low_memory=False),
    pd.read_csv(os.path.join(BASE_PATH, 'SWAT', 'attack.csv'), low_memory=False)
], ignore_index=True)
for col in swat_df.columns:
    if 'attack' in col.lower() or 'normal' in col.lower():
        swat_df['label'] = swat_df[col].apply(lambda x: 0 if str(x).strip().lower()=='normal' else 1)
        swat_df.drop(columns=[col], inplace=True)
        break
drop_cols = [c for c in swat_df.columns if 'time' in c.lower() or 'date' in c.lower()]
swat_df.drop(columns=drop_cols, errors='ignore', inplace=True)
if len(swat_df) > 50000:
    swat_df = swat_df.sample(50000, random_state=RANDOM_STATE)
X_swat, y_swat, feat_swat = load_and_prep(swat_df)
print(f"  SWaT: {X_swat.shape}")

# WSN-DS
wsn_ds_df = pd.read_csv(os.path.join(BASE_PATH, 'WSN-DS', 'WSN-DS.csv'), low_memory=False)
for col in wsn_ds_df.columns:
    if 'attack' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
        wsn_ds_df['label'] = wsn_ds_df[col].apply(lambda x: 0 if str(x).strip()=='0' or str(x).strip().lower()=='normal' else 1)
        wsn_ds_df.drop(columns=[col], inplace=True)
        break
X_wsn_ds, y_wsn_ds, feat_wsn_ds = load_and_prep(wsn_ds_df)
print(f"  WSN_DS: {X_wsn_ds.shape}")

datasets = {
    'WSN_Fault': (X_fault, y_fault, feat_fault),
    'Intel_Lab': (X_intel, y_intel, feat_intel),
    'SWaT':      (X_swat, y_swat, feat_swat),
    'WSN_DS':    (X_wsn_ds, y_wsn_ds, feat_wsn_ds),
}

colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

# =============================================================================
# TRAIN + SHAP PER DATASET
# =============================================================================

print("\n[2] Training models and computing SHAP values...")

all_shap = {}

for ds_name, (X, y, feat_cols) in datasets.items():
    print(f"\n  --- {ds_name} ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    try:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    except: pass

    print(f"  Training model...")
    model = train_model(X_train, y_train, input_size=X_train.shape[1])
    model.eval()

    # SHAP — use small background + small test set
    bg = X_train[:50]
    test_sample = X_test[:XAI_SAMPLES]

    def model_predict(x):
        with torch.no_grad():
            out = model(torch.FloatTensor(x))
            return torch.softmax(out, dim=1).numpy()

    print(f"  Computing SHAP values (this may take a few minutes)...")
    explainer = shap.KernelExplainer(model_predict, bg)
    shap_values = explainer.shap_values(test_sample, nsamples=50)

    # shap_values[1] = anomaly class
    sv = np.array(shap_values[1])
    all_shap[ds_name] = {
        'shap_values': sv,
        'X_test': test_sample,
        'feature_names': feat_cols[:X.shape[1]//2] if len(feat_cols) >= X.shape[1]//2 else feat_cols,
        'color': colors[list(datasets.keys()).index(ds_name)]
    }
    print(f"  SHAP done for {ds_name}!")

# =============================================================================
# FIGURE XAI-1: SHAP Feature Importance Bar Chart (All Datasets)
# =============================================================================

print("\n[3] Generating XAI Figures...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Figure XAI-1: SHAP Feature Importance — All Datasets',
             fontsize=13, fontweight='bold')

for ax, (ds_name, shap_data) in zip(axes.flat, all_shap.items()):
    sv = shap_data['shap_values']
    feat = shap_data['feature_names']
    color = shap_data['color']

    # Mean absolute SHAP per feature
    mean_shap = np.abs(sv).mean(axis=0)
    n_feat = min(10, len(feat), len(mean_shap))
    top_idx = np.argsort(mean_shap)[-n_feat:][::-1]
    top_feat = [feat[i] if i < len(feat) else f'F{i}' for i in top_idx]
    top_vals = mean_shap[top_idx]

    bars = ax.barh(top_feat[::-1], top_vals[::-1], color=color, edgecolor='black')
    ax.set_title(ds_name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_ylabel('Feature')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'FigureXAI1_SHAP_Feature_Importance.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  FigureXAI1 saved!")

# =============================================================================
# FIGURE XAI-2: SHAP Summary Dot Plot (Best Dataset — WSN_DS)
# =============================================================================

best_ds = 'WSN_DS'
shap_data = all_shap[best_ds]
sv = shap_data['shap_values']
X_s = shap_data['X_test']
feat = shap_data['feature_names']

n_feat = min(10, sv.shape[1], len(feat))
mean_shap = np.abs(sv).mean(axis=0)
top_idx = np.argsort(mean_shap)[-n_feat:][::-1]
top_feat = [feat[i] if i < len(feat) else f'F{i}' for i in top_idx]

fig, ax = plt.subplots(figsize=(10, 7))
for i, (fi, fname) in enumerate(zip(top_idx, top_feat)):
    sv_col = sv[:, fi]
    feat_col = X_s[:, fi]
    feat_norm = (feat_col - feat_col.min()) / (feat_col.max() - feat_col.min() + 1e-8)
    sc = ax.scatter(sv_col, [i] * len(sv_col),
                    c=feat_norm, cmap='coolwarm', alpha=0.6, s=20)

ax.set_yticks(range(n_feat))
ax.set_yticklabels(top_feat)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('SHAP Value (Impact on Anomaly Detection)')
ax.set_title(f'Figure XAI-2: SHAP Summary Plot — {best_ds}',
             fontsize=13, fontweight='bold')
plt.colorbar(sc, ax=ax, label='Feature Value (Low → High)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'FigureXAI2_SHAP_Summary_Plot.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  FigureXAI2 saved!")

# =============================================================================
# FIGURE XAI-3: SHAP Heatmap (WSN_Fault)
# =============================================================================

best2 = 'WSN_Fault'
shap_data2 = all_shap[best2]
sv2 = shap_data2['shap_values']
feat2 = shap_data2['feature_names']

n_feat2 = min(10, sv2.shape[1], len(feat2))
mean_shap2 = np.abs(sv2).mean(axis=0)
top_idx2 = np.argsort(mean_shap2)[-n_feat2:][::-1]
top_feat2 = [feat2[i] if i < len(feat2) else f'F{i}' for i in top_idx2]

shap_matrix = sv2[:50, top_idx2]

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(shap_matrix.T, cmap='RdBu_r', center=0,
            yticklabels=top_feat2,
            xticklabels=[f'S{i+1}' for i in range(shap_matrix.shape[0])],
            ax=ax, cbar_kws={'label': 'SHAP Value'})
ax.set_title(f'Figure XAI-3: SHAP Heatmap — {best2} (50 Samples)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'FigureXAI3_SHAP_Heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  FigureXAI3 saved!")

# =============================================================================
# FIGURE XAI-4: Mean SHAP Comparison Across Datasets
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(10)
width = 0.2
ds_list = list(all_shap.keys())

for i, (ds_name, shap_data) in enumerate(all_shap.items()):
    sv = shap_data['shap_values']
    mean_shap = np.abs(sv).mean(axis=0)
    top10 = np.sort(mean_shap)[-10:][::-1]
    ax.bar(x + i * width, top10, width, label=ds_name,
           color=colors[i], edgecolor='black', alpha=0.85)

ax.set_xlabel('Top 10 Features (ranked by importance)')
ax.set_ylabel('Mean |SHAP Value|')
ax.set_title('Figure XAI-4: SHAP Feature Importance Comparison Across Datasets',
             fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f'Rank {i+1}' for i in range(10)])
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'FigureXAI4_SHAP_Cross_Dataset.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  FigureXAI4 saved!")

# =============================================================================
# DONE
# =============================================================================

print("\n" + "=" * 60)
print("XAI ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\nXAI Figures saved to: {FIGURES_PATH}")
print("  - FigureXAI1_SHAP_Feature_Importance.png")
print("  - FigureXAI2_SHAP_Summary_Plot.png")
print("  - FigureXAI3_SHAP_Heatmap.png")
print("  - FigureXAI4_SHAP_Cross_Dataset.png")
print("\nTotal new figures: 4")
print("Add these to Section 7 (XAI) of your paper.")
print("=" * 60)
