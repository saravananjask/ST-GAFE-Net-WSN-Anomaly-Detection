# =============================================================================
# WSN Fault Detection — Fixed Loader + Retrain
# Fixes the 1-column issue and reruns only WSN_Fault dataset
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, precision_recall_curve, auc)
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# CONFIG
# =============================================================================
BASE_PATH = r"E:\Journals\Journal 2026\June 26\New WSN 22.6.26\Dataset"
OUTPUT_PATH = r"E:\Journals\Journal 2026\June 26\New WSN 22.6.26\Results"
FAULT_PATH = os.path.join(BASE_PATH, 'Dataset-for-WSN-fault-detection-master')

DEVICE = torch.device('cpu')
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_FOLDS = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_SAMPLES = 100000

os.makedirs(os.path.join(OUTPUT_PATH, 'Tables'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'Figures'), exist_ok=True)

print("=" * 60)
print("WSN Fault Detection — Fixed Loader")
print("=" * 60)

# =============================================================================
# STEP 1: FIXED WSN FAULT LOADER
# =============================================================================

print("\n[1] Loading WSN Fault Detection Dataset (Fixed)...")

fault_folders = {
    'Out_of_bound': os.path.join(FAULT_PATH, 'Out of bound'),
    'Stuck_at': os.path.join(FAULT_PATH, 'Stuck-at')
}

beta_folders = {
    'Gain': os.path.join(FAULT_PATH, 'Gain'),
    'Offset': os.path.join(FAULT_PATH, 'Offset')
}

all_dfs = []

# Load Out of Bound and Stuck-at (direct xlsx files)
for fault_name, folder_path in fault_folders.items():
    if not os.path.exists(folder_path):
        print(f"  Not found: {folder_path}")
        continue
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx'):
            fpath = os.path.join(folder_path, file)
            try:
                df = pd.read_excel(fpath, header=0)
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna(axis=1, how='all')

                # label: files starting with 'y' are ground truth (normal=0)
                # files without 'y' prefix are faulty (anomaly=1)
                if file.startswith('y') or file.startswith('Y'):
                    df['label'] = 0
                else:
                    df['label'] = 1

                df['fault_type'] = fault_name
                all_dfs.append(df)
            except Exception as e:
                print(f"  Error reading {file}: {e}")

# Load Gain and Offset (beta subfolders)
for fault_name, folder_path in beta_folders.items():
    if not os.path.exists(folder_path):
        print(f"  Not found: {folder_path}")
        continue
    for beta_folder in os.listdir(folder_path):
        beta_path = os.path.join(folder_path, beta_folder)
        if os.path.isdir(beta_path):
            for file in os.listdir(beta_path):
                if file.endswith('.xlsx'):
                    fpath = os.path.join(beta_path, file)
                    try:
                        df = pd.read_excel(fpath, header=0)
                        df = df.apply(pd.to_numeric, errors='coerce')
                        df = df.dropna(axis=1, how='all')

                        if file.startswith('y') or file.startswith('Y'):
                            df['label'] = 0
                        else:
                            df['label'] = 1

                        df['fault_type'] = fault_name
                        df['beta'] = beta_folder
                        all_dfs.append(df)
                    except Exception as e:
                        print(f"  Error reading {file}: {e}")

if not all_dfs:
    print("ERROR: No files loaded. Check your dataset path.")
    exit()

# Combine all
combined = pd.concat(all_dfs, ignore_index=True)
print(f"\n  Raw combined shape: {combined.shape}")
print(f"  Columns: {list(combined.columns)}")
print(f"  Label distribution: {dict(Counter(combined['label']))}")

# Drop non-numeric columns except label
drop_cols = ['fault_type', 'beta'] if 'beta' in combined.columns else ['fault_type']
combined.drop(columns=drop_cols, errors='ignore', inplace=True)
combined = combined.apply(pd.to_numeric, errors='coerce')
combined = combined.fillna(0)
combined = combined.replace([np.inf, -np.inf], 0)

print(f"\n  Final shape after cleaning: {combined.shape}")
print(f"  Features: {combined.shape[1] - 1}")
print(f"  Labels: {dict(Counter(combined['label']))}")

# Sample if too large
if len(combined) > MAX_SAMPLES:
    combined = combined.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Sampled to {MAX_SAMPLES} rows")

# =============================================================================
# STEP 2: PREPROCESS
# =============================================================================

print("\n[2] Preprocessing...")

y = combined['label'].values.astype(int)
X_df = combined.drop(columns=['label'])
X_df = X_df.select_dtypes(include=[np.number])

scaler = StandardScaler()
X = scaler.fit_transform(X_df.values)

print(f"  X shape: {X.shape}")
print(f"  Classes: {dict(Counter(y))}")

# =============================================================================
# STEP 3: FFT FEATURES
# =============================================================================

print("\n[3] Adding FFT Features...")
fft_features = np.abs(fft(X, axis=1))[:, :X.shape[1] // 2]
X_fft = np.hstack([X, fft_features])
print(f"  Shape after FFT: {X_fft.shape}")

# =============================================================================
# STEP 4: MODEL
# =============================================================================

class BiGRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRUBlock, self).__init__()
        self.bigru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.bigru(x)
        out = out.squeeze(1)
        return self.norm(out)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = self.W(x)
        h_expand = h.unsqueeze(1).expand(-1, h.size(0), -1)
        h_repeat = h.unsqueeze(0).expand(h.size(0), -1, -1)
        concat = torch.cat([h_expand, h_repeat], dim=-1)
        e = self.leaky_relu(self.a(concat).squeeze(-1))
        alpha = self.softmax(e)
        out = torch.matmul(alpha, h)
        return out


class STGAFENet(nn.Module):
    def __init__(self, input_size, hidden_size=64, gat_out=32, num_classes=2):
        super(STGAFENet, self).__init__()
        self.temporal = BiGRUBlock(input_size, hidden_size)
        self.gat = GraphAttentionLayer(input_size, gat_out)
        fusion_size = hidden_size * 2 + gat_out
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        t_out = self.temporal(x)
        s_out = self.gat(x)
        fused = torch.cat([t_out, s_out], dim=-1)
        return self.classifier(fused)


# =============================================================================
# STEP 5: TRAIN / EVAL FUNCTIONS
# =============================================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = model(X_batch)
            probs = torch.softmax(out, dim=1)[:, 1].numpy()
            preds = out.argmax(dim=1).numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr_val = np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1)

    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except:
        auc_roc = 0.0

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc_roc': auc_roc, 'mcc': mcc, 'fpr': fpr_val,
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob
    }


# =============================================================================
# STEP 6: 5-FOLD CV + TEST
# =============================================================================

print("\n[4] Running 5-Fold CV + Testing...")

X_cv, X_test, y_cv, y_test = train_test_split(
    X_fft, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

try:
    sm = SMOTE(random_state=RANDOM_STATE)
    X_cv, y_cv = sm.fit_resample(X_cv, y_cv)
    print(f"  After SMOTE: {dict(Counter(y_cv))}")
except Exception as e:
    print(f"  SMOTE skipped: {e}")

input_size = X_cv.shape[1]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
fold_histories = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
    print(f"  Fold {fold + 1}/{N_FOLDS}", end=' ')

    X_train, X_val = X_cv[train_idx], X_cv[val_idx]
    y_train, y_val = y_cv[train_idx], y_cv[val_idx]

    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.LongTensor(y_train)
    X_vl_t = torch.FloatTensor(X_val)
    y_vl_t = torch.LongTensor(y_val)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=BATCH_SIZE)

    model = STGAFENet(input_size=input_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        train_losses.append(loss)
        if (epoch + 1) % 10 == 0:
            val_m = evaluate(model, val_loader)
            val_accs.append(val_m['accuracy'])

    val_metrics = evaluate(model, val_loader)
    fold_results.append(val_metrics)
    fold_histories.append({'train_loss': train_losses, 'val_acc': val_accs})
    print(f"-> F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc_roc']:.4f}")

# Final test
X_cv_t = torch.FloatTensor(X_cv)
y_cv_t = torch.LongTensor(y_cv)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

full_loader = DataLoader(TensorDataset(X_cv_t, y_cv_t),
                         batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE)

final_model = STGAFENet(input_size=input_size).to(DEVICE)
final_optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
final_criterion = nn.CrossEntropyLoss()

start_time = time.time()
for epoch in range(EPOCHS):
    train_epoch(final_model, full_loader, final_optimizer, final_criterion)
train_time = time.time() - start_time

start_inf = time.time()
test_metrics = evaluate(final_model, test_loader)
inf_time = (time.time() - start_inf) / len(y_test) * 1000

print(f"\n  TEST -> Acc: {test_metrics['accuracy']:.4f} | "
      f"F1: {test_metrics['f1']:.4f} | "
      f"AUC: {test_metrics['auc_roc']:.4f} | "
      f"MCC: {test_metrics['mcc']:.4f}")

# =============================================================================
# STEP 7: SAVE UPDATED TABLE 4
# =============================================================================

print("\n[5] Saving Updated Table 4...")

metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'mcc', 'fpr']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'MCC', 'FPR']

rows = []
for fold_i, res in enumerate(fold_results, start=1):
    row = {'Fold': f'Fold {fold_i}'}
    for m, ml in zip(metric_cols, metric_labels):
        row[ml] = round(res[m], 4)
    rows.append(row)

mean_row = {'Fold': 'Mean'}
std_row = {'Fold': '± Std'}
for m, ml in zip(metric_cols, metric_labels):
    vals = [r[m] for r in fold_results]
    mean_row[ml] = round(np.mean(vals), 4)
    std_row[ml] = round(np.std(vals), 4)

rows.append(mean_row)
rows.append(std_row)

t4 = pd.DataFrame(rows)
t4.to_csv(os.path.join(OUTPUT_PATH, 'Tables', 'Table4_5Fold_CV_WSN_Fault_FIXED.csv'), index=False)
print("  Table 4 (Fixed) saved!")
print(t4.to_string(index=False))

# =============================================================================
# STEP 8: SAVE UPDATED FIGURES
# =============================================================================

print("\n[6] Saving Updated Figures...")

figures_path = os.path.join(OUTPUT_PATH, 'Figures')

# Confusion Matrix
cm = confusion_matrix(test_metrics['y_true'], test_metrics['y_pred'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
ax.set_title('WSN Fault Detection — Confusion Matrix (Fixed)', fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'Figure10_WSN_Fault_CM_Fixed.png'), dpi=150)
plt.close()

# ROC Curve
fpr_r, tpr_r, _ = roc_curve(test_metrics['y_true'], test_metrics['y_prob'])
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_r, tpr_r, color='#2196F3', lw=2,
        label=f"AUC = {test_metrics['auc_roc']:.4f}")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title('WSN Fault Detection — ROC Curve (Fixed)', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'Figure8_WSN_Fault_ROC_Fixed.png'), dpi=150)
plt.close()

# Training Loss
fig, ax = plt.subplots(figsize=(8, 5))
for i, h in enumerate(fold_histories):
    ax.plot(h['train_loss'], alpha=0.5, label=f'Fold {i+1}')
ax.set_title('WSN Fault Detection — Training Loss (Fixed)', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'Figure6_WSN_Fault_Loss_Fixed.png'), dpi=150)
plt.close()

print("  Figures saved!")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("WSN FAULT — FIXED RESULTS")
print("=" * 60)
print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
print(f"  Precision : {test_metrics['precision']:.4f}")
print(f"  Recall    : {test_metrics['recall']:.4f}")
print(f"  F1 Score  : {test_metrics['f1']:.4f}")
print(f"  AUC-ROC   : {test_metrics['auc_roc']:.4f}")
print(f"  MCC       : {test_metrics['mcc']:.4f}")
print(f"  FPR       : {test_metrics['fpr']:.4f}")
print(f"  Train Time: {round(train_time, 2)}s")
print(f"  Inf. Time : {round(inf_time, 4)}ms/sample")
print("=" * 60)
print("COMPLETED!")
