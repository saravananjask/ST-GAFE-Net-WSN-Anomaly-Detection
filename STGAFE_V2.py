# =============================================================================
# ST-GAFE-Net V2  -  revision pipeline (answers Reviewer 1 and Reviewer 2)
#
# What is different from ST_GAFE_Net_FIXED.py
#   1. Real sequences: every sample is a window  X[N nodes, T steps, F features]
#   2. BiGRU runs over the T steps of each node (not over a length-1 sequence)
#   3. FFT is taken ALONG TIME for each node/feature (not across feature columns)
#   4. Real GAT: nodes = sensors/channels, edges = top-k |correlation| computed
#      on TRAINING data only, self-loops, 4 heads, masked softmax attention
#   5. Chronological (blocked) splits with a purge gap, no random sampling
#   6. Scaler / graph fitted on the training part of EACH fold only
#   7. WSN_Fault loader pairs data files with their y-label files
#      (old loader read the y files as data -> trivial result)
#   8. Intel Lab: no ground truth exists, so faults are INJECTED (stuck-at,
#      offset, gain, out-of-bound) with a rule that does not use the label
#      columns; train and test windows are injected with different seeds
#   9. Same windows / splits / epochs for ST-GAFE-Net, ablations and baselines
#
# Usage (from the folder that contains this file):
#   python STGAFE_V2.py --root "F:\June 26\New WSN 22.6.26\Dataset" --out Results_V2 --stage main
#   stages: main | ablation | baselines | stats | timing | all
#   add --quick for a fast smoke test
# =============================================================================
import os, sys, time, json, argparse, warnings, platform, glob, pickle, copy
try:
    import resource                      # Linux / Mac only
except ImportError:
    resource = None

def peak_mem_mb():
    """peak memory of this process in MB (works on Windows with psutil, on Linux/Mac without)"""
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return float('nan')
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix)

p = argparse.ArgumentParser()
p.add_argument('--root', default=r'F:\June 26\New WSN 22.6.26\Dataset')
p.add_argument('--out', default='Results_V2')
p.add_argument('--stage', default='main')
p.add_argument('--datasets', default='WSN_Fault,Intel_Lab,SWaT,SWaT_seen,WSN_DS')
p.add_argument('--rep', default='raw', choices=['raw', 'diff'],
               help="diff = drift-robust input for WSN_Fault / Intel_Lab: (x - mean over sensors, step-to-step change)")
p.add_argument('--seeds', type=int, default=3)
p.add_argument('--epochs', type=int, default=12)
p.add_argument('--T', type=int, default=10)
p.add_argument('--max_train', type=int, default=30000)
p.add_argument('--folds', type=int, default=5)
p.add_argument('--smote', action='store_true', help='SMOTE inside each training fold (flattened windows)')
p.add_argument('--quick', action='store_true')
args = p.parse_args()
if args.quick:
    args.epochs, args.max_train, args.seeds, args.folds = 3, 6000, 1, 3

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEV = 'cpu'
OUT = args.out
for d in ['cache', 'preds', 'tables', 'graphs']:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
DATASETS = args.datasets.split(',')
T_WIN = args.T

# =============================================================================
# DATA LOADERS  ->  dict(Xtr,ytr,ttr,Xte,yte,tte, nodes, purge, extra)
# X shape = (n, N, T, F)  (float32, UNSCALED)
# =============================================================================
def windows(S, lab, tv, start, end, T, stride):
    """S (L,N,F) -> windows inside [start,end). label = any anomaly in window."""
    if end - start < T:
        return (np.zeros((0, S.shape[1], T, S.shape[2]), np.float32), np.zeros(0, int), np.zeros(0))
    w = swv(S[start:end], T, axis=0)                  # (n,N,F,T)
    w = np.ascontiguousarray(w.transpose(0, 1, 3, 2))[::stride]
    lw = swv(lab[start:end], T)[::stride].max(1)
    tw = tv[start:end - T + 1][::stride]
    return w.astype(np.float32), lw.astype(int), tw

def load_wsn_fault(root):
    base = os.path.join(root, 'Dataset-for-WSN-fault-detection-master')
    Xs, ys, ts, ft, bt = [], [], [], [], []
    for ftype, folder in [('Stuck_at', 'Stuck-at'), ('Out_of_bound', 'Out of bound'),
                          ('Gain', 'Gain'), ('Offset', 'Offset')]:
        files = glob.glob(os.path.join(base, folder, '**', '*.xlsx'), recursive=True)
        for f in sorted(files):
            name = os.path.basename(f)
            if name.startswith('y') or name.startswith('~'):
                continue
            # label file = 'y' + data name  (Offset folder uses 'y' + digits only, e.g. y102.xlsx)
            digits = ''.join(ch for ch in name if ch.isdigit() or ch == '.').strip('.')
            cands = [os.path.join(os.path.dirname(f), 'y' + name), os.path.join(os.path.dirname(f), 'y' + digits + '.xlsx')]
            yf = next((c for c in cands if os.path.exists(c)), None)
            if yf is None:
                print(f"  WARNING: no label file for {f} - skipped")
                continue
            d = pd.read_excel(f, header=None).iloc[2:, :12].apply(pd.to_numeric, errors='coerce')
            y = pd.read_excel(yf, header=None).iloc[:, 0].values
            n = min(len(d), len(y))
            d = d.iloc[:n].values.astype(np.float32); y = (y[:n] == -1).astype(int)
            ok = ~np.isnan(d).any(1)
            # row = (t0,t1,t2) x (T1,H1,T2,H2)  ->  (N=2 sensors, T=3, F=2 [temp,hum])
            X = d.reshape(n, 3, 2, 2).transpose(0, 2, 1, 3)
            beta = os.path.basename(os.path.dirname(f)) if 'beta' in os.path.dirname(f) else 'rate'
            Xs.append(X[ok]); ys.append(y[ok]); ts.append(np.arange(n)[ok])
            ft += [ftype] * ok.sum(); bt += [beta] * ok.sum()
    X = np.concatenate(Xs); y = np.concatenate(ys); t = np.concatenate(ts)
    ft = np.array(ft); bt = np.array(bt)
    cut = int(0.8 * t.max())            # same base rows in every file -> split by row position
    tr = t < cut - 3; te = t >= cut
    print(f"  WSN_Fault: {len(X)} rows, fault rate {y.mean():.3f}, train {tr.sum()}, test {te.sum()}")
    return dict(Xtr=X[tr], ytr=y[tr], ttr=t[tr], Xte=X[te], yte=y[te], tte=t[te],
                nodes=['Sensor1', 'Sensor2'], purge=3,
                extra_te=dict(ftype=ft[te], beta=bt[te]))

def load_intel(root, n_nodes=20):
    path = os.path.join(root, 'Intel lab data.txt', 'data.txt')
    df = pd.read_csv(path, sep=r'\s+', header=None, on_bad_lines='skip',
                     names=['date', 'time', 'epoch', 'mote', 'temp', 'hum', 'light', 'volt'])
    df = df[['epoch', 'mote', 'temp', 'hum', 'light', 'volt']].apply(pd.to_numeric, errors='coerce').dropna()
    df = df[(df.mote >= 1) & (df.mote <= 54) & df.temp.between(0, 50) & df.hum.between(0, 100)]
    top = df.mote.value_counts().index[:n_nodes].tolist()
    top.sort()
    df = df[df.mote.isin(top)]
    ep = np.arange(int(df.epoch.min()), int(df.epoch.max()) + 1)
    chans = ['temp', 'hum', 'light', 'volt']
    S = np.zeros((len(ep), len(top), len(chans)), np.float32)
    for j, c in enumerate(chans):
        pv = df.pivot_table(index='epoch', columns='mote', values=c, aggfunc='mean').reindex(ep)[top]
        pv = pv.ffill(limit=10).bfill(limit=10)
        pv = pv.fillna(pv.median())
        S[:, :, j] = pv.values
    lab0 = np.zeros(len(ep), int)
    cut = int(0.8 * len(ep)); T = T_WIN; stride = 3
    sig = S[:cut].std(0) + 1e-6                         # (N,F) from TRAIN part only
    def build(a, b, seed):
        X, _, t = windows(S, lab0, ep, a, b, T, stride)
        rng = np.random.default_rng(seed)
        y = np.zeros(len(X), int)
        for i in range(len(X)):
            if rng.random() < 0.3:
                y[i] = 1
                for nd in rng.choice(X.shape[1], rng.integers(1, 3), replace=False):
                    kind = rng.integers(4); s0 = rng.integers(0, T - 3)
                    for f in (0, 1):
                        seg = X[i, nd, s0:, f]
                        if kind == 0:   seg[:] = seg[0]                                  # stuck-at
                        elif kind == 1: seg += rng.uniform(1, 4) * sig[nd, f]            # offset
                        elif kind == 2: seg *= 1 + rng.uniform(0.2, 1.0)                 # gain
                        else:
                            k = rng.integers(1, 4); seg[:k] = seg[:k] + rng.choice([-1, 1]) * rng.uniform(6, 10) * sig[nd, f]  # out of bound
        return X, y, t
    Xtr, ytr, ttr = build(0, cut - T, 123)
    Xte, yte, tte = build(cut, len(ep), 456)
    print(f"  Intel_Lab: {len(ep)} epochs, {len(top)} motes, train {len(Xtr)}, test {len(Xte)}, fault rate {ytr.mean():.3f}")
    return dict(Xtr=Xtr, ytr=ytr, ttr=ttr, Xte=Xte, yte=yte, tte=tte,
                nodes=[f'Mote{m}' for m in top], purge=T, extra_te={})

def load_swat(root, step=5, mode='heldout'):
    """SWaT: attack.csv holds ONLY the attack-labelled rows (35 separate episodes, real timestamps);
    normal.csv holds normal rows for the same days. Merge both by timestamp, build windows only inside
    continuous 1-second runs (never across glued episodes), and split by TIME:
    train = earlier attack episodes, test = the last 12 unseen attack episodes (+ normal data after the cut)."""
    base = os.path.join(root, 'SWAT')
    ts_l, v_l, y_l = [], [], []
    for fname, yv in [('normal.csv', 0), ('attack.csv', 1)]:
        for ch in pd.read_csv(os.path.join(base, fname), chunksize=200000, low_memory=False):
            t = pd.to_datetime(ch.iloc[:, 0].astype(str).str.strip(), format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
            ok = t.notna().values
            v = ch.iloc[:, 1:52].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
            ts_l.append(t.values.astype('datetime64[s]').astype(np.int64)[ok]); v_l.append(v[ok])
            y_l.append(np.full(ok.sum(), yv, np.int8))
    t = np.concatenate(ts_l); v = np.concatenate(v_l); y = np.concatenate(y_l)
    del ts_l, v_l, y_l
    order = np.lexsort((-y.astype(int), t))                      # same timestamp -> keep the attack row
    t, v, y = t[order], v[order], y[order]
    keep = np.r_[True, np.diff(t) != 0]
    t, v, y = t[keep], v[keep], y[keep]
    # attack episodes (from the attack rows only)
    ta = t[y == 1]; epi = np.r_[0, np.cumsum(np.diff(ta) > 1)]
    n_ep = int(epi.max()) + 1
    test_n = max(3, round(0.35 * n_ep))
    cut = int(ta[np.searchsorted(epi, n_ep - test_n)])           # start time of the first TEST episode
    # continuous runs (gap > 1 s breaks a run), keep only complete blocks of `step` rows
    run = np.r_[0, np.cumsum(np.diff(t) > 1)]
    starts = np.r_[0, np.where(np.diff(run) != 0)[0] + 1]
    pos = np.arange(len(t)) - starts[run]
    rlen = np.bincount(run)[run]
    kp = pos < (rlen // step) * step
    t, v, y, run = t[kp], v[kp], y[kp], run[kp]
    S = v.reshape(-1, step, v.shape[1]).mean(1)[:, :, None]
    lab = y.reshape(-1, step).max(1).astype(int); tb = t[::step]; rb = run[::step]
    T = T_WIN
    Xtr, ytr, ttr, Xte, yte, tte = [], [], [], [], [], []
    bounds = np.r_[0, np.where(np.diff(rb) != 0)[0] + 1, len(rb)]
    def add(lst, w, l, tw):
        lst[0].append(w); lst[1].append(l); lst[2].append(tw)
    TR, TE = ([], [], []), ([], [], [])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < T: continue
        if mode == 'heldout':                                   # unseen attack episodes, split by time
            w, l, tw = windows(S, lab, tb, a, b, T, 3)
            m_tr = tw + T * step < cut; m_te = tw >= cut         # windows that straddle the cut are dropped
            add(TR, w[m_tr], l[m_tr], tw[m_tr]); add(TE, w[m_te], l[m_te], tw[m_te])
        else:                                                   # 'seen': first 70% / last 30% of every constant-label segment
            lr = lab[a:b]; ch = np.r_[0, np.where(np.diff(lr) != 0)[0] + 1, len(lr)]
            for c0, c1 in zip(ch[:-1], ch[1:]):
                s0, s1 = a + c0, a + c1
                if s1 - s0 < 4 * T: continue
                sp = s0 + int(0.7 * (s1 - s0))
                add(TR, *windows(S, lab, tb, s0, sp, T, 3))
                add(TE, *windows(S, lab, tb, sp + T, s1, T, 3))   # gap of T steps between train and test
    Xtr, ytr, ttr = TR; Xte, yte, tte = TE
    cat = np.concatenate
    d = dict(Xtr=cat(Xtr), ytr=cat(ytr), ttr=cat(ttr), Xte=cat(Xte), yte=cat(yte), tte=cat(tte),
             nodes=[f'ch{i}' for i in range(S.shape[1])], purge=T * step, extra_te={})
    print(f"  SWaT: {len(S)} steps in {len(bounds)-1} continuous runs, {n_ep} attack episodes "
          f"(mode={mode}; {n_ep-test_n} train / {test_n} test episodes in heldout mode); train {len(d['Xtr'])} (rate {d['ytr'].mean():.3f}), "
          f"test {len(d['Xte'])} (rate {d['yte'].mean():.3f})")
    return d

def load_wsn_ds(root):
    df = pd.read_csv(os.path.join(root, 'WSN-DS', 'WSN-DS.csv'), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    atk = df['Attack type'].astype(str).str.strip()
    feats = [c for c in df.columns if c not in ('id', 'Time', 'who CH', 'Attack type')]
    A = df[feats].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    order = np.lexsort((df['Time'].values, df['id'].values))
    A, ids, tm, atk = A[order], df['id'].values[order], df['Time'].values[order], atk.values[order]
    T = 5
    n = len(A) - T + 1
    ok = ids[:n] == ids[T - 1:T - 1 + n]
    W = swv(A, T, axis=0)[:n][ok][..., None]                                 # (n,15,T,1)
    last = np.arange(T - 1, T - 1 + n)[ok]; first = np.arange(n)[ok]
    y = (atk[last] != 'Normal').astype(int); ty = atk[last]
    # WSN-DS is a simulated per-node record set whose attacks are not spread evenly in time,
    # so a chronological split would measure scenario shift. Use a NODE-DISJOINT split instead:
    # all windows of one sensor node go to train or to test (no overlap leakage between windows).
    nid = ids[last]
    uniq = np.unique(nid); perm = np.random.default_rng(0).permutation(len(uniq))
    rank = np.empty(len(uniq), int); rank[perm] = np.arange(len(uniq))
    grp = rank[np.searchsorted(uniq, nid)]                 # shuffled node rank = "group time" for blocked folds
    te = grp >= int(0.8 * len(uniq)); tr = ~te
    print(f"  WSN_DS: {len(W)} windows, {len(uniq)} nodes, train {tr.sum()} (rate {y[tr].mean():.3f}), test {te.sum()} (rate {y[te].mean():.3f})")
    return dict(Xtr=W[tr], ytr=y[tr], ttr=grp[tr], Xte=W[te], yte=y[te], tte=grp[te],
                nodes=feats, purge=0, extra_te=dict(ftype=ty[te]))

LOADERS = dict(WSN_Fault=load_wsn_fault, Intel_Lab=load_intel, SWaT=load_swat, WSN_DS=load_wsn_ds,
               SWaT_seen=lambda r: load_swat(r, mode='seen'))

def to_rep(d, name):
    """optional drift-robust representation (only for datasets whose nodes measure the same quantities)"""
    if args.rep == 'diff' and name in ('WSN_Fault', 'Intel_Lab'):
        def f(X):
            cm = X - X.mean(1, keepdims=True)                         # remove level shared by all sensors
            dx = np.diff(X, axis=2, prepend=X[:, :, :1])              # step-to-step change
            return np.concatenate([cm, dx], -1).astype(np.float32)
        d = dict(d); d['Xtr'] = f(d['Xtr']); d['Xte'] = f(d['Xte'])
    return d

def get_data(name):
    f = os.path.join(OUT, 'cache', f'{name}_T{T_WIN}_v3.pkl')      # v3 = timestamp-based SWaT loader
    if os.path.exists(f) and not args.quick:
        return to_rep(pickle.load(open(f, 'rb')), name)
    print(f"Loading {name} ...")
    d = LOADERS[name](args.root)
    d['T'] = d['Xtr'].shape[2]
    if not args.quick:
        pickle.dump(d, open(f, 'wb'))
    return to_rep(d, name)

# =============================================================================
# PREPROCESSING INSIDE A FOLD (fit on train part only)
# =============================================================================
def cap(idx, y, n, rng):
    if len(idx) <= n:
        return idx
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    kp = min(len(pos), int(n * len(pos) / len(idx)) + 1)
    kn = n - kp
    return np.sort(np.concatenate([rng.choice(pos, kp, replace=False), rng.choice(neg, min(kn, len(neg)), replace=False)]))

def fit_scaler(X):
    m = X.mean((0, 2), keepdims=True); s = X.std((0, 2), keepdims=True)
    return m, np.where(s < 1e-6, 1.0, s)

def apply_scaler(X, sc):
    return np.clip((X - sc[0]) / sc[1], -10, 10).astype(np.float32)

def build_graph(Xs, k=3):
    """top-k |corr| between node signals (feature 0), TRAIN only. static, undirected, self-loops."""
    N = Xs.shape[1]
    sig = Xs[:20000, :, :, 0].transpose(1, 0, 2).reshape(N, -1)
    c = np.nan_to_num(np.abs(np.corrcoef(sig)))
    np.fill_diagonal(c, 0)
    A = np.zeros((N, N), bool)
    for i in range(N):
        A[i, np.argsort(-c[i])[:min(k, N - 1)]] = True
    A = A | A.T | np.eye(N, dtype=bool)
    return A

def smote_windows(X, y, seed):
    from imblearn.over_sampling import SMOTE
    sh = X.shape
    Xr, yr = SMOTE(random_state=seed).fit_resample(X.reshape(len(X), -1), y)
    return Xr.reshape(-1, *sh[1:]).astype(np.float32), yr

# =============================================================================
# MODELS
# =============================================================================
class GATLayer(nn.Module):
    def __init__(s, d, heads=4, drop=0.1):
        super().__init__()
        s.h, s.dk, s.drop = heads, d // heads, drop
        s.W = nn.Linear(d, d, bias=False)
        s.a_src = nn.Parameter(torch.randn(heads, s.dk) * 0.1)
        s.a_dst = nn.Parameter(torch.randn(heads, s.dk) * 0.1)
        s.norm = nn.LayerNorm(d); s.keep = False; s.last = None
    def forward(s, x, adj):
        B, N, _ = x.shape
        h = s.W(x).view(B, N, s.h, s.dk)
        e = F.leaky_relu((h * s.a_src).sum(-1).unsqueeze(2) + (h * s.a_dst).sum(-1).unsqueeze(1), 0.2)  # B,i,j,h
        e = e.masked_fill(~s.adj[None, :, :, None], float('-inf'))
        a = torch.softmax(e, dim=2)
        if s.keep: s.last = a.detach()
        a = F.dropout(a, s.drop, s.training)
        out = torch.einsum('bijh,bjhd->bihd', a, h).reshape(B, N, -1)
        return s.norm(x + F.elu(out))

class STGAFE(nn.Module):
    def __init__(s, N, T, Fd, adj, use_t=True, use_f=True, use_g=True, hid=32, d=64, heads=4):
        super().__init__()
        s.use_t, s.use_f, s.use_g = use_t, use_f, use_g
        dim = 0
        if use_t: s.gru = nn.GRU(Fd, hid, batch_first=True, bidirectional=True); dim += 2 * hid
        if use_f: s.fft_fc = nn.Sequential(nn.Linear(Fd * (T // 2 + 1), hid), nn.ReLU()); dim += hid
        s.raw = (not use_t) and (not use_f)
        if s.raw: s.raw_fc = nn.Sequential(nn.Linear(T * Fd, 2 * hid), nn.ReLU()); dim += 2 * hid
        s.proj = nn.Sequential(nn.Linear(dim, d), nn.ReLU(), nn.Dropout(0.2))
        s.node_emb = nn.Parameter(torch.zeros(N, d))
        s.gats = nn.ModuleList([GATLayer(d, heads) for _ in range(2)]) if use_g else None
        a = torch.tensor(adj, dtype=torch.bool)
        if use_g:
            for g in s.gats: g.register_buffer('adj', a)
        s.head = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
    def embed(s, x):
        B, N, T, Fd = x.shape
        f = []
        if s.use_t:
            _, h = s.gru(x.reshape(B * N, T, Fd)); f.append(torch.cat([h[0], h[1]], -1).view(B, N, -1))
        if s.use_f:
            sp = torch.log1p(torch.fft.rfft(x, dim=2).abs())          # FFT along TIME  (B,N,T//2+1,F)
            f.append(s.fft_fc(sp.reshape(B, N, -1)))
        if s.raw: f.append(s.raw_fc(x.reshape(B, N, -1)))
        z = s.proj(torch.cat(f, -1)) + s.node_emb
        if s.use_g:
            for g in s.gats: z = g(z, g.adj)
        return z
    def forward(s, x):
        z = s.embed(x)
        return s.head(torch.cat([z.mean(1), z.max(1).values], -1))

class SeqBaseline(nn.Module):
    """LSTM / GRU / CNN-LSTM / Transformer over the T steps (all nodes flattened per step) and GCN."""
    def __init__(s, kind, N, T, Fd, adj, hid=64):
        super().__init__()
        s.kind, D = kind, N * Fd
        if kind == 'LSTM': s.rnn = nn.LSTM(D, hid, batch_first=True, bidirectional=True); o = 2 * hid
        elif kind == 'GRU': s.rnn = nn.GRU(D, hid, batch_first=True, bidirectional=True); o = 2 * hid
        elif kind == 'CNN-LSTM':
            s.conv = nn.Conv1d(D, hid, 3, padding=1); s.rnn = nn.LSTM(hid, hid, batch_first=True, bidirectional=True); o = 2 * hid
        elif kind == 'Transformer':
            s.emb = nn.Linear(D, 32); s.pos = nn.Parameter(torch.zeros(1, T, 32))
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 4, 64, 0.1, batch_first=True), 1); o = 32
        elif kind == 'GCN':
            s.gru = nn.GRU(Fd, 32, batch_first=True, bidirectional=True)
            A = torch.tensor(adj, dtype=torch.float32); dg = A.sum(1)
            s.register_buffer('An', A / torch.sqrt(dg[:, None] * dg[None, :]))
            s.g1 = nn.Linear(64, 64); s.g2 = nn.Linear(64, 64); o = 128
        s.head = nn.Sequential(nn.Linear(o, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
    def forward(s, x):
        B, N, T, Fd = x.shape
        if s.kind == 'GCN':
            _, h = s.gru(x.reshape(B * N, T, Fd)); z = torch.cat([h[0], h[1]], -1).view(B, N, -1)
            z = F.relu(s.g1(torch.einsum('ij,bjd->bid', s.An, z)))
            z = F.relu(s.g2(torch.einsum('ij,bjd->bid', s.An, z)))
            return s.head(torch.cat([z.mean(1), z.max(1).values], -1))
        seq = x.permute(0, 2, 1, 3).reshape(B, T, N * Fd)
        if s.kind in ('LSTM', 'GRU'): _, h = s.rnn(seq); h = h[0] if s.kind == 'LSTM' else h; r = torch.cat([h[0], h[1]], -1)
        elif s.kind == 'CNN-LSTM':
            c = F.relu(s.conv(seq.transpose(1, 2))).transpose(1, 2); _, (h, _) = s.rnn(c); r = torch.cat([h[0], h[1]], -1)
        else: r = s.enc(s.emb(seq) + s.pos).mean(1)
        return s.head(r)

# =============================================================================
# TRAIN / PREDICT / METRICS
# =============================================================================
def metrics(y, prob, thr=0.5):
    pr = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    return dict(Accuracy=accuracy_score(y, pr), Precision=precision_score(y, pr, zero_division=0),
                Recall=recall_score(y, pr, zero_division=0), F1=f1_score(y, pr, zero_division=0),
                AUC=auc, MCC=matthews_corrcoef(y, pr), FPR=fp / max(fp + tn, 1), TN=tn, FP=fp, FN=fn, TP=tp)

def best_thr(yv, pv):
    """decision threshold chosen on the VALIDATION part of the training data (never on test).
    Scores of a rare class are often tiny, so a fixed 0.5 cut-off is unreliable under drift."""
    if len(np.unique(yv)) < 2:
        return 0.5, 0.0
    grid = np.unique(np.r_[np.logspace(-4, np.log10(0.9999), 80), 0.5])
    f = [f1_score(yv, (pv >= g).astype(int), zero_division=0) for g in grid]
    i = int(np.argmax(f)); return float(grid[i]), float(f[i])

def predict(model, X, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + bs])), 1)[:, 1].numpy())
    return np.concatenate(out)

def train_model(model, Xtr, ytr, seed, Xval=None, yval=None, epochs=None, bs=256, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    epochs = epochs or args.epochs
    w = torch.tensor([1.0, float(max(1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1)))],
                     dtype=torch.float32).clamp(max=10.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, max(1, epochs // 3), 0.5)
    lossf = nn.CrossEntropyLoss(weight=w)
    best, best_state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(Xtr)); tot = 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2: continue
            opt.zero_grad()
            loss = lossf(model(torch.from_numpy(Xtr[b])), torch.from_numpy(ytr[b]))
            loss.backward(); opt.step(); tot += loss.item() * len(b)
        sch.step()
        vf = np.nan
        if Xval is not None:
            vp = predict(model, Xval); vf = best_thr(yval, vp)[1]
            if vf > best: best, best_state = vf, copy.deepcopy(model.state_dict())
        hist.append((ep + 1, tot / len(Xtr), vf))
    if best_state is not None: model.load_state_dict(best_state)
    return hist

def prepare(d, tr_idx, seed, use_smote=False):
    """scaler + graph fitted on tr_idx only. returns scaled train, scaler, adj"""
    Xr = d['Xtr'][tr_idx]; sc = fit_scaler(Xr); Xs = apply_scaler(Xr, sc)
    adj = build_graph(Xs); ys = d['ytr'][tr_idx]
    if use_smote: Xs, ys = smote_windows(Xs, ys, seed)
    return Xs, ys, sc, adj

def make_model(kind, d, adj):
    N, T, Fd = d['Xtr'].shape[1:]
    V = dict(full=(1, 1, 1), no_fft=(1, 0, 1), no_graph=(1, 1, 0), temporal_only=(1, 0, 0),
             fft_only=(0, 1, 0), graph_only=(0, 0, 1), temporal_fft_graph=(1, 1, 1))
    if kind in V: return STGAFE(N, T, Fd, adj, *[bool(v) for v in V[kind]])
    return SeqBaseline(kind, N, T, Fd, adj)

def inner_split(d, idx):
    """last 10% (by time) of the training part = early-stopping set, purged from training"""
    t = d['ttr'][idx]; q = np.quantile(t, 0.9)
    return idx[t < q - d['purge']], idx[t >= q]

def run_final(d, kind, seed, name):
    rng = np.random.default_rng(seed)
    idx = cap(np.arange(len(d['ytr'])), d['ytr'], args.max_train, rng)
    tr, va = inner_split(d, idx)
    Xs, ys, sc, adj = prepare(d, tr, seed, args.smote)
    model = make_model(kind, d, adj)
    Xv = apply_scaler(d['Xtr'][va], sc); yv = d['ytr'][va]
    t0 = time.time(); hist = train_model(model, Xs, ys, seed, Xv, yv); tt = time.time() - t0
    Xt = apply_scaler(d['Xte'], sc); pt = predict(model, Xt)
    thr = best_thr(yv, predict(model, Xv))[0]
    m = metrics(d['yte'], pt, thr); m.update(dataset=name, model=kind, seed=seed, thr=thr, F1_at_0p5=metrics(d['yte'], pt)['F1'],
                                              train_sec=round(tt, 1), params=sum(q.numel() for q in model.parameters()))
    np.savez_compressed(os.path.join(OUT, 'preds', f'{name}_{kind}_seed{seed}.npz'), y=d['yte'], p=pt, thr=thr)
    return m, model, sc, adj, hist

def run_cv(d, name, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(d['ytr'])); t = d['ttr']
    q = np.quantile(t, np.linspace(0, 1, args.folds + 1)); res = []
    for k in range(args.folds):
        lo, hi = q[k], q[k + 1] + (1e-9 if k == args.folds - 1 else 0)
        va = idx[(t >= lo) & (t < hi)]
        tr = idx[(t < lo - d['purge']) | (t >= hi + d['purge'])]
        if len(va) == 0 or len(np.unique(d['ytr'][va])) < 2 or len(tr) == 0:
            continue
        tr = cap(tr, d['ytr'], args.max_train, rng)
        tr1, va1 = inner_split(d, tr)                       # threshold is chosen on the training side only
        Xs, ys, sc, adj = prepare(d, tr1, seed + k, args.smote)
        model = make_model('full', d, adj)
        train_model(model, Xs, ys, seed + k, apply_scaler(d['Xtr'][va1], sc), d['ytr'][va1])
        thr = best_thr(d['ytr'][va1], predict(model, apply_scaler(d['Xtr'][va1], sc)))[0]
        m = metrics(d['ytr'][va], predict(model, apply_scaler(d['Xtr'][va], sc)), thr)
        m.update(dataset=name, fold=k + 1, seed=seed); res.append(m)
        print(f"    fold {k+1}: F1 {m['F1']:.4f}")
    return res

def load_rows(fname):
    """checkpoint: rows already finished by an earlier (stopped) run. Not used in --quick mode."""
    f = os.path.join(OUT, 'tables', fname)
    if args.quick or not os.path.exists(f): return []
    try: return pd.read_csv(f).to_dict('records')
    except Exception: return []

def have(rows, **kw):
    return any(all(r.get(k) == v for k, v in kw.items()) for r in rows)

def save_csv(rows, fname):
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'tables', fname), index=False)

def summarize(df, keys, cols=('Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC', 'FPR')):
    g = df.groupby(keys)[list(cols)].agg(['mean', 'std'])
    g.columns = [f'{a}_{b}' for a, b in g.columns]
    return g.reset_index()

# =============================================================================
# STAGES
# =============================================================================
def stage_main():
    final, cvrows, per_type = load_rows('final_test_by_seed.csv'), load_rows('cv_by_fold.csv'), load_rows('per_attack_type_recall.csv')
    for name in DATASETS:
        d = get_data(name)
        for s in range(args.seeds):
            if have(final, dataset=name, seed=s):
                print(f"[main] {name} seed {s}: already done, skipping"); continue
            print(f"[main] {name} seed {s}")
            cvrows += run_cv(d, name, s)
            m, model, sc, adj, hist = run_final(d, 'full', s, name)
            print(f"    TEST F1 {m['F1']:.4f}  AUC {m['AUC']:.4f}  MCC {m['MCC']:.4f}  FPR {m['FPR']:.4f}")
            final.append(m)
            save_csv(final, 'final_test_by_seed.csv'); save_csv(cvrows, 'cv_by_fold.csv')      # checkpoint after every seed
            if s == 0:
                pd.DataFrame(hist, columns=['epoch', 'train_loss', 'val_F1']).to_csv(
                    os.path.join(OUT, 'tables', f'history_{name}.csv'), index=False)
                pd.DataFrame(adj.astype(int), index=d['nodes'], columns=d['nodes']).to_csv(
                    os.path.join(OUT, 'graphs', f'adjacency_{name}.csv'))
                if d['extra_te'].get('ftype') is not None:
                    pt = np.load(os.path.join(OUT, 'preds', f'{name}_full_seed0.npz'))['p']
                    ex = pd.DataFrame(dict(ftype=d['extra_te']['ftype'], y=d['yte'], pred=(pt >= float(np.load(os.path.join(OUT, 'preds', f'{name}_full_seed0.npz'))['thr'])).astype(int)))
                    for ft, g in ex[ex.y == 1].groupby('ftype'):
                        per_type.append(dict(dataset=name, type=ft, n=len(g), recall=g.pred.mean()))
                save_csv(per_type, 'per_attack_type_recall.csv') if per_type else None
        save_csv(final, 'final_test_by_seed.csv'); save_csv(cvrows, 'cv_by_fold.csv')
    df = pd.DataFrame(final); summarize(df, ['dataset']).to_csv(os.path.join(OUT, 'tables', 'final_test_summary.csv'), index=False)
    if cvrows: summarize(pd.DataFrame(cvrows), ['dataset']).to_csv(os.path.join(OUT, 'tables', 'cv_summary.csv'), index=False)
    if per_type: save_csv(per_type, 'per_attack_type_recall.csv')
    print(summarize(df, ['dataset'], ('F1', 'AUC', 'MCC', 'FPR')).round(4).to_string())

def stage_ablation():
    rows = load_rows('ablation_by_seed.csv')
    for name in DATASETS:
        d = get_data(name)
        for kind in ['full', 'no_fft', 'no_graph', 'temporal_only', 'fft_only', 'graph_only']:
            for s in range(args.seeds):
                if have(rows, dataset=name, model=kind, seed=s): continue
                m = run_final(d, kind, s, name)[0]; rows.append(m)
                print(f"[ablation] {name} {kind} seed {s}: F1 {m['F1']:.4f}")
                save_csv(rows, 'ablation_by_seed.csv')
        save_csv(rows, 'ablation_by_seed.csv')
    summarize(pd.DataFrame(rows), ['dataset', 'model']).to_csv(os.path.join(OUT, 'tables', 'ablation_summary.csv'), index=False)

def stage_baselines():
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    rows = load_rows('baselines_by_seed.csv')
    ALLB = ['RF', 'RF_fft', 'HGB', 'MLP', 'MLP_fft', 'LR', 'LSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'GCN']
    for name in DATASETS:
        d = get_data(name)
        for s in range(args.seeds):
            if all(have(rows, dataset=name, model=k, seed=s) for k in ALLB): continue
            rng = np.random.default_rng(s)
            idx = cap(np.arange(len(d['ytr'])), d['ytr'], args.max_train, rng)
            tr, va = inner_split(d, idx)                    # same split and threshold rule as ST-GAFE-Net
            Xs, ys, sc, adj = prepare(d, tr, s, args.smote)
            Xt = apply_scaler(d['Xte'], sc); Xv = apply_scaler(d['Xtr'][va], sc); yv = d['ytr'][va]
            fl = lambda X: X.reshape(len(X), -1)
            ff = lambda X: np.hstack([fl(X), np.abs(np.fft.rfft(X, axis=2)).reshape(len(X), -1)])  # same FFT-along-time features
            skl = {'RF': (RandomForestClassifier(200, n_jobs=-1, class_weight='balanced', random_state=s), fl),
                   'RF_fft': (RandomForestClassifier(200, n_jobs=-1, class_weight='balanced', random_state=s), ff),
                   'HGB': (HistGradientBoostingClassifier(random_state=s), fl),
                   'MLP': (MLPClassifier((128, 64), early_stopping=True, random_state=s, max_iter=60), fl),
                   'MLP_fft': (MLPClassifier((128, 64), early_stopping=True, random_state=s, max_iter=60), ff),
                   'LR': (LogisticRegression(max_iter=500, class_weight='balanced'), fl)}
            for k, (mdl, tf) in skl.items():
                if have(rows, dataset=name, model=k, seed=s): continue
                mdl.fit(tf(Xs), ys); pt = mdl.predict_proba(tf(Xt))[:, 1]
                thr = best_thr(yv, mdl.predict_proba(tf(Xv))[:, 1])[0]
                m = metrics(d['yte'], pt, thr); m.update(dataset=name, model=k, seed=s, thr=thr, F1_at_0p5=metrics(d['yte'], pt)['F1']); rows.append(m)
                np.savez_compressed(os.path.join(OUT, 'preds', f'{name}_{k}_seed{s}.npz'), y=d['yte'], p=pt, thr=thr)
                print(f"[baseline] {name} {k} seed {s}: F1 {m['F1']:.4f}"); save_csv(rows, 'baselines_by_seed.csv')
            for k in ['LSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'GCN']:
                if have(rows, dataset=name, model=k, seed=s): continue
                m = run_final(d, k, s, name)[0]; rows.append(m)
                print(f"[baseline] {name} {k} seed {s}: F1 {m['F1']:.4f}"); save_csv(rows, 'baselines_by_seed.csv')
        save_csv(rows, 'baselines_by_seed.csv')
    summarize(pd.DataFrame(rows), ['dataset', 'model']).to_csv(os.path.join(OUT, 'tables', 'baselines_summary.csv'), index=False)

def stage_stats(B=1000):
    from scipy.stats import binomtest
    rng = np.random.default_rng(0); rows, ci = [], []
    def load(n, k):
        f = os.path.join(OUT, 'preds', f'{n}_{k}_seed0.npz')
        return np.load(f) if os.path.exists(f) else None
    for name in DATASETS:
        ours = load(name, 'full')
        if ours is None: continue
        y = ours['y']
        for k in ['full', 'RF', 'RF_fft', 'HGB', 'MLP', 'MLP_fft', 'LR', 'LSTM', 'GRU', 'CNN-LSTM', 'Transformer', 'GCN']:
            o = load(name, k)
            if o is None: continue
            bs = []
            for _ in range(B):
                i = rng.integers(0, len(y), len(y)); yy = y[i]
                if len(np.unique(yy)) < 2: continue
                pr = (o['p'][i] >= float(o['thr'])).astype(int)
                bs.append((f1_score(yy, pr, zero_division=0), roc_auc_score(yy, o['p'][i]), matthews_corrcoef(yy, pr)))
            bs = np.array(bs); lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
            ci.append(dict(dataset=name, model=k, F1_lo=lo[0], F1_hi=hi[0], AUC_lo=lo[1], AUC_hi=hi[1], MCC_lo=lo[2], MCC_hi=hi[2]))
            if k == 'full': continue
            ok_o = ((ours['p'] >= float(ours['thr'])).astype(int) == y); ok_b = ((o['p'] >= float(o['thr'])).astype(int) == y)
            n10 = int((ok_o & ~ok_b).sum()); n01 = int((~ok_o & ok_b).sum())   # n10: ours right, baseline wrong
            pv = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1.0     # exact two-sided McNemar
            rows.append(dict(dataset=name, baseline=k, n10_ours_correct_only=n10, n01_baseline_correct_only=n01,
                             p_exact=pv, direction='ST-GAFE-Net better' if n10 > n01 else ('baseline better' if n01 > n10 else 'tie')))
    df = pd.DataFrame(rows)
    if len(df):                                           # Holm-Bonferroni over all comparisons
        order = df.p_exact.argsort().values; m = len(df); adj = np.zeros(m); run = 0
        for r, i in enumerate(order):
            run = max(run, (m - r) * df.p_exact.iloc[i]); adj[i] = min(1, run)
        df['p_holm'] = adj; df['significant_holm_0.05'] = df.p_holm < 0.05
    df.to_csv(os.path.join(OUT, 'tables', 'mcnemar_exact_holm.csv'), index=False)
    pd.DataFrame(ci).to_csv(os.path.join(OUT, 'tables', 'bootstrap_CI_seed0.csv'), index=False)
    print(df.round(4).to_string())

def stage_timing():
    rows = []
    cpu = platform.processor() or ''
    try:
        cpu = [l.split(':')[1].strip() for l in open('/proc/cpuinfo') if 'model name' in l][0]
    except Exception: pass
    for name in DATASETS:
        d = get_data(name); N, T, Fd = d['Xtr'].shape[1:]
        sc = fit_scaler(d['Xtr'][:20000]); adj = build_graph(apply_scaler(d['Xtr'][:20000], sc))
        model = STGAFE(N, T, Fd, adj).eval(); x1 = torch.randn(1, N, T, Fd); xb = torch.randn(256, N, T, Fd)
        with torch.no_grad():
            for _ in range(50): model(x1)
            ts = []
            for _ in range(500):
                t0 = time.perf_counter(); model(x1); ts.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            for _ in range(20): model(xb)
            thr = (time.perf_counter() - t0) / (20 * 256) * 1000
        torch.save(model.state_dict(), '_tmp.pt'); sz = os.path.getsize('_tmp.pt') / 1024; os.remove('_tmp.pt')
        rows.append(dict(dataset=name, nodes=N, T=T, F=Fd, params=sum(q.numel() for q in model.parameters()),
                         model_size_KB=round(sz, 1), latency_ms_mean_b1=np.mean(ts), latency_ms_median_b1=np.median(ts),
                         latency_ms_p95_b1=np.percentile(ts, 95), ms_per_sample_b256=thr,
                         peak_RSS_MB=peak_mem_mb(),
                         cpu=cpu, threads=torch.get_num_threads(), torch=torch.__version__, python=platform.python_version(),
                         os=platform.platform(), note='forward pass only; excludes preprocessing and SHAP'))
        print(rows[-1])
    save_csv(rows, 'timing_complexity.csv')

if __name__ == '__main__':
    st = args.stage
    if st in ('main', 'all'): stage_main()
    if st in ('ablation', 'all'): stage_ablation()
    if st in ('baselines', 'all'): stage_baselines()
    if st in ('stats', 'all'): stage_stats()
    if st in ('timing', 'all'): stage_timing()
    print('DONE. Tables in', os.path.join(OUT, 'tables'))
