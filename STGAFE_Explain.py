# =============================================================================
# ST-GAFE-Net  -  explainability for the revision (Reviewer 1, comment 8)
#
#  * Explainer     : SHAP KernelExplainer (interventional), players = SENSOR NODES (not single
#                    feature columns) -> avoids splitting credit between correlated raw/FFT columns.
#  * Output        : probability of the anomaly class (softmax output of the trained model)
#  * Masking       : a switched-off node is replaced by the same node from K background windows
#                    (random TRAINING windows); the K outputs are averaged
#  * Instances     : n_explain test windows (half anomalies, half normals) + one true positive,
#                    one false positive and one false negative for local explanations
#  * Stability     : the whole analysis is repeated for several seeds (model retrained each time);
#                    Spearman rank correlation and top-5 overlap of node importances are reported
#  * Spatial part  : GAT attention weights (last layer, mean over heads) for anomaly vs normal
#                    windows, compared with the correlation graph
#
# Usage:
#   pip install shap
#   python STGAFE_Explain.py --root "F:\June 26\New WSN 22.6.26\Dataset" --out Results_V2 --datasets Intel_Lab,SWaT_seen,WSN_DS
#   python STGAFE_Explain.py --out Results_V2_diff --rep diff --datasets WSN_Fault,Intel_Lab
# Needs STGAFE_V2.py in the same folder.
# =============================================================================
import argparse, sys, os
ap = argparse.ArgumentParser()
ap.add_argument('--root', default=r'F:\June 26\New WSN 22.6.26\Dataset')
ap.add_argument('--out', default='Results_V2')
ap.add_argument('--datasets', default='Intel_Lab,SWaT_seen,WSN_DS,SWaT')
ap.add_argument('--rep', default='raw')
ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--n_explain', type=int, default=100)
ap.add_argument('--nsamples', type=int, default=300)
ap.add_argument('--n_bg', type=int, default=8)
ap.add_argument('--max_train', type=int, default=15000)
ap.add_argument('--epochs', type=int, default=12)
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
sys.argv = ['STGAFE_V2.py', '--root', a.root, '--out', a.out, '--rep', a.rep, '--epochs', str(a.epochs),
            '--max_train', str(a.max_train)] + (['--quick'] if a.quick else [])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
import STGAFE_V2 as M

# the explanation run retrains the model; it must NOT overwrite the saved predictions of the main run
np.savez_compressed = lambda *args, **kw: None
OUTX = os.path.join(a.out, 'explain'); os.makedirs(OUTX, exist_ok=True)

def node_names(name, d):
    if name.startswith('SWaT'):
        try:
            cols = pd.read_csv(os.path.join(a.root, 'SWAT', 'normal.csv'), nrows=0).columns[1:52]
            return [c.strip() for c in cols]
        except Exception:
            pass
    return list(d['nodes'])

def node_shap(model, x, bgs, N, nsamples):
    """Shapley values of the N sensor nodes for ONE window x (1,N,T,F)."""
    def f(Z):
        Z = np.asarray(Z, np.float32)[:, :, None, None]
        out = 0
        for b in bgs:
            out = out + M.predict(model, (x * Z + b * (1 - Z)).astype(np.float32))
        return out / len(bgs)
    ex = shap.KernelExplainer(f, np.zeros((1, N)))
    sv = ex.shap_values(np.ones((1, N)), nsamples=nsamples, silent=True)
    return np.asarray(sv).reshape(-1)[:N]

def attention(model, X, N):
    gats = [g for g in (model.gats or [])]
    if not gats: return None
    for g in gats: g.keep = True
    model.eval(); mats = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            model(torch.from_numpy(X[i:i + 512])); mats.append(gats[-1].last.mean(-1).numpy())  # mean over heads
    for g in gats: g.keep = False
    return np.concatenate(mats).mean(0)                                                          # (N,N)

for name in a.datasets.split(','):
    d = M.get_data(name); names = node_names(name, d); N = len(names)
    rows, locals_, stab = [], [], []
    for seed in range(a.seeds):
        print(f"[explain] {name} seed {seed}: training ...", flush=True)
        m, model, sc, adj, hist = M.run_final(d, 'full', seed, name)
        thr = m['thr']; rng = np.random.default_rng(seed)
        Xte = M.apply_scaler(d['Xte'], sc); yte = d['yte']; p = M.predict(model, Xte); pred = (p >= thr).astype(int)
        # background = random training windows
        Xtr = M.apply_scaler(d['Xtr'], sc); bgs = [Xtr[i][None] for i in rng.choice(len(Xtr), a.n_bg, replace=False)]
        # instances: half anomalies / half normals, plus TP, FP, FN for local explanations
        pos = np.where(yte == 1)[0]; neg = np.where(yte == 0)[0]; h = a.n_explain // 2
        idx = np.r_[rng.choice(pos, min(h, len(pos)), replace=False), rng.choice(neg, min(h, len(neg)), replace=False)]
        tp = np.where((yte == 1) & (pred == 1))[0]; fp = np.where((yte == 0) & (pred == 1))[0]; fn = np.where((yte == 1) & (pred == 0))[0]
        loc = {}
        if len(tp): loc['TP'] = tp[np.argmax(p[tp])]
        if len(fp): loc['FP'] = fp[np.argmax(p[fp])]
        if len(fn): loc['FN'] = fn[np.argmin(p[fn])]
        S = np.array([node_shap(model, Xte[i][None], bgs, N, a.nsamples) for i in idx])
        imp = np.abs(S).mean(0)
        for j in range(N): rows.append(dict(dataset=name, seed=seed, node=names[j], mean_abs_shap=imp[j],
                                             mean_abs_shap_anomaly=np.abs(S[yte[idx] == 1, j]).mean(),
                                             mean_abs_shap_normal=np.abs(S[yte[idx] == 0, j]).mean()))
        stab.append(imp)
        for k, i in loc.items():
            sv = node_shap(model, Xte[i][None], bgs, N, a.nsamples)
            locals_.append((seed, k, float(p[i]), sv))
        if seed == 0:
            A = attention(model, Xte[np.r_[pos[:1000], neg[:1000]]], N)
            Aa = attention(model, Xte[pos[:1000]], N); An = attention(model, Xte[neg[:1000]], N)
            if A is not None:
                ed = [dict(dataset=name, src=names[i], dst=names[j], attn_anomaly=Aa[i, j], attn_normal=An[i, j],
                           diff=Aa[i, j] - An[i, j], in_graph=bool(adj[i, j])) for i in range(N) for j in range(N) if adj[i, j] and i != j]
                pd.DataFrame(ed).sort_values('diff', key=abs, ascending=False).to_csv(
                    os.path.join(OUTX, f'attention_edges_{name}.csv'), index=False)
                fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(Aa - An, cmap='coolwarm', vmin=-abs(Aa - An).max(), vmax=abs(Aa - An).max())
                ax.set_title(f'{name}: GAT attention, anomaly minus normal'); ax.set_xlabel('neighbour node'); ax.set_ylabel('target node')
                fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(OUTX, f'attention_{name}.png'), dpi=200); plt.close(fig)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUTX, f'shap_node_importance_{name}.csv'), index=False)
    # stability across seeds
    stab = np.array(stab); st = []
    for i in range(len(stab)):
        for j in range(i + 1, len(stab)):
            rho = spearmanr(stab[i], stab[j]).correlation
            t1, t2 = set(np.argsort(-stab[i])[:5]), set(np.argsort(-stab[j])[:5])
            st.append(dict(dataset=name, seed_a=i, seed_b=j, spearman=rho, top5_overlap=len(t1 & t2) / 5))
    pd.DataFrame(st).to_csv(os.path.join(OUTX, f'shap_stability_{name}.csv'), index=False)
    # global figure
    g = df.groupby('node').mean_abs_shap.agg(['mean', 'std']).sort_values('mean', ascending=False).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(g.index, g['mean'], xerr=g['std'].fillna(0), color='#3b78b5', ecolor='#555')
    ax.set_xlabel('mean |SHAP| of the sensor node (probability units)'); ax.set_title(f'{name}: sensor importance (mean over {a.seeds} seeds)')
    fig.tight_layout(); fig.savefig(os.path.join(OUTX, f'shap_global_{name}.png'), dpi=200); plt.close(fig)
    # local figure (seed 0)
    L = [x for x in locals_ if x[0] == 0]
    if L:
        fig, axs = plt.subplots(1, len(L), figsize=(5 * len(L), 4.5), squeeze=False)
        for ax, (s, k, pr, sv) in zip(axs[0], L):
            top = np.argsort(-np.abs(sv))[:8][::-1]
            ax.barh([names[t] for t in top], sv[top], color=['#c0392b' if v > 0 else '#2e86c1' for v in sv[top]])
            ax.set_title(f'{name} {k}: p(anomaly)={pr:.2f}'); ax.set_xlabel('node SHAP')
        fig.tight_layout(); fig.savefig(os.path.join(OUTX, f'shap_local_{name}.png'), dpi=200); plt.close(fig)
    print(f"[explain] {name} done. mean Spearman across seeds:", np.mean([s['spearman'] for s in st]) if st else 'n/a')
print('DONE. Files in', OUTX)
