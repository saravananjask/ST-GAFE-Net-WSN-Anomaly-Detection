# ST-GAFE-Net: window-based graph attention network for WSN fault and anomaly detection

Code for the paper "An Explainable Spatio-Temporal Graph Attention Network with
Frequency-Enhanced Learning for Fault and Anomaly Detection in Wireless Sensor Networks".

## Files
- `STGAFE_V2.py` : main pipeline (training, blocked CV, baselines, ablation, McNemar tests, timing)
- `STGAFE_Explain.py` : node-level SHAP, local explanations and GAT attention (needs `STGAFE_V2.py`)
- `results_tables.xlsx` : all result tables used in the paper (one sheet per table; `_raw` and `_drift` sheets for the raw and drift-robust runs, plus SHAP and attention tables)

Scripts of the first submission were removed from this repository. They are not used for any reported result.

## Datasets (not included, download them yourself)
- WSN Fault: https://github.com/tmoulahi/Dataset-for-WSN-fault-detection
- Intel Berkeley Lab: http://db.csail.mit.edu/labdata/labdata.html
- SWaT: https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system
- WSN-DS: https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds

Put them in one folder and pass it with `--root`. The expected layout is read in `get_data` inside `STGAFE_V2.py`.

## Run
```
python STGAFE_V2.py --root <data folder> --out Results_V2 --stage main --seeds 5 --max_train 30000
python STGAFE_V2.py --root <data folder> --out Results_V2 --stage ablation --seeds 3 --max_train 15000
python STGAFE_V2.py --root <data folder> --out Results_V2 --stage baselines --seeds 3 --max_train 15000
python STGAFE_V2.py --root <data folder> --out Results_V2 --stage stats
python STGAFE_V2.py --root <data folder> --out Results_V2 --stage timing
python STGAFE_V2.py --root <data folder> --out Results_V2_diff --rep diff --datasets WSN_Fault,Intel_Lab --stage main
python STGAFE_Explain.py --root <data folder> --out Results_V2 --datasets Intel_Lab,SWaT_seen,WSN_DS,SWaT
python STGAFE_Explain.py --root <data folder> --out Results_V2_diff --rep diff --datasets WSN_Fault,Intel_Lab
```
Add `--quick` for a fast smoke test.

## Protocol in short
- Windows X[N nodes, T steps, F features]; BiGRU over time, FFT along time, 2 GAT layers on a fixed top-3 correlation graph from training data.
- Chronological, episode-based or node-disjoint splits with a purge gap; scaler, graph and decision threshold fitted on training data only; no SMOTE (class-weighted loss).
- Seeds 0-4 for the main results, 0-2 for baselines and ablation.
- Intel Lab has no labels: faults are injected by fixed rules (seeds 123 for training, 456 for test).
- Timing: forward pass only, CPU, excludes preprocessing and SHAP.

## Citation
Please cite the paper once it is published.
