"""Evidence for apply_shrub_rule.py, on held-out training regions (same 5 region folds).
  python validate_shrub_rule.py --data ./data      (run run_validation.py and apply_shrub_rule.py first)
Prints, per held-out region with both Shrubland and Grassland, how well P(Forest)/(P(Forest)+P(Cropland))
separates them (ROC AUC), for the texture model and for the model with random-convolution features."""
import argparse, os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from geo_common import load_meta, postprocess
p = argparse.ArgumentParser(); p.add_argument('--data', default='./data'); p.add_argument('--out', default='./runs')
args = p.parse_args()
tr, te = load_meta(args.data); y = tr.label.values; treg = tr.reg.values
folds = np.load(os.path.join(args.out, 'folds5.npy'))
F, _ = pd.read_pickle(os.path.join(args.out, 'gbm_feats.pkl')); T, _ = pd.read_pickle(os.path.join(args.out, 'tex_feats.pkl'))
R, _ = pd.read_pickle(os.path.join(args.out, 'rocket_feats.pkl'))
P = dict(objective='multiclass', num_class=6, learning_rate=0.05, num_leaves=15, min_child_samples=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, extra_trees=True, verbose=-1, num_threads=1, seed=0)
for name, X, post in (('texture features', pd.concat([F, T], axis=1).values, (1, 1, 1.0, 0.02)),
                      ('+ random-conv features', pd.concat([F, T, R], axis=1).values, (1, 2, 1.0, 0.02))):
    o = np.zeros((len(y), 6))
    for k in range(5):
        a = folds != k; c = np.bincount(y[a], minlength=6); cw = c.sum() / (6 * np.maximum(c, 1))
        o[~a] = lgb.train(P, lgb.Dataset(X[a], y[a], weight=cw[y[a]]), 300).predict(X[~a])
    Q = postprocess(o, tr, *post); w = Q[:, 0] / (Q[:, 0] + Q[:, 3] + 1e-9)
    per = {g: round(roc_auc_score(y[m] == 1, w[m]), 2) for g in np.unique(treg) if g != 'R06'
           for m in [np.isin(y, [1, 2]) & (treg == g)] if (y[m] == 1).sum() >= 3 and (y[m] == 2).sum() >= 3}
    print(f'{name:24s} Shrubland-vs-Grassland AUC per held-out region: {per}')
