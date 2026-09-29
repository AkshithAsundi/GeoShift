"""Reproduce the final submission on CPU (no GPU needed, ~15 min on a laptop).
  python make_submission.py --data ./data
Steps: features -> LightGBM on all 19 regions (3 seeds) -> neighbour smoothing + region prior
-> confident test tiles become pseudo-labels -> retrain (3 seeds) -> blend -> smoothing -> submission.csv
Model settings were chosen by region-held-out CV; blend/post-processing settings are read from
runs/tuned.json written by run_validation.py (defaults below match it)."""
import argparse, os, time, numpy as np, pandas as pd, lightgbm as lgb
from geo_common import load_meta, postprocess, CLASSES
import features, texture

p = argparse.ArgumentParser()
p.add_argument('--data', default='./data'); p.add_argument('--out', default='./runs')
p.add_argument('--threads', type=int, default=0); p.add_argument('--sub', default='submission.csv')
p.add_argument('--grass_x', type=float, default=None, help='override the CV-tuned Grassland/Cropland threshold')
args = p.parse_args(); os.makedirs(args.out, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f'[{time.time() - t0:5.0f}s]', *a, flush=True)
tr, te = load_meta(args.data); y = tr.label.values

def cached(path, fn):
    if os.path.exists(path): return pd.read_pickle(path)
    v = fn(); pd.to_pickle(v, path); return v

Xtr = np.load(os.path.join(args.data, 'train_images.npy'), mmap_mode='r')
Xte = np.load(os.path.join(args.data, 'test_images.npy'), mmap_mode='r')
F, Ft = cached(os.path.join(args.out, 'gbm_feats.pkl'), lambda: (features.batched(Xtr), features.batched(Xte)))
log('spectral features', F.shape)
Tt, Tte = cached(os.path.join(args.out, 'tex_feats.pkl'), lambda: (texture.batched(Xtr), texture.batched(Xte)))
log('texture features', Tt.shape)
FA = pd.concat([F, Tt], axis=1).values; FAt = pd.concat([Ft, Tte], axis=1).values

PARAMS = dict(objective='multiclass', num_class=6, learning_rate=0.05, num_leaves=15, min_child_samples=100,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, extra_trees=True,
              verbose=-1, num_threads=args.threads)
cnt = np.bincount(y, minlength=6); CW = cnt.sum() / (6 * np.maximum(cnt, 1))

def fit_predict(X, yy, w, tag, seeds=(0, 1, 2)):
    out = 0
    for s in seeds:
        f = os.path.join(args.out, f'final_{tag}_s{s}.npy')
        if not os.path.exists(f):
            m = lgb.train({**PARAMS, 'seed': s}, lgb.Dataset(X, yy, weight=w), 300)
            np.save(f, m.predict(FAt)); log(tag, 'seed', s, 'done')
        out = out + np.load(f) / len(seeds)
    return out

# round 1: all labelled regions
T1 = fit_predict(FA, y, CW[y], 'r1')
# pseudo-labels: confident, post-processed round-1 test predictions (threshold/weight from CV)
P1 = postprocess(T1, te, 2, 2.0, 1.0, 0.02)
keep = P1.max(1) >= 0.9
yp = P1[keep].argmax(1)
log('pseudo-labelled test tiles:', int(keep.sum()), 'of', len(te))
T2 = fit_predict(np.concatenate([FA, FAt[keep]]), np.concatenate([y, yp]),
                 np.concatenate([CW[y], 0.5 * CW[yp]]), 'r2')
# blend (weighted geometric mean) + post-processing, both tuned by run_validation.py
import json
tj = os.path.join(args.out, 'tuned.json')
cfg = json.load(open(tj)) if os.path.exists(tj) else dict(blend_weight_model4=0.0, post=[1, 2, 0, 0.02], grass_ratio_threshold=0.3)
log('blend/post settings', cfg, '(from run_validation.py)' if os.path.exists(tj) else '(defaults)')
wt = cfg['blend_weight_model4']
L = wt * np.log(np.clip(T1, 1e-6, 1)) + (1 - wt) * np.log(np.clip(T2, 1e-6, 1))
B = np.exp(L - L.max(1, keepdims=True)); B /= B.sum(1, keepdims=True)
Q = postprocess(B, te, *cfg['post'])  # full 6-class output; no class is blocked
np.save(os.path.join(args.out, 'final_test_probs.npy'), Q)
lab = Q.argmax(1)
X = cfg.get('grass_ratio_threshold', 0.5) if args.grass_x is None else args.grass_x  # Cropland -> Grassland when P(g)/(P(g)+P(c)) >= X
ratio = Q[:, 2] / (Q[:, 2] + Q[:, 3] + 1e-9)
lab[(lab == 3) & (ratio >= X)] = 2
log(f'Grassland/Cropland threshold {X}: {int(((Q.argmax(1) == 3) & (lab == 2)).sum())} tiles Cropland -> Grassland')
pd.DataFrame({'Id': te.Id, 'label': lab}).to_csv(args.sub, index=False)
log('wrote', args.sub)
print(pd.crosstab(te.reg.values, np.array(CLASSES)[lab]))
