"""Reproduce every validation number in the presentation (region-held-out CV).
  python run_validation.py --data ./data
Each step is cached in ./runs, so the script can be stopped and resumed.

Validation design: the 19 training regions are split into 5 groups (runs/folds5.npy, GroupKFold by
region). Every prediction below is made by a model that never saw that tile's region. Random splits are
invalid here: neighbouring tiles overlap by 50%, so they would leak.

Scores reported:
  all     - leaderboard-style macro F1 over all held-out tiles
  5-class - mean F1 of the five non-Shrubland classes, excluding region R06 (why: when R06 is held out
            its 4,090 Shrubland tiles become false Grassland; R06 is in training for the real test)
"""
import argparse, itertools, json, os, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import confusion_matrix
from geo_common import (load_meta, get_folds, postprocess, macro_f1, f1_no_shrub, per_class_f1, CLASSES)
import features, texture

p = argparse.ArgumentParser()
p.add_argument('--data', default='./data'); p.add_argument('--out', default='./runs')
p.add_argument('--threads', type=int, default=0)
args = p.parse_args(); os.makedirs(args.out, exist_ok=True)
t0 = time.time(); log = lambda *a: print(f'[{time.time() - t0:5.0f}s]', *a, flush=True)

tr, te = load_meta(args.data); y = tr.label.values; reg = tr.reg.values
folds = get_folds(tr, args.out, 5)
M = reg != 'R06'

def cached(path, fn):
    if os.path.exists(path): return pd.read_pickle(path)
    v = fn(); pd.to_pickle(v, path); return v

Xtr = np.load(os.path.join(args.data, 'train_images.npy'), mmap_mode='r')
Xte = np.load(os.path.join(args.data, 'test_images.npy'), mmap_mode='r')
F, Ft = cached(os.path.join(args.out, 'gbm_feats.pkl'), lambda: (features.batched(Xtr), features.batched(Xte)))
Tt, Tte = cached(os.path.join(args.out, 'tex_feats.pkl'), lambda: (texture.batched(Xtr), texture.batched(Xte)))
FS, FSt = F.values, Ft.values
FA, FAt = pd.concat([F, Tt], axis=1).values, pd.concat([Ft, Tte], axis=1).values
log('features: spectral', F.shape[1], '+ texture', Tt.shape[1])

BASE = dict(objective='multiclass', num_class=6, learning_rate=0.05, num_leaves=31, min_child_samples=40,
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
            num_threads=args.threads)
REG_ET = dict(BASE, num_leaves=15, min_child_samples=100, feature_fraction=0.5, lambda_l2=5, extra_trees=True)
DEFAULT_POST = (2, 2.0, 1.0, 0.02)  # smoothing radius, smoothing weight, EM damping, EM floor

def cw_of(labels):
    c = np.bincount(labels, minlength=6); return c.sum() / (6 * np.maximum(c, 1))

def cv(name, X, Xt, params, extra=None):
    """Region-grouped CV. `extra(k)` may return additional (X, y, w) rows for fold k (self-training)."""
    fo, ft = os.path.join(args.out, f'cv_{name}_oof.npy'), os.path.join(args.out, f'cv_{name}_test.npy')
    if os.path.exists(fo): return np.load(fo), np.load(ft)
    oof = np.zeros((len(y), 6)); T = 0
    for k in range(5):
        a = folds != k
        Xk, yk, wk = X[a], y[a], cw_of(y[a])[y[a]]
        if extra is not None:
            ex, ey, ew = extra(k)
            Xk, yk, wk = np.concatenate([Xk, ex]), np.concatenate([yk, ey]), np.concatenate([wk, ew])
        m = lgb.train(params, lgb.Dataset(Xk, yk, weight=wk), 300)
        oof[~a] = m.predict(X[~a]); T = T + m.predict(Xt) / 5
    np.save(fo, oof); np.save(ft, T); log('done', name)
    return oof, T

rows = []
def report(name, oof, post=DEFAULT_POST):
    Q = postprocess(oof, tr, *post) if post else oof
    rows.append(dict(step=name, all=round(macro_f1(y, Q), 4), five_class=round(f1_no_shrub(y[M], Q[M]), 4),
                     **{c: round(v, 3) for c, v in zip(CLASSES, per_class_f1(y[M], Q[M]))}))
    log(rows[-1])
    return Q

# 1-4: feature and model ablation
o1, _ = cv('spectral', FS, FSt, BASE);           report('1 spectral features', o1, None)
report('2 + neighbour smoothing & region prior', o1)
o3, _ = cv('texture', FA, FAt, BASE);            report('3 + texture features', o3)
o4, t4 = cv('texture_et', FA, FAt, REG_ET);      Q4 = report('4 + regularised extra-trees', o4)

# 5: self-training - confident post-processed predictions on the held-out fold (from models that never
#    saw it) and on test are added as pseudo-labels (threshold 0.9, weight 0.5)
Q4t = postprocess(t4, te, *DEFAULT_POST)
def pseudo(k):
    b = folds == k; pb = Q4[b].max(1) >= 0.9; pt = Q4t.max(1) >= 0.9
    yb, yt = Q4[b][pb].argmax(1), Q4t[pt].argmax(1); cw = cw_of(y[folds != k])
    return (np.concatenate([FA[b][pb], FAt[pt]]), np.concatenate([yb, yt]),
            np.concatenate([0.5 * cw[yb], 0.5 * cw[yt]]))
o5, _ = cv('selftrain', FA, FAt, REG_ET, pseudo); report('5 + self-training', o5)

# 6: blend of 4 and 5 + post-processing tuned on the 5-class score
def mix(a, b, w):
    L = w * np.log(np.clip(a, 1e-6, 1)) + (1 - w) * np.log(np.clip(b, 1e-6, 1))
    E = np.exp(L - L.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)
score = lambda P: f1_no_shrub(y[M], P[M])
w = max(np.linspace(0, 1, 21), key=lambda w: score(mix(o4, o5, w)))
B = mix(o4, o5, w)
grid = list(itertools.product([1, 2, 3], [0, 1, 2, 4], [0, 0.3, 0.5, 1.0], [0.02, 0.05]))
best = max(grid, key=lambda g: score(postprocess(B, tr, *g)))
Qf = report(f'6 blend (w={w:.2f}) + tuned post-processing', B, best)
# 7: Grassland-vs-Cropland decision threshold. Pasture looks like cropland (parcels), so a tile predicted
#    Cropland becomes Grassland when P(grass)/(P(grass)+P(crop)) >= X. X tuned on the same CV predictions.
def grass_rule(Q, X):
    p = Q.argmax(1).copy(); r = Q[:, 2] / (Q[:, 2] + Q[:, 3] + 1e-9); p[(p == 3) & (r >= X)] = 2; return p
Xs = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
X = max(Xs, key=lambda X: f1_no_shrub(y[M], grass_rule(Qf, X)[M]))
pf = grass_rule(Qf, X)
rows.append(dict(step=f'7 + Grassland/Cropland threshold X={X}', all=round(macro_f1(y, pf), 4),
                 five_class=round(f1_no_shrub(y[M], pf[M]), 4), **{c: round(v, 3) for c, v in zip(CLASSES, per_class_f1(y[M], pf[M]))}))
log(rows[-1])
json.dump(dict(blend_weight_model4=float(w), post=list(best), grass_ratio_threshold=float(X)),
          open(os.path.join(args.out, 'tuned.json'), 'w'))
log('tuned settings', dict(blend_weight_model4=w, post=best, grass_ratio_threshold=X))

# Shrubland experiment: can a dedicated detector find the 95 Shrubland tiles outside R06?
fs = os.path.join(args.out, 'cv_shrub_detector.npy')
if not os.path.exists(fs):
    t = (y == 1).astype(int); s = np.zeros(len(y))
    P = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_child_samples=20, feature_fraction=0.5,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, verbose=-1, num_threads=args.threads)
    for k in range(5):
        a = (folds != k) & M  # R06 excluded: target the scattered kind of Shrubland
        m = lgb.train(P, lgb.Dataset(FA[a], t[a], weight=np.where(t[a] == 1, 20, 1)), 200)
        s[folds == k] = m.predict(FA[folds == k])
    np.save(fs, s)
s = np.load(fs)[M]; t = (y[M] == 1)
hits = {k: int(t[np.argsort(-s)[:k]].sum()) for k in (50, 100, 200)}
log(f'Shrubland detector (outside R06, {t.sum()} true tiles): true Shrubland among top-k =', hits)

pd.DataFrame(rows).to_csv(os.path.join(args.out, 'validation_table.csv'), index=False)
print('\n' + pd.DataFrame(rows).to_string(index=False))
print('\nConfusion matrix, final step, excluding R06 (rows = true, cols = predicted):')
print(pd.DataFrame(confusion_matrix(y[M], pf[M], labels=range(6)), index=CLASSES, columns=CLASSES))
