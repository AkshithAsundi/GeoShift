"""Shared utilities for the GeoShift pipeline (numpy only, no torch)."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

NCLS = 6
CLASSES = ['Forest', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Water/Wetland']


def load_meta(data_dir):
    tr = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    te = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    for d in (tr, te):
        p = d.Id.str.split('_', expand=True)
        d['reg'] = p[1].values
        d['r'] = p[2].astype(int).values
        d['c'] = p[3].astype(int).values
    tr['label'] = np.load(os.path.join(data_dir, 'train_labels.npy'))
    return tr, te


def get_folds(tr, out_dir, n=5):
    """Region-grouped folds, saved once so every model uses identical splits."""
    path = os.path.join(out_dir, f'folds{n}.npy')
    if os.path.exists(path):
        return np.load(path)
    f = np.zeros(len(tr), int)
    for k, (_, b) in enumerate(GroupKFold(n_splits=n).split(tr, tr.label, tr.reg)):
        f[b] = k
    os.makedirs(out_dir, exist_ok=True)
    np.save(path, f)
    return f


# ---------------------------------------------------------------- mosaics
def build_mosaics(meta, X, pad):
    """Stitch overlapping tiles (stride 32, size 64) back into per-region images.
    Returns {region: (img uint8 [4,H,W], valid uint8 [H,W])}, padded by `pad` on all sides."""
    mos = {}
    for g, d in meta.groupby('reg'):
        H, W = d.r.max() + 64, d.c.max() + 64
        M = np.zeros((4, H + 2 * pad, W + 2 * pad), np.uint8)
        V = np.zeros((H + 2 * pad, W + 2 * pad), np.uint8)
        for i, r, c in zip(d.index.values, d.r.values, d.c.values):
            M[:, pad + r:pad + r + 64, pad + c:pad + c + 64] = X[i]
            V[pad + r:pad + r + 64, pad + c:pad + c + 64] = 1
        mos[g] = (M, V)
    return mos


def crop(mos, g, r, c, C, pad):
    """C x C context window centred on the tile whose top-left is (r, c). Returns uint8 [5,C,C]
    (4 bands + validity mask)."""
    M, V = mos[g]
    r0 = pad + r + 32 - C // 2
    c0 = pad + c + 32 - C // 2
    return np.concatenate([M[:, r0:r0 + C, c0:c0 + C], V[None, r0:r0 + C, c0:c0 + C]], 0)


def region_stats(mos):
    """Per-region, per-band mean/std over valid pixels -> dict region -> array [4,2]."""
    st = {}
    for g, (M, V) in mos.items():
        v = V.astype(bool)
        px = M[:, v].astype(np.float64)
        st[g] = np.stack([px.mean(1), px.std(1) + 1.0], 1).astype(np.float32)
    return st


# ---------------------------------------------------------------- post-processing
def smooth(P, meta, rad=2, w=2.0):
    """Add w * (mean prob of other tiles within +-rad grid steps) to each tile's probs."""
    reg = meta.reg.values
    r = meta.r.values // 32
    c = meta.c.values // 32
    out = P.copy()
    for g in np.unique(reg):
        idx = np.where(reg == g)[0]
        H, W = r[idx].max() + 1, c[idx].max() + 1
        S = np.zeros((H, W, P.shape[1]))
        M = np.zeros((H, W))
        S[r[idx], c[idx]] = P[idx]
        M[r[idx], c[idx]] = 1
        IS = np.pad(S.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
        IM = np.pad(M.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        r0 = np.clip(r[idx] - rad, 0, H); r1 = np.clip(r[idx] + rad + 1, 0, H)
        c0 = np.clip(c[idx] - rad, 0, W); c1 = np.clip(c[idx] + rad + 1, 0, W)
        s = IS[r1, c1] - IS[r0, c1] - IS[r1, c0] + IS[r0, c0] - P[idx]
        m = IM[r1, c1] - IM[r0, c1] - IM[r1, c0] + IM[r0, c0] - 1
        nb = np.where(m[:, None] > 0, s / np.maximum(m, 1)[:, None], P[idx])
        out[idx] = P[idx] + w * nb
    return out / out.sum(1, keepdims=True)


def em_prior(P, meta, damp=0.5, floor=0.05, iters=20):
    """Per-region label-shift correction (Saerens EM), assuming a ~uniform training prior
    (models are trained class-balanced). damp<1 and floor keep rare classes alive for macro F1."""
    reg = meta.reg.values
    out = P.copy()
    for g in np.unique(reg):
        i = reg == g
        p = P[i]
        pi = np.full(P.shape[1], 1.0 / P.shape[1])
        q = p
        for _ in range(iters):
            q = p * (pi * P.shape[1]) ** damp
            q /= q.sum(1, keepdims=True)
            pi = np.maximum(q.mean(0), floor)
            pi /= pi.sum()
        out[i] = q
    return out


def postprocess(P, meta, rad=2, w=2.0, damp=0.5, floor=0.05):
    Q = P
    if w > 0:
        Q = smooth(Q, meta, rad, w)
    if damp > 0:
        Q = em_prior(Q, meta, damp, floor)
    return Q


def shrub_hedge(P, frac=0.015, cls=1):
    """Force the top `frac` of tiles by P[:,cls] to class `cls`. RISKY under leaderboard macro F1:
    if `cls` is absent from the hidden labels, any prediction of it adds a class scoring 0
    (a 1/6 cut). Only useful if you can actually detect the class, which we can't for Shrubland."""
    lab = P.argmax(1).copy()
    k = int(round(frac * len(P)))
    if k > 0:
        lab[np.argsort(-P[:, cls])[:k]] = cls
    return lab


def macro_f1(y, P):
    """Leaderboard-style macro F1: averages over classes present in truth OR predictions."""
    lab = P if P.ndim == 1 else P.argmax(1)
    return f1_score(y, lab, average='macro', zero_division=0)


def f1_no_shrub(y, P):
    """Mean F1 of the 5 non-Shrubland classes (Shrubland CV is uninformative: ~all from R06)."""
    return np.delete(per_class_f1(y, P), 1).mean()


def per_class_f1(y, P):
    lab = P if P.ndim == 1 else P.argmax(1)
    return f1_score(y, lab, average=None, labels=list(range(NCLS)), zero_division=0)
