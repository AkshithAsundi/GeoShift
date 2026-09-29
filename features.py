import numpy as np, pandas as pd

EPS = 1e-3
QS = [5, 25, 50, 75, 95]

def _stats(a, name, out):
    # a: (n, H, W)
    f = a.reshape(len(a), -1)
    out[f'{name}_mean'] = f.mean(1)
    out[f'{name}_std'] = f.std(1)
    q = np.percentile(f, QS, axis=1)
    for qi, qq in zip(QS, q):
        out[f'{name}_p{qi}'] = qq

def tile_features(X):
    X = X.astype(np.float32)
    B, G, R, N = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    out = {}
    for nm, a in zip('BGRN', (B, G, R, N)):
        _stats(a, nm, out)
    vis = (B + G + R) / 3
    bright = (B + G + R + N) / 4
    ndvi = (N - R) / (N + R + EPS)
    ndwi = (G - N) / (G + N + EPS)
    gr = (G - R) / (G + R + EPS)
    rb = (R - B) / (R + B + EPS)
    nv = N / (vis + EPS)
    for nm, a in [('bright', bright), ('ndvi', ndvi), ('ndwi', ndwi), ('gr', gr), ('rb', rb), ('nv', nv)]:
        _stats(a, nm, out)
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        out[f'ndvi_gt{t}'] = (ndvi > t).mean((1, 2))
    for t in (-0.1, 0.0, 0.1, 0.2):
        out[f'ndwi_gt{t}'] = (ndwi > t).mean((1, 2))
    # chromaticity of tile means
    m = np.stack([B.mean((1, 2)), G.mean((1, 2)), R.mean((1, 2)), N.mean((1, 2))], 1)
    ch = m / (m.sum(1, keepdims=True) + EPS)
    for i, nm in enumerate('BGRN'):
        out[f'chrom_{nm}'] = ch[:, i]
    # texture on brightness, NIR and NDVI
    for nm, a in [('bright', bright), ('N', N), ('ndvi', ndvi)]:
        gx = np.abs(np.diff(a, axis=2)).mean((1, 2))
        gy = np.abs(np.diff(a, axis=1)).mean((1, 2))
        out[f'{nm}_grad'] = gx + gy
        out[f'{nm}_grad_rel'] = (gx + gy) / (np.abs(a).mean((1, 2)) + EPS)
        lap = a[:, 1:-1, 1:-1] * 4 - a[:, :-2, 1:-1] - a[:, 2:, 1:-1] - a[:, 1:-1, :-2] - a[:, 1:-1, 2:]
        out[f'{nm}_lap'] = np.abs(lap).mean((1, 2))
        blk = a.reshape(len(a), 8, 8, 8, 8)
        bstd = blk.std((2, 4)); bmean = blk.mean((2, 4))
        out[f'{nm}_blkstd_mean'] = bstd.mean((1, 2))
        out[f'{nm}_blkmean_std'] = bmean.std((1, 2))
    # centre vs whole (label = dominant class of tile; centre may matter)
    c = slice(16, 48)
    for nm, a in [('ndvi', ndvi), ('bright', bright), ('ndwi', ndwi)]:
        out[f'{nm}_ctr_mean'] = a[:, c, c].mean((1, 2))
    return pd.DataFrame(out).astype(np.float32)

def batched(X, bs=2000):
    return pd.concat([tile_features(X[i:i + bs]) for i in range(0, len(X), bs)], ignore_index=True)

