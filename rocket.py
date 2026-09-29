"""Random-convolution texture features (ROCKET / MiniRocket style, adapted to 2-D multispectral tiles).
192 random 3x3 kernels over 6 channels (B, G, R, NIR standardised + NDVI + NDWI) at dilations 1, 2, 4, 8
(receptive fields 30 m to 170 m). Per kernel: PPV (share of positive responses) and mean positive response.
Kernel weights and biases are fixed by a seed and fitted biases come from training tiles only."""
import numpy as np, pandas as pd

MEAN = np.array([43.0, 48.6, 51.6, 84.9], np.float32)[:, None, None]
STD = np.array([13.0, 13.5, 16.9, 22.3], np.float32)[:, None, None]
DILS = (1, 2, 4, 8)
K_PER = 48


def prep(X):
    x = X.astype(np.float32)
    B, G, R, N = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    ndvi = (N - R) / (N + R + 1e-3); ndwi = (G - N) / (G + N + 1e-3)
    return np.concatenate([(x - MEAN) / STD, 3 * ndvi[:, None], 3 * ndwi[:, None]], 1)  # (n,6,64,64)


def kernels(seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(len(DILS), K_PER, 6 * 9)).astype(np.float32)
    W -= W.mean(2, keepdims=True)
    q = rng.uniform(0.25, 0.75, size=(len(DILS), K_PER))
    return W, q


def responses(x, W_d, d):
    n = len(x)
    xp = np.pad(x, ((0, 0), (0, 0), (d, d), (d, d)), mode='reflect')
    cols = np.stack([xp[:, :, i * d:i * d + 64, j * d:j * d + 64] for i in range(3) for j in range(3)], 2)  # n,6,9,64,64
    cols = cols.reshape(n, 54, 4096)
    return np.einsum('kc,ncp->nkp', W_d, cols, optimize=True)  # n,K,4096


def fit_biases(Xsample, W, q):
    x = prep(Xsample)
    b = np.zeros(q.shape, np.float32)
    for di, d in enumerate(DILS):
        r = responses(x, W[di], d)
        for k in range(K_PER):
            b[di, k] = -np.quantile(r[:, k].ravel(), q[di, k])
    return b


def features(X, W, b, bs=128):
    out = []
    for s in range(0, len(X), bs):
        x = prep(np.asarray(X[s:s + bs])); f = []
        for di, d in enumerate(DILS):
            r = responses(x, W[di], d) + b[di][None, :, None]
            pos = r > 0
            f.append(pos.mean(2)); f.append(np.where(pos, r, 0).mean(2))
        out.append(np.concatenate(f, 1))
    F = np.concatenate(out)
    names = [f'rk_d{d}_{t}{k}' for d in DILS for t in ('ppv', 'mpv') for k in range(K_PER)]
    return pd.DataFrame(F, columns=names).astype(np.float32)
