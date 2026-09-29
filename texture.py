"""Texture features aimed at Cropland vs Grassland vs Shrubland (field edges, parcels, patchiness)."""
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

EPS = 1e-6
_yy, _xx = np.mgrid[-32:32, -32:32]
_RAD = np.fft.fftshift(np.hypot(_yy, _xx))
_ANG = np.fft.fftshift(np.mod(np.arctan2(_yy, _xx), np.pi))
_RBINS = [(1, 3), (3, 6), (6, 12), (12, 24), (24, 46)]


def _struct(a):
    gy, gx = np.gradient(a, axis=(1, 2))
    J11 = uniform_filter(gx * gx, (1, 5, 5)); J22 = uniform_filter(gy * gy, (1, 5, 5))
    J12 = uniform_filter(gx * gy, (1, 5, 5))
    tr = J11 + J22
    coh = np.sqrt((J11 - J22) ** 2 + 4 * J12 ** 2) / (tr + EPS)
    w = tr / (tr.sum((1, 2), keepdims=True) + EPS)
    theta = 0.5 * np.arctan2(2 * J12, J11 - J22)
    # dominant-orientation strength: |weighted mean of exp(2i*theta)|
    z = (w * np.exp(2j * theta)).sum((1, 2))
    return {'coh_mean': coh.mean((1, 2)), 'coh_wmean': (coh * w).sum((1, 2)),
            'coh_p75': np.percentile(coh.reshape(len(a), -1), 75, axis=1), 'orient_strength': np.abs(z)}


def _fft(a):
    a = a - a.mean((1, 2), keepdims=True)
    P = np.abs(np.fft.fft2(a)) ** 2
    tot = P.sum((1, 2)) + EPS
    out = {}
    for lo, hi in _RBINS:
        m = (_RAD >= lo) & (_RAD < hi)
        out[f'fft_r{lo}'] = P[:, m].sum(1) / tot
    m = _RAD >= 3
    hist = np.stack([P[:, m & (_ANG >= k * np.pi / 8) & (_ANG < (k + 1) * np.pi / 8)].sum(1) for k in range(8)], 1)
    hist = hist / (hist.sum(1, keepdims=True) + EPS)
    out['fft_aniso'] = hist.max(1) * 8
    out['fft_ang_entropy'] = -(hist * np.log(hist + EPS)).sum(1)
    out['fft_logtot'] = np.log(tot)
    return out


def _glcm_lbp(a, lo, hi, name):
    q = np.clip((a - lo) / (hi - lo) * 15, 0, 15).astype(np.uint8)
    out = {f'{name}_{p}_d{d}': [] for p in ('contrast', 'homogeneity', 'energy', 'correlation') for d in (1, 4)}
    lbp_h = []
    for t in q:
        g = graycomatrix(t, [1, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=16, symmetric=True, normed=True)
        for p in ('contrast', 'homogeneity', 'energy', 'correlation'):
            v = graycoprops(g, p).mean(1)
            out[f'{name}_{p}_d1'].append(v[0]); out[f'{name}_{p}_d4'].append(v[1])
        l = local_binary_pattern(t, 8, 1, 'uniform')
        lbp_h.append(np.bincount(l.astype(int).ravel(), minlength=10)[:10] / l.size)
    out = {k: np.nan_to_num(np.array(v)) for k, v in out.items()}
    lbp_h = np.array(lbp_h)
    for i in range(10):
        out[f'{name}_lbp{i}'] = lbp_h[:, i]
    return out


def texture_features(X):
    X = X.astype(np.float32)
    R, N = X[:, 2], X[:, 3]
    ndvi = (N - R) / (N + R + 1e-3)
    out = {}
    for nm, a in (('nir', N / (N.mean((1, 2), keepdims=True) + EPS)), ('ndvi', ndvi)):
        for k, v in _struct(a).items(): out[f'{nm}_{k}'] = v
        for k, v in _fft(a).items(): out[f'{nm}_{k}'] = v
    # scale-free NIR for GLCM (per-tile percentile stretch) + absolute NDVI
    lo = np.percentile(N.reshape(len(N), -1), 2, 1)[:, None, None]; hi = np.percentile(N.reshape(len(N), -1), 98, 1)[:, None, None]
    out.update(_glcm_lbp((N - lo) / (hi - lo + EPS), 0, 1, 'gnir'))
    out.update(_glcm_lbp(ndvi, -0.1, 0.8, 'gndvi'))
    # multi-scale heterogeneity of NDVI (16px blocks = 160 m)
    b = ndvi.reshape(len(ndvi), 4, 16, 4, 16).mean((2, 4))
    out['ndvi_blk16_std'] = b.std((1, 2)); out['ndvi_blk16_range'] = b.max((1, 2)) - b.min((1, 2))
    return pd.DataFrame(out).astype(np.float32)


def batched(X, bs=1000):
    return pd.concat([texture_features(np.asarray(X[i:i + bs])) for i in range(0, len(X), bs)], ignore_index=True)
