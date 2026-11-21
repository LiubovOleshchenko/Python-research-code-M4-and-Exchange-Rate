"""Dataset D1: M4 Quarterly, Finance domain (5,305 series), horizon 8. Builds PSR/TDA/raw inputs once.
Usage: python d1_prep.py [hist]   (hist = optional max history length, e.g. 20 for the limited-history condition)"""
import sys, numpy as np, pandas as pd, time
from common import choose_tau, choose_m, tda_features
from multiprocessing import Pool
L, H, P, MMAX, K = 16, 8, 4, 4, 12
HIST = int(sys.argv[1]) if (__name__ == "__main__" and len(sys.argv) > 1) else 0

def load():
    info = pd.read_csv('../data/M4-info.csv'); ids = set(info[(info.SP == 'Quarterly') & (info.category == 'Finance')].M4id)
    tr = pd.read_csv('../data/Quarterly-train.csv', index_col=0); te = pd.read_csv('../data/Quarterly-test.csv', index_col=0)
    tr, te = tr.loc[tr.index.isin(ids)], te.loc[te.index.isin(ids)]
    X = [r.dropna().values.astype(float) for _, r in tr.iterrows()]; Y = te.loc[tr.index].values.astype(float)
    return list(tr.index), X, Y

def psr_vec(w, m, tau):
    v = np.zeros((P, MMAX), np.float32)
    for p in range(P):
        for j in range(m): v[p, j] = w[len(w) - 1 - p - j * tau]
    return np.r_[v.ravel(), m / MMAX, tau / 4].astype(np.float32)

def series_job(args):
    x, = args
    if HIST: x = x[-(HIST + H):] if len(x) >= HIST + H else x  # HIST obs before calibration window
    n = len(x); fit = x[:n - H]
    s = np.mean(np.abs(x[-L:])) + 1e-8
    z = np.diff(fit / s) if len(fit) > 3 else np.zeros(4)
    tau = choose_tau(z, 4) if len(z) >= 12 else 1
    tau = min(tau, (L - P) // (MMAX - 1))
    m = choose_m(z, tau, MMAX) if len(z) >= 12 else 2
    m = max(2, min(m, MMAX))
    def window(t):  # origin t: inputs x[t-L:t]
        w = x[max(0, t - L):t]
        if len(w) < L: w = np.r_[np.full(L - len(w), w[0]), w]
        wn = (w - w[-1]) / s
        cloud = np.stack([wn[(m - 1 - j) * tau: L - j * tau] for j in range(m)], 1)
        return wn.astype(np.float32), psr_vec(wn, m, tau), tda_features(cloud)
    out = {'raw': [], 'psr': [], 'tda': [], 'y': [], 'kind': []}
    origins = list(range(max(L // 2, 4), n - 2 * H + 1))[-K:]
    for t in origins + [n - H, n]:
        r, pv, td = window(t)
        out['raw'].append(r); out['psr'].append(pv); out['tda'].append(td)
        out['y'].append(((x[t:t + H] - x[t - 1]) / s) if t + H <= n else np.zeros(H))
        out['kind'].append(0 if t not in (n - H, n) else (1 if t == n - H else 2))
    return out, s, m, tau, x[-1]

if __name__ == '__main__':
    t0 = time.time(); ids, X, Y = load()
    with Pool(2) as pool: res = pool.map(series_job, [(x,) for x in X], chunksize=50)
    D = {k: [] for k in ['raw', 'psr', 'tda', 'y', 'kind', 'sid']}
    S, M, T, LAST = [], [], [], []
    for i, (o, s, m, tau, last) in enumerate(res):
        for k in o: D[k] += o[k]
        D['sid'] += [i] * len(o['kind']); S.append(s); M.append(m); T.append(tau); LAST.append(last)
    D = {k: np.array(v) for k, v in D.items()}
    tag = f'_h{HIST}' if HIST else ''
    np.savez_compressed(f'd1{tag}.npz', **D, scale=np.array(S), m=np.array(M), tau=np.array(T), last=np.array(LAST), Ytest=Y,
                        lens=np.array([len(x) for x in X]))
    print('series', len(X), 'samples', len(D['kind']), 'train', (D['kind'] == 0).sum(), 'time', time.time() - t0)
    print('m dist', np.bincount(M), 'tau dist', np.bincount(T))
