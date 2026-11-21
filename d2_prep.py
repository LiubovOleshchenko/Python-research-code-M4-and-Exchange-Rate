"""Dataset D2: Exchange-Rate (8 currencies, 7,588 daily obs). TE matrix, causal graphs, PSR/TDA inputs.
Usage: python d2_prep.py H [cond]  cond in {clean, missing, noise}"""
import sys, json, numpy as np, time
from multiprocessing import Pool
from common import choose_tau, choose_m, tda_features, te_matrix, pc_lagged
H = int(sys.argv[1]); COND = sys.argv[2] if len(sys.argv) > 2 else 'clean'
L, P, MMAX, PL = 32, 4, 6, 4
raw = np.loadtxt('../data/exchange_rate.txt.gz', delimiter=',')
NT, NV, NTE = 5311, 759, 1518            # 70/10/20 chronological split
rng = np.random.default_rng(123)
if COND == 'missing':                      # 10% values removed at random, then linearly interpolated
    X = raw.copy()
    for j in range(X.shape[1]):
        miss = rng.random(len(X)) < 0.10; miss[[0, -1]] = False
        X[miss, j] = np.interp(np.where(miss)[0], np.where(~miss)[0], X[~miss, j])
    raw = X
elif COND == 'noise':                      # additive Gaussian observation noise, sd = 0.5 x sd of daily changes
    raw = raw + rng.normal(0, 1, raw.shape) * 0.5 * np.diff(raw[:NT], axis=0).std(0)
mu, sd = raw[:NT].mean(0), raw[:NT].std(0); X = (raw - mu) / sd
R = np.diff(X[:NT], axis=0)

def var_params(j):
    tau = min(choose_tau(R[:, j], 4), (L - P) // (MMAX - 1)); m = max(2, min(choose_m(R[-2000:, j], tau, MMAX), MMAX)); return m, tau

def build(args):
    j, origins, m, tau = args; out = []
    for t in origins:
        w = X[t - L:t, j] - X[t - 1, j]
        v = np.zeros((P, MMAX), np.float32)
        for p in range(P):
            for k in range(m): v[p, k] = w[L - 1 - p - k * tau]
        cloud = np.stack([w[(m - 1 - k) * tau: L - k * tau] for k in range(m)], 1)
        out.append((w.astype(np.float32), np.r_[v.ravel(), m / MMAX, tau / 4].astype(np.float32), tda_features(cloud)))
    return out

if __name__ == '__main__':
    t0 = time.time()
    if COND == 'clean' or True:
        T, Pv, Sig = te_matrix(R, n_surr=100)
        A_te = pc_lagged(R, PL, 0.05, candidates=Sig)     # full method: PC restricted to TE-significant sources
        A_pc = pc_lagged(R, PL, 0.05, candidates=None)    # PC alone (used in the "TE removed" ablation)
    t_cg = time.time() - t0
    params = [var_params(j) for j in range(8)]
    tr_o = list(range(L, NT - H + 1)); va_o = list(range(NT, NT + NV - H + 1)); te_o = list(range(NT + NV, len(X) - H + 1))
    origins = tr_o + va_o + te_o; kind = np.r_[np.zeros(len(tr_o)), np.ones(len(va_o)), 2 * np.ones(len(te_o))].astype(int)
    with Pool(2) as pool: res = pool.map(build, [(j, origins, *params[j]) for j in range(8)])
    Dx = np.stack([np.diff(X, axis=0, prepend=X[:1])[o - PL:o][::-1] for o in origins])       # (n, PL, 8) recent changes, lag 1..PL
    raw_in = np.stack([[r[0] for r in res[j]] for j in range(8)], 1)    # (n, 8, L)
    psr = np.stack([[r[1] for r in res[j]] for j in range(8)], 1)
    tda = np.stack([[r[2] for r in res[j]] for j in range(8)], 1)
    Yt = np.stack([np.stack([X[o:o + H, j] - X[o - 1, j] for j in range(8)]) for o in origins])  # (n, 8, H)
    last = np.stack([X[o - 1] for o in origins])
    np.savez_compressed(f'd2_H{H}_{COND}.npz', raw=raw_in, psr=psr, tda=tda, Dx=Dx, Y=Yt, last=last, kind=kind, origins=np.array(origins),
                        TE=T, TEp=Pv, A_te=A_te, A_pc=A_pc, mu=mu, sd=sd, Xtrue=(np.loadtxt('../data/exchange_rate.txt.gz', delimiter=',') - mu) / sd,
                        params=np.array(params), t_cg=t_cg)
    print(COND, H, 'n', len(origins), 'time', round(time.time() - t0, 1), 'cg_time', round(t_cg, 1), 'params', params)
    print('TE sig edges', Sig.sum(), 'A_te edges', A_te.sum() - np.trace(A_te), 'A_pc edges', A_pc.sum() - np.trace(A_pc))
    print(np.round(T, 4)); print(A_te); print(A_pc)
