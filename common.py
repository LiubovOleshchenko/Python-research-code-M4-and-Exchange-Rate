"""Core components of the PSR-TDA-TE-CG-NODE-CP forecasting method (re-implementation used for the revision experiments)."""
import numpy as np, torch, torch.nn as nn
from ripser import ripser
from scipy import stats

# ---------------- Phase-space reconstruction ----------------
def ami(x, lag, bins=8):
    a, b = x[:-lag], x[lag:]
    h, _, _ = np.histogram2d(a, b, bins=bins)
    p = h / h.sum(); px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())

def choose_tau(x, max_tau=4):
    """First local minimum of average mutual information (Fraser-Swinney); fallback: argmin."""
    if len(x) < 8: return 1
    v = [ami(x, l) for l in range(1, max_tau + 1)]
    for i in range(1, len(v) - 1):
        if v[i] < v[i - 1] and v[i] <= v[i + 1]: return i + 1
    return int(np.argmin(v)) + 1

def embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    return np.stack([x[(m - 1 - j) * tau:(m - 1 - j) * tau + n] for j in range(m)], 1)  # columns: x_t, x_{t-tau}, ...

def choose_m(x, tau, max_m=4, rtol=10.0, thr=0.1):
    """False nearest neighbours (Kennel et al.): smallest m with FNN fraction < thr."""
    for m in range(1, max_m):
        if len(x) - m * tau < 5: return m
        E = embed(x, m + 1, tau); Em = E[:, :m]
        d = ((Em[:, None, :] - Em[None]) ** 2).sum(-1); np.fill_diagonal(d, np.inf)
        nn_ = d.argmin(1); dm = np.sqrt(d[np.arange(len(E)), nn_]) + 1e-12
        extra = np.abs(E[:, m] - E[nn_, m])
        if np.mean(extra / dm > rtol) < thr: return m
    return max_m

# ---------------- Topological features ----------------
def tda_features(cloud):
    """Vietoris-Rips persistent homology (H0, H1) -> 6 summary features."""
    if cloud.shape[1] < 2: cloud = np.c_[cloud, np.r_[cloud[1:, 0], cloud[-1, 0]]]
    dg = ripser(cloud, maxdim=1)['dgms']
    out = []
    for k in (0, 1):
        d = dg[k]; d = d[np.isfinite(d[:, 1])] if len(d) else d
        p = (d[:, 1] - d[:, 0]) if len(d) else np.zeros(1)
        p = p[p > 0] if (p > 0).any() else np.zeros(1)
        tot = p.sum(); q = p / tot if tot > 0 else np.ones_like(p) / len(p)
        ent = float(-(q * np.log(q + 1e-12)).sum())
        out += [tot, p.max(), ent]
    return np.array(out, dtype=np.float32)

# ---------------- Transfer entropy & causal graph ----------------
def _disc(x, b=3):
    q = np.quantile(x, np.linspace(0, 1, b + 1)[1:-1]); return np.digitize(x, q)

def transfer_entropy(x, y, b=3, lag=1):
    """TE_{X->Y} (k=l=1), histogram estimator on quantile bins, in nats."""
    xd, yd = _disc(x, b), _disc(y, b)
    yf, yp, xp = yd[lag:], yd[:-lag], xd[:-lag]
    n = len(yf); idx = (yf * b + yp) * b + xp
    p3 = np.bincount(idx, minlength=b ** 3).reshape(b, b, b) / n
    p_yp_xp = p3.sum(0); p_yf_yp = p3.sum(2); p_yp = p3.sum((0, 2))
    te = 0.0
    for i in range(b):
        for j in range(b):
            for k in range(b):
                if p3[i, j, k] > 0:
                    te += p3[i, j, k] * np.log(p3[i, j, k] * p_yp[j] / (p_yp_xp[j, k] * p_yf_yp[i, j]))
    return te

def te_matrix(X, n_surr=100, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    d = X.shape[1]; T = np.zeros((d, d)); P = np.ones((d, d))
    for i in range(d):
        for j in range(d):
            if i == j: continue
            te = transfer_entropy(X[:, i], X[:, j])
            sur = [transfer_entropy(np.roll(X[:, i], rng.integers(20, len(X) - 20)), X[:, j]) for _ in range(n_surr)]
            T[i, j] = te; P[i, j] = (1 + np.sum(np.array(sur) >= te)) / (n_surr + 1)
    return T, P, (P < alpha)

def _pcorr_p(y, x, Z):
    if Z.shape[1]:
        beta = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z], np.c_[y, x], rcond=None)[0]
        r = np.c_[y, x] - np.c_[np.ones(len(Z)), Z] @ beta
    else:
        r = np.c_[y, x] - np.c_[y, x].mean(0)
    rho = np.corrcoef(r[:, 0], r[:, 1])[0, 1]
    n = len(y) - Z.shape[1] - 3
    z = 0.5 * np.log((1 + rho) / (1 - rho + 1e-12)) * np.sqrt(max(n, 1))
    return 2 * (1 - stats.norm.cdf(abs(z)))

def pc_lagged(X, max_lag=4, alpha=0.05, candidates=None, max_cond=2):
    """Lagged PC-style parent discovery (PCMCI-like condition-selection step) with Fisher-z partial-correlation tests.
    Returns adjacency A[i,j]=1 if variable i (at some lag 1..max_lag) is a parent of j."""
    T, d = X.shape
    lagged = {(i, l): X[max_lag - l:T - l, i] for i in range(d) for l in range(1, max_lag + 1)}
    A = np.zeros((d, d), int)
    for j in range(d):
        y = X[max_lag:, j]
        cand = [(i, l) for (i, l) in lagged if (candidates is None or candidates[i, j] or i == j)]
        for k in range(max_cond + 1):
            keep = []
            for c in cand:
                others = [o for o in cand if o != c]
                # condition on the k strongest other candidates
                if k and others:
                    sc = [abs(np.corrcoef(lagged[o], y)[0, 1]) for o in others]
                    Z = np.stack([lagged[others[t]] for t in np.argsort(sc)[::-1][:k]], 1)
                else: Z = np.zeros((len(y), 0))
                if _pcorr_p(y, lagged[c], Z) < alpha: keep.append(c)
            cand = keep
        for (i, l) in cand: A[i, j] = 1
    return A

# ---------------- Neural ODE forecaster ----------------
class ODEFunc(nn.Module):
    def __init__(s, d, c, hdim=64):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d + c, hdim), nn.Tanh(), nn.Linear(hdim, hdim), nn.Tanh(), nn.Linear(hdim, d))
    def forward(s, z, c): return s.net(torch.cat([z, c], -1))

class NODEForecaster(nn.Module):
    """z(0)=g_phi(u); dz/dt=f_theta(z,c); y_hat(t_k)=W z(t_k)+b; RK4 with step dt."""
    def __init__(s, in_dim, cond_dim, h, d=32, dt=0.5):
        super().__init__(); s.h, s.dt = h, dt
        s.enc = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(), nn.Linear(64, d))
        s.cond = nn.Linear(max(cond_dim, 1), 8) if cond_dim else None
        s.f = ODEFunc(d, 8 if cond_dim else 0); s.dec = nn.Linear(d, 1)
    def forward(s, u, c=None):
        z = s.enc(u); cc = s.cond(c) if s.cond is not None else z[:, :0]
        outs, n = [], int(round(1 / s.dt))
        for _ in range(s.h):
            for _ in range(n):
                k1 = s.f(z, cc); k2 = s.f(z + .5 * s.dt * k1, cc); k3 = s.f(z + .5 * s.dt * k2, cc); k4 = s.f(z + s.dt * k3, cc)
                z = z + s.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            outs.append(s.dec(z))
        return torch.cat(outs, 1)

def train_node(U, C, Y, h, seed, epochs=25, bs=512, lr=1e-3, wd=1e-4, patience=4, dt=0.5, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(U); rng = np.random.default_rng(seed); idx = rng.permutation(n); nv = max(int(.1 * n), 1)
    va, tr = idx[:nv], idx[nv:]
    U, Y = torch.tensor(U, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    C = torch.tensor(C, dtype=torch.float32) if C is not None else None
    m = NODEForecaster(U.shape[1], C.shape[1] if C is not None else 0, h, dt=dt)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, bad, state, ep_run = np.inf, 0, None, 0
    for ep in range(epochs):
        m.train(); perm = tr[rng.permutation(len(tr))]
        for b in range(0, len(perm), bs):
            bi = perm[b:b + bs]
            loss = (m(U[bi], C[bi] if C is not None else None) - Y[bi]).abs().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad(): vl = (m(U[va], C[va] if C is not None else None) - Y[va]).abs().mean().item()
        ep_run = ep + 1
        if verbose: print(ep, vl, flush=True)
        if vl < best - 1e-5: best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); m.eval(); m.epochs_run = ep_run
    return m

def predict(m, U, C=None):
    with torch.no_grad():
        return m(torch.tensor(U, dtype=torch.float32), torch.tensor(C, dtype=torch.float32) if C is not None else None).numpy()

# ---------------- Split conformal ----------------
def conformal_q(res_cal, alpha=0.1):
    """Per-horizon (1-alpha) quantile of absolute calibration residuals with finite-sample correction."""
    n = res_cal.shape[0]; k = int(np.ceil((n + 1) * (1 - alpha)))
    s = np.sort(np.abs(res_cal), 0); return s[min(k, n) - 1]

# ---------------- Metrics ----------------
def metrics(y, yh, eps=1e-8):
    e = y - yh
    return dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)),
                MAPE=100 * np.mean(np.abs(e) / np.maximum(np.abs(y), eps)),
                sMAPE=200 * np.mean(np.abs(e) / np.maximum(np.abs(y) + np.abs(yh), eps)))
