"""Proposed method / ablations on D2. Usage: python run_d2.py variant seed H cond"""
import sys, time, numpy as np, torch
from common import train_node, predict, conformal_q
torch.set_num_threads(1)
variant, seed, H, cond = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
D = np.load(f'd2_H{H}_{cond}.npz'); kind = D['kind']; n = len(kind)
TE = D['TE']; Wte = TE / TE.max()
def parents(mask, w):  # (n, 8 targets, 8*PL): changes of variable i weighted by mask[i,j]*w[i,j]; own variable excluded
    Dx = D['Dx']; out = np.zeros((n, 8, 8 * Dx.shape[1]), np.float32)
    for j in range(8):
        g = (mask[:, j] * w[:, j]).astype(np.float32); g[j] = 0
        out[:, j] = (Dx * g[None, None, :]).transpose(0, 2, 1).reshape(n, -1)
    return out
onehot = np.broadcast_to(np.eye(8, dtype=np.float32), (n, 8, 8))
tda = D['tda']; tm = tda[kind == 0].reshape(-1, 6); tda = (tda - tm.mean(0)) / (tm.std(0) + 1e-8)
ones = np.ones((8, 8))
if variant == 'full':     U = np.concatenate([D['psr'], parents(D['A_te'], Wte), onehot], -1); C = tda
if variant == 'noTDA':    U = np.concatenate([D['psr'], parents(D['A_te'], Wte), onehot], -1); C = None
if variant == 'noCG':     U = np.concatenate([D['psr'], parents(ones, Wte), onehot], -1); C = tda      # (c) causal graph removed: all sources, TE-weighted
if variant == 'noTE':     U = np.concatenate([D['psr'], parents(D['A_pc'], ones), onehot], -1); C = tda  # (d) TE removed: PC graph, unweighted
if variant == 'nodeonly': U = np.concatenate([D['raw'], onehot], -1); C = None                       # (e) Neural ODE only
U = U.astype(np.float32); Y = D['Y'].astype(np.float32)
fl = lambda a, k: a[kind == k].reshape(-1, a.shape[-1]) if a is not None else None
tr_idx = np.where(kind == 0)[0][::2]                                                                 # stride-2 training origins
fl0 = lambda a: a[tr_idx].reshape(-1, a.shape[-1]) if a is not None else None
t0 = time.time(); m = train_node(fl0(U), fl0(C), fl0(Y), H, seed, epochs=20, dt=1.0); tt = time.time() - t0
q = conformal_q(fl(Y, 1) - predict(m, fl(U, 1), fl(C, 1)), 0.1)
t1 = time.time(); pt = predict(m, fl(U, 2), fl(C, 2)).reshape(-1, 8, H); ti = time.time() - t1
o = D['origins'][kind == 2]; Xt = D['Xtrue']
ytrue = np.stack([Xt[t:t + H].T for t in o])                         # clean ground truth (standardized units)
yhat = D['last'][kind == 2][:, :, None] + pt
np.savez_compressed(f'res/d2_H{H}_{cond}_{variant}_{seed}.npz', yhat=yhat, ytrue=ytrue, lo=yhat - q, hi=yhat + q, q=q,
                    train_time=tt, infer_time=ti, epochs=m.epochs_run)
print(variant, seed, H, cond, 'train %.1f' % tt, 'ep', m.epochs_run, 'MAE %.4f' % np.abs(ytrue - yhat).mean(),
      'naive %.4f' % np.abs(ytrue - D['last'][kind == 2][:, :, None]).mean(), 'cov %.3f' % np.mean((ytrue >= yhat - q) & (ytrue <= yhat + q)))
