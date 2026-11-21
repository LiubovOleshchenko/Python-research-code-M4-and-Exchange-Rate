"""Train/evaluate the proposed method and its ablations on D1. Usage: python run_d1.py variant seed [tag]"""
import sys, time, numpy as np, torch
from common import train_node, predict, conformal_q
torch.set_num_threads(1)
variant, seed = sys.argv[1], int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else ''
D = np.load(f'd1{tag}.npz'); kind = D['kind']
tda = D['tda']; tda = (tda - tda[kind == 0].mean(0)) / (tda[kind == 0].std(0) + 1e-8)
if variant == 'full':      U, C = D['psr'], tda          # PSR state + TDA conditioning
elif variant == 'noTDA':   U, C = D['psr'], None         # (b) topological features removed
elif variant == 'nodeonly': U, C = D['raw'], None        # (e) Neural ODE on raw lag window only
elif variant == 'rawTDA':  U, C = D['raw'], tda          # PSR removed, TDA kept
Y = D['y'].astype(np.float32)
t0 = time.time()
m = train_node(U[kind == 0], C[kind == 0] if C is not None else None, Y[kind == 0], 8, seed)
tt = time.time() - t0
pc = predict(m, U[kind == 1], C[kind == 1] if C is not None else None)
q = conformal_q(Y[kind == 1] - pc, 0.1)
t1 = time.time(); pt = predict(m, U[kind == 2], C[kind == 2] if C is not None else None); ti = time.time() - t1
s, last = D['scale'][:, None], D['last'][:, None]
yhat = last + s * pt; lo, hi = yhat - s * q, yhat + s * q
np.savez_compressed(f'res/d1{tag}_{variant}_{seed}.npz', yhat=yhat, lo=lo, hi=hi, train_time=tt, infer_time=ti, epochs=m.epochs_run)
print(variant, seed, 'train %.1fs' % tt, 'epochs', m.epochs_run, 'q', q.round(3))
