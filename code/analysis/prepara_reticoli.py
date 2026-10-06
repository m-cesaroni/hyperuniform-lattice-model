"""Genera i reticoli usati dagli script di analisi (cartella reticoli/):
   equilibrio20000.pkl  — 20 000 celle casuali portate all'equilibrio (riempimento 0,70)
   crescita4096.npz     — 4 096 celle cresciute per nascite adiacenti e assestate
   equilibrio60000.pkl  — 60 000 celle in equilibrio (per commutatore_lyman.py)
Tempi: pochi minuti il primo e il secondo, 3-5 minuti il terzo."""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import mt_trasparenza as mt
os.makedirs('reticoli', exist_ok=True)
rng = np.random.default_rng(1); X, L = mt.ret_casuale(20000, rng); X, fn = mt.rilassa(X, L, passi=4000)
passo = L / 20000 ** (1 / 3); pickle.dump((X, L, passo, None), open('reticoli/equilibrio20000.pkl', 'wb')); print('equilibrio20000 ok', fn)
rng = np.random.default_rng(2); X, L = mt.ret_crescita(4096, rng); np.savez('reticoli/crescita4096.npz', X=X, L=L); print('crescita4096 ok')
rng = np.random.default_rng(11); X, L = mt.ret_casuale(60000, rng); X, fn = mt.rilassa(X, L, passi=1800)
pickle.dump((X, L), open('reticoli/equilibrio60000.pkl', 'wb')); print('equilibrio60000 ok', fn)
