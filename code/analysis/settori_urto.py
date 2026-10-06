import os
import numpy as np, sys, time, scipy.sparse as sp
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'));import kdlib
from scipy.sparse.linalg import cg
t0=time.time()
z=np.load('reticoli/crescita4096.npz');X=z['X'];L=float(z['L'])
C=kdlib.build(X,L);K=C['K'];n,E,F,T=C['sizes'];h=C['h']
D0=K[n:n+E,:n].tocsr();D1=K[n+E:n+E+F,n:n+E].tocsr();D2=K[n+E+F:,n+E:n+E+F].tocsr()
L0=(D0.T@D0).tocsr();L3=(D2@D2.T).tocsr()
rng=np.random.default_rng(2)
def quota_luce_1forma(a):
    # parte longitudinale (che tocca le celle): D0 phi con (D0^T D0) phi = D0^T a ; il resto e' trasversale (luce)
    phi,_=cg(L0,D0.T@a,rtol=1e-10,maxiter=4000);al=D0@phi
    return 1-(al@al)/(a@a)
def quota_luce_2forma(b):
    # parte che tocca i vuoti: D2^T gamma con (D2 D2^T) gamma = D2 b ; il resto e' chiusa (luce)
    gam,_=cg(L3,D2@b,rtol=1e-10,maxiter=4000);bc=D2.T@gam
    return 1-(bc@bc)/(b@b)
print(f"reticolo di MT in equilibrio: {n} celle, {E} varchi, {F} canali, {T} vuoti   [{time.time()-t0:.0f}s]")
print("\nurto su un solo elemento: quota dell'energia che va nel settore della luce (varchi e canali che si rincorrono)")
for nome,fn,size in [("varco (campo su un collegamento)",quota_luce_1forma,E),("canale (campo su un triangolo)",quota_luce_2forma,F)]:
    q=[fn(np.eye(1,size,rng.integers(size)).ravel()) for _ in range(12)]
    print(f"  {nome:34s}: luce {np.mean(q)*100:5.1f}%  (da {np.min(q)*100:.0f} a {np.max(q)*100:.0f}),  settori con celle o vuoti {100-np.mean(q)*100:5.1f}%")
print(f"  {'cella':34s}: luce   0.0%,  settore celle-varchi 100%   (per struttura)")
print(f"  {'vuoto':34s}: luce   0.0%,  settore canali-vuoti 100%   (per struttura)")
# urto distribuito su una piccola regione invece che su un elemento solo
c=np.array([L/2]*3);rc=np.linalg.norm(kdlib.mt.mi(C['em']-c,L),axis=1);w=np.exp(-rc**2/(2*1.5**2))
for nome,a in [("varchi di una regione, tutti nello stesso verso (campo uniforme)",w*C['dv'][:,2]*np.sqrt(h[1])),("varchi di una regione, versi casuali",w*rng.normal(size=E)*np.sqrt(h[1]))]:
    print(f"  {nome:62s}: luce {quota_luce_1forma(a)*100:5.1f}%")
print(f"[{time.time()-t0:.0f}s]")
