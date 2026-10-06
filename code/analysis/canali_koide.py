import os
import numpy as np, sys, itertools, time
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'));import mt_trasparenza as mt
from scipy.spatial import Delaunay
mi=mt.mi;t0=time.time()
def sbil(X,L):
    n=len(X);m=3.5*(L/n**(1/3));ext,orig=[X],[np.arange(n)]
    for sh in itertools.product((-1,0,1),repeat=3):
        if sh==(0,0,0):continue
        mask=np.ones(n,bool)
        for ax in range(3):
            if sh[ax]==1:mask&=X[:,ax]<m
            elif sh[ax]==-1:mask&=X[:,ax]>L-m
        idx=np.where(mask)[0];ext.append(X[idx]+np.array(sh)*L);orig.append(idx)
    P=np.vstack(ext);O=np.concatenate(orig);S=Delaunay(P).simplices;c=P[S].mean(1);S=np.sort(O[S[np.all((c>=0)&(c<L),axis=1)]],axis=1)
    T=np.unique(np.sort(S[:,[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]].reshape(-1,3),axis=1),axis=0)
    def ang(a,b,c):
        u=mi(X[b]-X[a],L);v=mi(X[c]-X[a],L);return np.arccos(np.clip((u*v).sum(1)/np.linalg.norm(u,axis=1)/np.linalg.norm(v,axis=1),-1,1))
    A=np.stack([ang(T[:,0],T[:,1],T[:,2]),ang(T[:,1],T[:,0],T[:,2]),ang(T[:,2],T[:,0],T[:,1])],1);d=np.pi-A-2*np.pi/3
    a=np.sqrt(2/3*(d**2).sum(1));return np.median(a),a.mean(),d.std()
z=np.load('reticoli/crescita4096.npz');print(f"crescita (4096):            mediana {sbil(z['X'],float(z['L']))[0]:.4f}  media {sbil(z['X'],float(z['L']))[1]:.4f}")
for phi in [0.60,0.70,0.80]:
    mt.PHI=phi;mt.RHO=phi/(np.pi/6)
    rng=np.random.default_rng(1);X,L=mt.ret_casuale(8000,rng);X,fn=mt.rilassa(X,L,passi=3000)
    md,me,sd=sbil(X,L);print(f"equilibrio, riempimento {phi:.2f}: mediana {md:.4f}  media {me:.4f}  (deviazione standard {sd:.4f})   [{time.time()-t0:.0f}s]")
print(f"Koide: 2/9 = {2/9:.4f}")
