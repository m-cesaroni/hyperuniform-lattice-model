import os
import numpy as np, sys, time
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'));import mt_trasparenza as mt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'));import kdlib
from scipy.spatial import Delaunay
t0=time.time();mi=mt.mi
def misure(X,L):
    C=kdlib.build(X,L);n,E,F,T=C['sizes'];h1=C['h'][1];K=C['K'];D0=K[n:n+E,:n].tocsr()
    inc=(abs(D0)>0).astype(float).T.tocsr()          # celle x varchi
    Neff=[];
    for i in range(n):
        w=h1[inc[i].indices];Neff.append(w.sum()**2/(w*w).sum())
    # sbilanciamento dei canali (angoli dei triangoli di Delaunay)
    m=3.5*(L/n**(1/3));import itertools
    ext,orig=[X],[np.arange(n)]
    for sh in itertools.product((-1,0,1),repeat=3):
        if sh==(0,0,0):continue
        mask=np.ones(n,bool)
        for ax in range(3):
            if sh[ax]==1:mask&=X[:,ax]<m
            elif sh[ax]==-1:mask&=X[:,ax]>L-m
        idx=np.where(mask)[0];ext.append(X[idx]+np.array(sh)*L);orig.append(idx)
    P=np.vstack(ext);O=np.concatenate(orig);S=Delaunay(P).simplices;c=P[S].mean(1);S=np.sort(O[S[np.all((c>=0)&(c<L),axis=1)]],axis=1)
    Tr=np.unique(np.sort(S[:,[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]].reshape(-1,3),axis=1),axis=0)
    def ang(a,b,cc):
        u=mi(X[b]-X[a],L);v=mi(X[cc]-X[a],L);return np.arccos(np.clip((u*v).sum(1)/np.linalg.norm(u,axis=1)/np.linalg.norm(v,axis=1),-1,1))
    A=np.stack([ang(Tr[:,0],Tr[:,1],Tr[:,2]),ang(Tr[:,1],Tr[:,0],Tr[:,2]),ang(Tr[:,2],Tr[:,0],Tr[:,1])],1);d=np.pi-A-2*np.pi/3;amp=np.sqrt(2/3*(d**2).sum(1))
    return np.mean(Neff),np.median(amp),np.mean(amp)
print("riempimento   N vicine rigide   1/N^2        sbilanciamento canali (mediana / media)")
print(f"   alpha misurata: N = {1/np.sqrt(1/137.036):.3f}   ->  1/137.036             Koide: 0,2222")
for phi in [0.60,0.65,0.70,0.75,0.80]:
    mt.PHI=phi;mt.RHO=phi/(np.pi/6)
    rng=np.random.default_rng(1);X,L=mt.ret_casuale(4096,rng);X,fn=mt.rilassa(X,L,passi=2500)
    N,md,me=misure(X,L)
    print(f"   {phi:.2f}         {N:6.2f}           1/{N*N:6.1f}      {md:.4f} / {me:.4f}     [{time.time()-t0:.0f}s]")
