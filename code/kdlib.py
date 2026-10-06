import numpy as np, scipy.sparse as sp, sys, itertools
sys.path.insert(0,'/mnt/user-data/outputs')
import mt_trasparenza as mt
from scipy.spatial import cKDTree, Delaunay
mi=mt.mi
def build(X,L,floor=1e-3):
    n=len(X);passo=float(np.mean(cKDTree(X,boxsize=L).query(X,k=2)[0][:,1]))
    m=3.5*passo;ext,orig=[X],[np.arange(n)]
    for sh in itertools.product((-1,0,1),repeat=3):
        if sh==(0,0,0):continue
        mask=np.ones(n,bool)
        for ax in range(3):
            if sh[ax]==1:mask&=X[:,ax]<m
            elif sh[ax]==-1:mask&=X[:,ax]>L-m
        idx=np.where(mask)[0];ext.append(X[idx]+np.array(sh)*L);orig.append(idx)
    P=np.vstack(ext);O=np.concatenate(orig);S0=Delaunay(P).simplices;Q=P[S0];cen=Q.mean(1)
    keep=np.all((cen>=0)&(cen<L),axis=1);Q=Q[keep];S=np.sort(O[S0[keep]],axis=1);tcen=Q.mean(1)%L
    a=Q[:,0];Am=2*(Q[:,1:]-a[:,None,:]);bb=(Q[:,1:]**2).sum(2)-(a**2).sum(1)[:,None];cc=np.linalg.solve(Am,bb[...,None])[...,0]
    T=len(S);tvol=np.abs(np.einsum('ij,ij->i',Q[:,1]-a,np.cross(Q[:,2]-a,Q[:,3]-a)))/6
    pa=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    ek=np.concatenate([S[:,i].astype(np.int64)*n+S[:,j] for i,j in pa]);ue,einv=np.unique(ek,return_inverse=True);E=len(ue)
    edges=np.stack([ue//n,ue%n],1);dv=mi(X[edges[:,1]]-X[edges[:,0]],L);el=np.linalg.norm(dv,axis=1);em=X[edges[:,0]]+dv/2
    cce=np.tile(cc,(6,1));rel=mi(cce-em[einv],L);t=dv[einv]/el[einv][:,None]
    e1=np.cross(t,np.where(np.abs(t[:,:1])<0.9,[[1.0,0,0]],[[0,1.0,0]]));e1/=np.linalg.norm(e1,axis=1)[:,None];e2=np.cross(t,e1)
    x=(rel*e1).sum(1);y=(rel*e2).sum(1);ang=np.arctan2(y,x);o=np.lexsort((ang,einv));x,y,gi=x[o],y[o],einv[o]
    ini=np.r_[0,np.where(np.diff(gi))[0]+1];fin=np.r_[ini[1:],len(gi)];nx=np.arange(len(gi))+1;nx[fin-1]=ini
    Avor=0.5*np.abs(np.bincount(gi,x*y[nx]-x[nx]*y,minlength=E))
    tr=[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    tk=np.concatenate([(S[:,i].astype(np.int64)*n+S[:,j])*n+S[:,k] for i,j,k in tr]);ut,tinv=np.unique(tk,return_inverse=True);F=len(ut)
    tris=np.stack([ut//(n*n),(ut//n)%n,ut%n],1)
    cct=np.tile(cc,(4,1));o2=np.argsort(tinv,kind='stable');lv=np.zeros(F);lv[tinv[o2[0::2]]]=np.linalg.norm(mi(cct[o2[1::2]]-cct[o2[0::2]],L),axis=1)
    va=mi(X[tris[:,1]]-X[tris[:,0]],L);vb=mi(X[tris[:,2]]-X[tris[:,0]],L);ta=0.5*np.linalg.norm(np.cross(va,vb),axis=1);tc=(X[tris[:,0]]+(va+vb)/3)%L
    eidx=lambda p,q:np.searchsorted(ue,p.astype(np.int64)*n+q)
    tidx=lambda p,q,r:np.searchsorted(ut,(p.astype(np.int64)*n+q)*n+r)
    d0=sp.csr_matrix((np.r_[-np.ones(E),np.ones(E)],(np.r_[np.arange(E),np.arange(E)],np.r_[edges[:,0],edges[:,1]])),shape=(E,n))
    d1=sp.csr_matrix((np.tile([1.,1.,-1.],F),(np.repeat(np.arange(F),3),np.stack([eidx(tris[:,0],tris[:,1]),eidx(tris[:,1],tris[:,2]),eidx(tris[:,0],tris[:,2])],1).ravel())),shape=(F,E))
    d2=sp.csr_matrix((np.tile([1.,-1.,1.,-1.],T),(np.repeat(np.arange(T),4),np.stack([tidx(S[:,1],S[:,2],S[:,3]),tidx(S[:,0],S[:,2],S[:,3]),tidx(S[:,0],S[:,1],S[:,3]),tidx(S[:,0],S[:,1],S[:,2])],1).ravel())),shape=(T,F))
    V0=np.zeros(n);np.add.at(V0,edges[:,0],Avor*el/6);np.add.at(V0,edges[:,1],Avor*el/6)
    h=[V0,np.maximum(Avor/el,floor*np.median(Avor/el)),np.maximum(lv/ta,floor*np.median(lv/ta)),1/tvol]
    D0=sp.diags(np.sqrt(h[1]))@d0@sp.diags(1/np.sqrt(h[0]));D1=sp.diags(np.sqrt(h[2]))@d1@sp.diags(1/np.sqrt(h[1]));D2=sp.diags(np.sqrt(h[3]))@d2@sp.diags(1/np.sqrt(h[2]))
    K=sp.bmat([[None,D0.T,None,None],[D0,None,D1.T,None],[None,D1,None,D2.T],[None,None,D2,None]]).tocsr()
    sizes=[n,E,F,T];deg=np.concatenate([np.full(s,p) for p,s in enumerate(sizes)])
    G=sp.diags(np.where(deg%2==0,1.0,-1.0))
    pos=np.vstack([X,em%L,tc,tcen])
    return dict(K=K,G=G,deg=deg,sizes=sizes,pos=pos,h=h,edges=edges,dv=dv,em=em,passo=passo,L=L,n=n)
