import os
import numpy as np, pickle, sys, time
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'));import mt_trasparenza as mt
from scipy.spatial import cKDTree, Delaunay
mi=mt.mi;t0=time.time();SIG=1.0
X0,L,passo,pr0=pickle.load(open('reticoli/equilibrio20000.pkl','rb'));n=len(X0)
rng=np.random.default_rng(5)
# tre tessuti alla stessa densita'
Xr,_=mt.ret_casuale(n,rng)                       # casuale (nessun equilibrio)
Xh,_=mt.rilassa(Xr.copy(),L,passi=150)           # parzialmente equilibrato
Xe=X0                                            # equilibrio
def forze(X,pairs):
    d=mi(X[pairs[:,1]]-X[pairs[:,0]],L);r=np.linalg.norm(d,axis=1);m=r<SIG
    f=np.zeros(len(r));f[m]=(1-r[m]/SIG)/SIG;fv=(f/np.maximum(r,1e-12))[:,None]*d
    F=np.zeros_like(X);np.add.at(F,pairs[:,0],-fv);np.add.at(F,pairs[:,1],fv)
    U=0.5*np.sum((1-r[m]/SIG)**2);return F,U
def evolvi(X,V,T,dt=0.02,skin=0.4):
    X=X.copy();V=V.copy();ref=X.copy();pairs=cKDTree(X,boxsize=L).query_pairs(SIG+skin,output_type='ndarray')
    F,U=forze(X,pairs);out=[]
    for s in range(int(T/dt)):
        V+=0.5*dt*F;X=(X+dt*V)%L
        if np.max(np.linalg.norm(mi(X-ref,L),axis=1))>skin/2:pairs=cKDTree(X,boxsize=L).query_pairs(SIG+skin,output_type='ndarray');ref=X.copy()
        F,U=forze(X,pairs);V+=0.5*dt*F
        if (s+1)%25==0:out.append((X.copy(),V.copy()))
    return out
c=np.array([L/2]*3);i0=int(np.argmin(np.linalg.norm(mi(Xe-c,L),axis=1)))
def esperimento(X,nome,direz,v0=0.6,T=6.0):
    # senza urto (riferimento) e con urto
    base=evolvi(X,np.zeros_like(X),T);V=np.zeros_like(X);V[i0]=v0*direz;urto=evolvi(X,V,T)
    E0=0.5*v0**2;d0=np.linalg.norm(mi(X-X[i0],L),axis=1);rows=[]
    for k,((Xb,Vb),(Xu,Vu)) in enumerate(zip(base,urto)):
        dK=0.5*((Vu**2).sum(1)-(Vb**2).sum(1))          # energia cinetica dovuta all'urto, cella per cella
        loc=dK[d0<1.6].sum()/E0;lont=dK[d0>=1.6].sum()/E0
        # spostamento permanente indotto dall'urto (differenza fra le due traiettorie)
        dX=np.linalg.norm(mi(Xu-Xb,L),axis=1)
        rows.append(((k+1)*0.5,loc,lont,dX[d0>=1.6].mean(),dK.sum()/E0))
    return rows
print(f"celle {n}; cella colpita {i0}; impulso: energia 0.18")
# direzioni: faccia (verso la vicina piu' prossima), canale, vuoto, sul reticolo in equilibrio
T=cKDTree(Xe,boxsize=L);dd,ii=T.query(Xe[i0],k=14)
nface=mi(Xe[ii[1]]-Xe[i0],L);nface/=np.linalg.norm(nface)
# canale: centroide del triangolo (i0, due vicine adiacenti fra loro)
best=None
for a in range(1,8):
    for b in range(a+1,8):
        if np.linalg.norm(mi(Xe[ii[a]]-Xe[ii[b]],L))<1.35*passo:best=(a,b);break
    if best:break
nchan=(mi(Xe[ii[best[0]]]-Xe[i0],L)+mi(Xe[ii[best[1]]]-Xe[i0],L))/3;nchan/=np.linalg.norm(nchan)
# vuoto: direzione verso il circocentro del tetraedro piu' vicino che contiene i0 (vertice di Voronoi)
loc=np.where(np.linalg.norm(mi(Xe-Xe[i0],L),axis=1)<2.5*passo)[0];P=mi(Xe[loc]-Xe[i0],L)
tri=Delaunay(P);k0=int(np.where(loc==i0)[0][0]);tets=tri.simplices[np.any(tri.simplices==k0,axis=1)]
Q=P[tets[0]];a=Q[0];A=2*(Q[1:]-a);bvec=(Q[1:]**2).sum(1)-(a**2).sum();cc=np.linalg.solve(A,bvec)
nvoid=cc-P[k0];nvoid/=np.linalg.norm(nvoid)
print(f"direzioni: faccia-canale {np.degrees(np.arccos(abs(nface@nchan))):.0f} gradi, faccia-vuoto {np.degrees(np.arccos(abs(nface@nvoid))):.0f} gradi   [{time.time()-t0:.0f}s]")
res={}
for nome,X in [("equilibrio",Xe),("parziale",Xh),("casuale",Xr)]:
    res[nome]=esperimento(X,nome,nface);print(f"  {nome} fatto [{time.time()-t0:.0f}s]")
for nome,dz in [("eq. verso canale",nchan),("eq. verso vuoto",nvoid)]:
    res[nome]=esperimento(Xe,nome,dz);print(f"  {nome} fatto [{time.time()-t0:.0f}s]")
pickle.dump(res,open('urto.pkl','wb'))
print("\nenergia dell'urto: quota vicino alla cella colpita (r<1.6) / quota lontana (onda) / spostamento permanente indotto lontano / totale\n")
for nome,rows in res.items():
    print(nome)
    for t,loc,lont,disp,tot in rows[::3]:print(f"   t={t:4.1f}:  vicino {loc:6.3f}   onda {lont:6.3f}   spostamento indotto {disp:.2e}   totale {tot:6.3f}")
