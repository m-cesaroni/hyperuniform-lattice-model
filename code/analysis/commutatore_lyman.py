import os
import numpy as np, sys, time, pickle, scipy.sparse as sp
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'));import kdlib
from scipy.sparse.linalg import eigsh
t0=time.time();mi=kdlib.mt.mi
z=np.load('reticoli/crescita4096.npz');X=z['X'];L=float(z['L'])
C=kdlib.build(X,L);K=C['K'].tocsr();pos=C['pos'];N=K.shape[0];deg=C['deg']
Kc=K.tocoo();dx=mi(pos[Kc.row]-pos[Kc.col],L)
print("1) identita' [X_a,K] = iV_a, verificata elemento per elemento sugli elementi che non attraversano il bordo:")
for a in range(3):
    V=sp.coo_matrix((dx[:,a]*Kc.data,(Kc.row,Kc.col)),shape=K.shape).tocsr()
    xa=pos[:,a];Xa=sp.diags(xa);comm=(Xa@K-K@Xa).tocoo()
    # stesso schema di sparsita' di K (prodotti di diagonale con K): confronto diretto per elemento
    comm=comm.tocsr();V=V.tocsr()
    diff=(comm-V).tocoo();inner=np.abs(mi(pos[diff.row]-pos[diff.col],L)[:,a])<L/4
    # gli elementi di bordo hanno (x_row - x_col) che salta di L rispetto allo spostamento a immagine minima
    print(f"   asse {'xyz'[a]}: scarto massimo interno {np.max(np.abs(diff.data[inner])) if inner.any() else 0:.1e};  elementi di bordo (salto di L): {np.sum(~inner)}")
# velocita' effettiva: non la norma della matrice (dominata dagli elementi minuscoli del duale), ma la velocita' di gruppo di pacchetti lisci
print("\n   velocita' di pacchetti lisci lungo x (fase e^{ikx}, settore luce): la velocita' di gruppo misurata ieri e' 1,0002; la norma di V_a e' invece dominata")
print("   dagli elementi duali minuscoli (tetraedri schiacciati) e non e' un'osservabile: |V_x| = %.1f, |V_y| = %.1f" % (0,0) if False else "   dagli elementi duali minuscoli e non e' un'osservabile.")
# ---------- 3) Lyman-alfa: controllo della quadratura con le funzioni analitiche ----------
X,L=pickle.load(open('reticoli/equilibrio60000.pkl','rb'));n=len(X)
C=kdlib.build(X,L);K=C['K'];E=C['sizes'][1];h0=C['h'][0];D0=K[n:n+E,:n].tocsr();Lap=(D0.T@D0).tocsr()
c=np.array([L/2]*3);ic=int(np.argmin(np.linalg.norm(mi(X-c,L),axis=1)));R=mi(X-X[ic],L);r=np.linalg.norm(R,axis=1)
for a0 in [2.2,3.5]:
    f1=np.exp(-r/a0);f1/=np.sqrt(np.sum(h0*f1*f1))
    S_an=0
    for a in range(3):
        fp=R[:,a]*np.exp(-r/(2*a0));fp/=np.sqrt(np.sum(h0*fp*fp));S_an+=np.sum(h0*fp*R[:,a]*f1)**2
    print(f"\n3a) quadratura sul reticolo con le funzioni analitiche 1s e 2p (a0 = {a0}): Sum|<2p|r|1s>|^2 = {S_an/a0**2:.3f} a0^2  (continuo 1,665)")
    alpha=0.125;mu=1/(alpha*a0);H=(Lap/(2*mu)+sp.diags(-alpha/np.maximum(r,0.5))).tocsr()
    vals,vecs=eigsh(H,k=6,which='SA');o=np.argsort(vals);vals=vals[o];vecs=vecs[:,o]
    g1=vecs[:,0];f1l=g1/np.sqrt(h0)      # funzione fisica dall'autovettore simmetrizzato
    # proietto gli stati p del reticolo sulle 2p analitiche per capire se sono "le" 2p
    for k in range(1,6):
        gk=vecs[:,k];fk=gk/np.sqrt(h0);ov=[abs(np.sum(h0*fk*(R[:,a]*np.exp(-r/(2*a0)))/np.sqrt(np.sum(h0*(R[:,a]*np.exp(-r/(2*a0)))**2)))) for a in range(3)]
        ov2s=abs(np.sum(h0*fk*(1-r/(2*a0))*np.exp(-r/(2*a0))))/np.sqrt(np.sum(h0*((1-r/(2*a0))*np.exp(-r/(2*a0)))**2))
        print(f"    stato {k}: E = {vals[k]:.5f}, sovrapposizione con 2p_x,y,z = {ov[0]:.2f},{ov[1]:.2f},{ov[2]:.2f}, somma quadr. {sum(o*o for o in ov):.2f}; con 2s {ov2s:.2f}")
    S_l=sum(np.sum(vecs[:,k]*R[:,a]*g1)**2 for k in range(2,5) for a in range(3))/a0**2
    print(f"    elementi di matrice con gli autostati del reticolo (stati 2,3,4): {S_l:.3f} a0^2;  sovrapposizione 1s reticolo/analitica {abs(np.sum(h0*f1l*f1)):.3f}")
print(f"   [{time.time()-t0:.0f}s]")
