import os
import numpy as np, sys, time
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'));import kdlib
from scipy.sparse.linalg import expm_multiply
t0=time.time();mi=kdlib.mt.mi
z=np.load('reticoli/crescita4096.npz');X=z['X'];L=float(z['L'])
C=kdlib.build(X,L);K=C['K'].tocsr();n,E,F,T=C['sizes'];h0=C['h'][0];N=n+E+F+T
c=np.array([L/2]*3);R=mi(X-c,L)
print("1) Delta x . Delta k dallo spettro: pacchetti gaussiani di larghezza s sulle celle; Delta k_x = sqrt(<K^2>/3) (isotropia, omega=|k|)")
print("   s (celle)   Delta x_x (celle)   Delta k_x (rad/cella)   prodotto   (continuo: 0,500)")
for s in [0.6,0.8,1.0,1.5,2.0,3.0,4.0]:
    f=np.exp(-np.sum(R**2,1)/(4*s*s));g=np.zeros(N);g[:n]=np.sqrt(h0)*f;g/=np.linalg.norm(g)     # |g_i|^2 = probabilita' della cella
    p=g[:n]**2;dx=np.sqrt(np.sum(p*R[:,0]**2)-np.sum(p*R[:,0])**2)
    Kg=K@g;k2=Kg@Kg;dk=np.sqrt(k2/3)
    print(f"   {s:4.1f}        {dx:6.3f}              {dk:6.3f}             {dx*dk:6.3f}")
print(f"   [{time.time()-t0:.0f}s]")
print("\n2) Delta omega . Delta N dall'evoluzione: sopravvivenza |<psi(0)|psi(N)>|^2 sotto U = e^{-iK} per batch")
print("   s (celle)   Delta omega (rad/batch)   N_1/2 (batch)   prodotto   (Mandelstam-Tamm: >= pi/4 = 0,785)")
for s in [1.0,2.0,3.0]:
    f=np.exp(-np.sum(R**2,1)/(4*s*s));g=np.zeros(N,complex);g[:n]=np.sqrt(h0)*f;g/=np.linalg.norm(g)
    Kg=K@g;w1=np.real(np.vdot(g,Kg));w2=np.real(np.vdot(Kg,Kg));dw=np.sqrt(w2-w1*w1)
    ts=np.linspace(0,2.5/dw,26);surv=[]
    psi=g.copy();dt=ts[1]
    for k,t in enumerate(ts):
        if k:psi=expm_multiply(-1j*K*dt,psi)
        surv.append(abs(np.vdot(g,psi))**2)
    surv=np.array(surv);i=np.argmax(surv<0.5);Nh=np.interp(0.5,surv[i-1:i+1][::-1],ts[i-1:i+1][::-1]) if i>0 else np.nan
    print(f"   {s:4.1f}        {dw:6.3f}                  {Nh:6.2f}         {dw*Nh:6.3f}")
print(f"   [{time.time()-t0:.0f}s]")
