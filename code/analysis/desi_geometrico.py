import os
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
c=299792.458
# DESI DR2 BAO (arXiv 2503.14738, Tab. IV): z, DM/rd, sDM, DH/rd, sDH, corr ; BGS: DV/rd
bao=[(0.510,13.588,0.167,21.863,0.425,-0.459),(0.706,17.351,0.177,19.455,0.330,-0.404),(0.934,21.576,0.152,17.641,0.193,-0.40),
     (1.321,27.601,0.318,14.176,0.221,-0.434),(1.484,30.512,0.760,12.817,0.516,-0.500),(2.330,38.988,0.531,8.632,0.101,-0.431)]
bgs=(0.295,7.942,0.075)
wb=0.02237;Neff=3.044;Tnu=1.676e-4   # eV, temperatura dei neutrini oggi
wg=2.473e-5
def rho_nu(a,masses,h):
    # densita' dei neutrini (in unita' di rho_crit h^2), interpolazione relativistico -> non relativistico
    tot=0.0
    for m in masses:
        rel=(7/8)*(4/11)**(4/3)*wg/a**4
        tot+=rel*np.sqrt(1+(m*a/(3.15*Tnu))**2)
    return tot
def E2(z,p,beta,masses):
    H0,wc=p[0],p[1];h=H0/100;a=1/(1+z)
    wr=wg*(1+0.2271*0.0)  # fotoni; i neutrini sono trattati a parte
    wnu0=rho_nu(1.0,masses,h)
    Om=(wc+wb)/h**2;Or=wr/h**2;Onu0=wnu0/h**2
    Ode=1-Om-Or-Onu0
    f=np.exp(-beta*(1-a))                 # energia oscura di MT: p0(a) = p0_0 exp(-beta (1-a))
    return Or/a**4+Om/a**3+rho_nu(a,masses,h)/h**2+Ode*f
def H(z,p,beta,masses):return p[0]*np.sqrt(E2(z,p,beta,masses))
def DM(z,p,beta,masses):return c*quad(lambda zz:1/H(zz,p,beta,masses),0,z,limit=200)[0]
def rd(p,masses):
    h=p[0]/100;wc=p[1];wnu=sum(masses)/93.14
    return 55.154*np.exp(-72.3*(wnu+0.0006)**2)/((wc+wb)**0.25351*wb**0.12807)
def chi2(p,beta,masses):
    r=rd(p,masses);s=0.0
    for z,dm,sdm,dh,sdh,rho in bao:
        tm=DM(z,p,beta,masses)/r;th=c/H(z,p,beta,masses)/r
        d=np.array([tm-dm,th-dh]);C=np.array([[sdm**2,rho*sdm*sdh],[rho*sdm*sdh,sdh**2]]);s+=d@np.linalg.solve(C,d)
    z,dv,sdv=bgs;s+=((z*(DM(z,p,beta,masses))**2*c/H(z,p,beta,masses))**(1/3)/r-dv)**2/sdv**2
    # fondo cosmico: scala acustica e densita' di materia fredda (Planck)
    zs=1089.9;rs=r*144.43/147.09;theta=100*rs/DM(zs,p,beta,masses)
    s+=((theta-1.04110)/0.00031)**2+((p[1]-0.1200)/0.0012)**2
    return s
def masses_from_sum(S):   # ordinamento normale: m1, m2=sqrt(m1^2+7.5e-5), m3=sqrt(m1^2+2.5e-3)
    from scipy.optimize import brentq
    if S<0.0588:S=0.0588
    m1=brentq(lambda m:m+np.sqrt(m*m+7.5e-5)+np.sqrt(m*m+2.5e-3)-S,0,S)
    return [m1,np.sqrt(m1*m1+7.5e-5),np.sqrt(m1*m1+2.5e-3)]
def fit(S,beta=None):
    ms=masses_from_sum(S) if S>0 else [0,0,0]
    if beta is None:
        f=lambda q:chi2(q[:2],q[2],ms);r=minimize(f,[68,0.12,0.0],method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1e-4,maxiter=800));return r.fun,r.x
    f=lambda q:chi2(q,beta,ms);r=minimize(f,[68,0.12],method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1e-4,maxiter=600));return r.fun,r.x
print("Somma masse  | modello standard (Lambda)        | energia oscura di MT (p0 variabile)")
print("   [eV]      |  chi2    H0    Omega_m          |  chi2    H0    Omega_m   beta    w_eff(z=0.5)")
rows=[]
for S in [0.0,0.06,0.09,0.12,0.15]:
    cl,pl=fit(S,0.0);cm,pm=fit(S)
    hl=pl[0]/100;hm=pm[0]/100
    # w effettivo dell'energia oscura di MT a z=0.5: w = -1 - (1/3) dln f/dln a = -1 - beta a/3
    a=1/1.5;w=-1-pm[2]*a/3
    rows.append((S,cl,cm))
    print(f"   {S:4.2f}      | {cl:6.2f}  {pl[0]:5.1f}  {(pl[1]+wb)/hl**2:.3f}          | {cm:6.2f}  {pm[0]:5.1f}  {(pm[1]+wb)/hm**2:.3f}   {pm[2]:+.2f}    {w:+.3f}")
