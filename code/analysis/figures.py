"""Regenerates figures 1-3 of docs/figures/ (run from code/analysis/, after prepara_reticoli.py).
Figure 4 is drawn from docs/figures/fig4_hyperuniformity_data.json."""
import os, re, json, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import Voronoi
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'..','..','docs','figures'); RES=os.path.join(HERE,'..','..','results')
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
INK='#1a1f26';ACC='#8a3b12';OK='#1f6f5a';SOFT='#5c6670';WARN='#8a6a12'

# figura 1: sezione piana del reticolo in equilibrio + schema
X,L,passo,_=pickle.load(open('reticoli/equilibrio20000.pkl','rb'))
R=0.44*passo; z0=L/2; dz=X[:,2]-z0; sec=np.abs(dz)<R; P=X[sec][:,:2]; rr=np.sqrt(R**2-dz[sec]**2)
fig=plt.figure(figsize=(10.5,4.8)); ax=fig.add_axes([0.03,0.04,0.44,0.86]); ax.set_aspect('equal'); ax.axis('off')
vor=Voronoi(P)
for rv in vor.ridge_vertices:
    if -1 in rv: continue
    v=vor.vertices[rv]; ax.plot(v[:,0],v[:,1],color=SOFT,lw=0.5,alpha=0.6,zorder=1)
for p,r in zip(P,rr): ax.add_patch(Circle(p,r,fc='#e9b98a',ec=ACC,lw=0.7,zorder=2))
w=8.0; ax.set_xlim(L/2-w,L/2+w); ax.set_ylim(L/2-w,L/2+w)
ax.set_title('(a) A plane section of the equilibrium lattice\n(cells of radius 0.44 $\\ell_P$ cut by the plane; Voronoi partition of the section)',fontsize=10,color=INK)
ax2=fig.add_axes([0.52,0.04,0.46,0.86]); ax2.set_aspect('equal'); ax2.axis('off')
d=1.0; C=np.array([[0,0],[d,0],[d/2,d*np.sqrt(3)/2]]); r=0.44*d
for c in C: ax2.add_patch(Circle(c,r,fc='#e9b98a',ec=ACC,lw=1.2,zorder=2))
cc=C.mean(0)
for i,j in [(0,1),(1,2),(2,0)]:
    m=(C[i]+C[j])/2; dv=m-cc; dv/=np.linalg.norm(dv); ax2.plot(*np.array([cc,m+0.9*dv]).T,color=SOFT,lw=0.8,ls='--',zorder=1)
m=(C[0]+C[1])/2; ax2.plot([m[0],m[0]],[m[1]-0.33,m[1]+0.33],color=OK,lw=4,solid_capstyle='round',zorder=3)
ax2.annotate('aperture\nthe face between two cells\n(electric field: phase difference across it)',xy=(m[0],m[1]-0.33),xytext=(m[0]+0.05,m[1]-0.9),fontsize=8.5,color=OK,ha='center',arrowprops=dict(arrowstyle='-',color=OK,lw=0.8))
ax2.scatter(*cc,s=70,color=ACC,zorder=4)
ax2.annotate('channel\nthe line where three cells meet\n(perpendicular to the page; magnetic field: torsion around it)',xy=cc,xytext=(cc[0]+1.05,cc[1]+0.75),fontsize=8.5,color=ACC,ha='left',arrowprops=dict(arrowstyle='-',color=ACC,lw=0.8))
ax2.text(cc[0]-1.6,cc[1]+1.15,'void\nthe point among four cells\n(the fourth lies above the page)',fontsize=8.5,color=INK,ha='left')
ax2.set_xlim(-1.7,2.9); ax2.set_ylim(-1.2,2.1); ax2.set_title('(b) The three places of the interstice',fontsize=10,color=INK)
fig.savefig(os.path.join(OUT,'fig1_lattice.png'),dpi=200,bbox_inches='tight'); plt.close(fig)

# figura 2: livelli dell'idrogeno
def parse(f):
    s=open(f,encoding='utf-8',errors='replace').read(); ryd=float(re.search(r'Rydberg di MT = ([\d.]+)',s).group(1)); st={}
    for m in re.finditer(r'^\s*\d+\s+(-?[\d.]+)\s+[\d.]+\s+[\d.]+\s+(\d[spd])\s+\(([\d.]+)\)',s,re.M):
        if float(m.group(3))>=0.5: st.setdefault(m.group(2),[]).append(float(m.group(1)))
    return ryd,st
r1,s1=parse(os.path.join(RES,'grande.txt')); r3,s3=parse(os.path.join(RES,'grande25.txt')); lev={}
for lab in ['1s','2s','2p']: lev[lab]=np.mean(s1[lab])/r1
for lab in ['3s','3p','3d']: lev[lab]=np.mean(s3[lab])/r3
fig,ax=plt.subplots(figsize=(6.2,4.6)); xs={'s':0,'p':1,'d':2}
for n in (1,2,3):
    ax.hlines(-1/n**2,-0.45,2.45,color=SOFT,lw=0.8,ls='--',zorder=1); ax.text(2.55,-1/n**2,f'n = {n}  (−R/{n}²)',va='center',fontsize=9,color=SOFT)
for lab,E in lev.items():
    x=xs[lab[1]]; n=int(lab[0]); dev=100*(abs(E)*n**2-1)
    ax.hlines(E,x-0.32,x+0.32,color=ACC,lw=3,zorder=3); ax.text(x,E+(0.012 if n>1 else 0.03),f'{lab}: {dev:+.2f} %',ha='center',va='bottom',fontsize=8.5,color=INK)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['s','p','d']); ax.set_xlim(-0.6,3.7); ax.set_ylabel('energy / Rydberg of the model'); ax.set_ylim(-1.08,0.02)
ax.set_title('Hydrogen on the lattice: levels against Rydberg (deviation of the binding energy)\nn = 1, 2: 2.7·10⁶ cells, a₀ = 4, box 32 a₀ · n = 3: a₀ = 2.5, box 50 a₀',fontsize=9.5,color=INK)
fig.savefig(os.path.join(OUT,'fig2_hydrogen_levels.png'),dpi=200,bbox_inches='tight'); fig.savefig(os.path.join(OUT,'fig2_hydrogen_levels.svg'),bbox_inches='tight'); plt.close(fig)

# figura 3: convergenza di Lyman-alfa (valori da results/)
runs=[('idrogeno',3,28.1,0.4309,(8,8)),('balmer',2,42.1,0.4405,(-120,-16)),('lyman4',4,21.1,0.4421,(8,4)),('grande',4,31.6,0.4253,(8,-2)),('grande3',3,42.1,0.4293,(8,-11)),('grande25',2.5,50.6,0.4337,(8,4))]
a0=np.array([r[1] for r in runs]);box=np.array([r[2] for r in runs]);f=np.array([r[3] for r in runs])
A=np.c_[np.ones(6),1/a0**2,1/box**2]; coef=np.linalg.lstsq(A,f,rcond=None)[0]; rs=np.sqrt(np.mean((A@coef-f)**2)); fcorr=f-coef[2]/box**2
fig,ax=plt.subplots(figsize=(6.6,4.6)); xx=np.linspace(0,0.27,50)
ax.plot(xx,coef[0]+coef[1]*xx,color=SOFT,lw=1,ls='--',label=f'fit  f = f∞ + {coef[1]:.2f}/a₀² + c/box²   (f∞ = {coef[0]:.3f}, rms {rs:.3f})')
for (name,a,b,fv,off),fc in zip(runs,fcorr):
    ax.scatter(1/a**2,fv,s=30+1.2*b,color=ACC,zorder=3); ax.scatter(1/a**2,fc,s=18,facecolor='white',edgecolor=ACC,zorder=4)
    ax.annotate(f'{name}: a₀ = {a:g}, box {b:.0f} a₀',(1/a**2,fv),xytext=off,textcoords='offset points',fontsize=7.5,color=INK)
ax.axhline(0.4162,color=OK,lw=1.4); ax.text(0.265,0.4162+0.0012,'observed 0.4162',color=OK,fontsize=9,ha='right')
ax.axhspan(0.410,0.414,color=OK,alpha=0.12); ax.text(0.004,0.4108,'extrapolated f∞: 0.410–0.414 (range over box laws)',color=OK,fontsize=8.5)
ax.set_xlabel('1 / a₀²   (a₀ in cells: finer lattice to the left)'); ax.set_ylabel('Lyman-α oscillator strength  f'); ax.set_xlim(0,0.27); ax.set_ylim(0.405,0.449); ax.legend(loc='upper left',fontsize=8,frameon=False)
ax.set_title('Lyman-α: convergence with resolution and box size\nfilled: measured · open: box term removed · marker size ∝ box',fontsize=10,color=INK)
fig.savefig(os.path.join(OUT,'fig3_lyman_convergence.png'),dpi=200,bbox_inches='tight'); fig.savefig(os.path.join(OUT,'fig3_lyman_convergence.svg'),bbox_inches='tight'); plt.close(fig)

# figura 4: S(k) dai dati salvati
D=json.load(open(os.path.join(OUT,'fig4_hyperuniformity_data.json'))); kc=np.array(D['k_sigma_bin_centres'])
style={'random (Poisson)':(SOFT,'o'),'random with a minimum distance':(WARN,'s'),'random, then relaxed to equilibrium':(ACC,'^'),'grown by births and pushes (MT)':(OK,'D')}
fig,ax=plt.subplots(figsize=(6.2,4.4))
for lab,v in D['S'].items(): col,mk=style[lab]; ax.plot(kc,v,marker=mk,color=col,lw=1.4,ms=5,label=lab)
ax.set_yscale('log'); ax.set_xlabel('k σ   (σ = cell size)'); ax.set_ylabel('structure factor  S(k)'); ax.set_ylim(1e-3,2)
ax.set_title('Hyperuniformity: density fluctuations of four lattices\nS(k) → 0 at small k only for the lattices in equilibrium',fontsize=10,color=INK)
ax.legend(fontsize=8,frameon=False,loc='center right',bbox_to_anchor=(1.0,0.42))
fig.savefig(os.path.join(OUT,'fig4_hyperuniformity.png'),dpi=200,bbox_inches='tight'); fig.savefig(os.path.join(OUT,'fig4_hyperuniformity.svg'),bbox_inches='tight'); plt.close(fig)
print('figures written to', OUT)
