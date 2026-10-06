"""
mt_idrogeno_pc.py — L'idrogeno sul reticolo di Max Theory (versione per PC)

Che cosa fa
  1. genera un reticolo casuale tridimensionale di celle e lo porta all'equilibrio (spinte chiuse)
  2. costruisce celle e varchi (Delaunay–Voronoi) con le rigidita' geometriche: faccia/lunghezza per ogni varco
  3. scrive l'equazione di Schrodinger di MT:  H = (1/2mu) * laplaciano_dei_varchi  -  alpha/r
  4. trova i livelli piu' bassi (1s, 2s, 2p, 3s, 3p, 3d) e confronta con Rydberg  E_n = -mu alpha^2 / 2n^2
  5. misura la degenerazione 2p, lo scarto 2s-2p, e la forza d'oscillatore di Lyman-alfa (osservato 0,4162)

Uso
  python mt_idrogeno_pc.py --n 100000 --a0 3.0        (prova: 5-10 minuti, ~2 GB)
  python mt_idrogeno_pc.py --n 500000 --a0 3.0        (corsa piena: 30-90 minuti, ~6-8 GB)
  python mt_idrogeno_pc.py --reticolo prova_reticolo.npz --tile 2 --a0 3.0 --out idrogeno
        (riusa il reticolo gia' in equilibrio affiancandolo 2x2x2: 8 volte le celle, scatola doppia, nessun assestamento)
  python mt_idrogeno_pc.py --reticolo prova_reticolo.npz --tile 3 --a0 4.0 --salva_varchi varchi3.npz --out grande
  python mt_idrogeno_pc.py --reticolo grande_reticolo.npz --varchi varchi3.npz --a0 3.0 --out grande3
        (corsa grande: 2,7 milioni di celle; la prima costruisce e salva i varchi, le successive li riusano)
  opzioni: --alpha 0.125  --phi 0.70  --passi 3000  --out risultati  --seed 1

Serve: python 3.9+, numpy, scipy  (pip install numpy scipy). Mettere mt_trasparenza.py nella stessa cartella.
I file prodotti: <out>_reticolo.npz (posizioni), <out>_stati.npz (livelli e nuvole), <out>.txt (riepilogo).
"""
import argparse, time, itertools, sys, os
import numpy as np, scipy.sparse as sp
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse.linalg import lobpcg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_trasparenza as mt

T0 = time.time()
def log(s): print(f"[{time.time()-T0:7.0f}s] {s}", flush=True)
mi = mt.mi

# ---------------------------------------------------------------- 2. celle e varchi (settore 0-1 del complesso)
def celle_e_varchi(X, L):
    n = len(X); passo = float(np.mean(cKDTree(X, boxsize=L).query(X, k=2)[0][:, 1]))
    m = 3.5 * passo; ext, orig = [X], [np.arange(n)]
    for sh in itertools.product((-1, 0, 1), repeat=3):
        if sh == (0, 0, 0): continue
        mask = np.ones(n, bool)
        for ax in range(3):
            if sh[ax] == 1: mask &= X[:, ax] < m
            elif sh[ax] == -1: mask &= X[:, ax] > L - m
        idx = np.where(mask)[0]; ext.append(X[idx] + np.array(sh) * L); orig.append(idx)
    P = np.vstack(ext); O = np.concatenate(orig)
    log(f"Delaunay su {len(P)} punti (con immagini)")
    S0 = Delaunay(P).simplices
    cen = P[S0].mean(1); keep = np.all((cen >= 0) & (cen < L), axis=1); del cen
    S0 = S0[keep]; S = np.sort(O[S0], axis=1); Q = P[S0]; del S0, P, O, keep
    log(f"{len(S)} tetraedri")
    # circocentri dei tetraedri = vertici di Voronoi
    a = Q[:, 0]; Am = 2 * (Q[:, 1:] - a[:, None, :]); bb = (Q[:, 1:] ** 2).sum(2) - (a ** 2).sum(1)[:, None]
    cc = np.linalg.solve(Am, bb[..., None])[..., 0]; del Am, bb, a, Q
    pa = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    ek = np.concatenate([S[:, i].astype(np.int64) * n + S[:, j] for i, j in pa]); ue, einv = np.unique(ek, return_inverse=True); E = len(ue)
    edges = np.stack([ue // n, ue % n], 1); dv = mi(X[edges[:, 1]] - X[edges[:, 0]], L); el = np.linalg.norm(dv, axis=1); em = X[edges[:, 0]] + dv / 2
    # area della faccia di Voronoi di ogni varco: poligono dei circocentri attorno al collegamento
    cce = np.tile(cc, (6, 1)); rel = mi(cce - em[einv], L); t = dv[einv] / el[einv][:, None]
    e1 = np.cross(t, np.where(np.abs(t[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])); e1 /= np.linalg.norm(e1, axis=1)[:, None]; e2 = np.cross(t, e1)
    x = (rel * e1).sum(1); y = (rel * e2).sum(1); ang = np.arctan2(y, x); o = np.lexsort((ang, einv)); x, y, gi = x[o], y[o], einv[o]
    ini = np.r_[0, np.where(np.diff(gi))[0] + 1]; fin = np.r_[ini[1:], len(gi)]; nx = np.arange(len(gi)) + 1; nx[fin - 1] = ini
    Avor = 0.5 * np.abs(np.bincount(gi, x * y[nx] - x[nx] * y, minlength=E))
    del cce, rel, t, e1, e2, x, y, ang, o, gi, cc, S, ek, einv
    V0 = np.zeros(n); np.add.at(V0, edges[:, 0], Avor * el / 6); np.add.at(V0, edges[:, 1], Avor * el / 6)   # volume di Voronoi della cella
    w = np.maximum(Avor / el, 1e-3 * np.median(Avor / el))                                               # rigidita' del varco
    d0 = sp.csr_matrix((np.r_[-np.ones(E), np.ones(E)], (np.r_[np.arange(E), np.arange(E)], np.r_[edges[:, 0], edges[:, 1]])), shape=(E, n))
    D0 = sp.diags(np.sqrt(w)) @ d0 @ sp.diags(1 / np.sqrt(V0))
    Lap = (D0.T @ D0).tocsr()                                                                             # laplaciano simmetrizzato
    log(f"{n} celle, {E} varchi, passo {passo:.3f}; vicine per cella {2*E/n:.1f}")
    return Lap, V0, edges, w, passo

# ---------------------------------------------------------------- funzioni analitiche (per le ipotesi di partenza e i controlli)
def analitiche(R, r, a0):
    f = {}
    f['1s'] = np.exp(-r / a0)
    f['2s'] = (1 - r / (2 * a0)) * np.exp(-r / (2 * a0))
    for k, ax in enumerate('xyz'): f['2p' + ax] = R[:, k] * np.exp(-r / (2 * a0))
    f['3s'] = (1 - 2 * r / (3 * a0) + 2 * r ** 2 / (27 * a0 ** 2)) * np.exp(-r / (3 * a0))
    for k, ax in enumerate('xyz'): f['3p' + ax] = R[:, k] * (1 - r / (6 * a0)) * np.exp(-r / (3 * a0))
    e3 = np.exp(-r / (3 * a0))
    f['3dxy'] = R[:, 0] * R[:, 1] * e3; f['3dyz'] = R[:, 1] * R[:, 2] * e3; f['3dxz'] = R[:, 0] * R[:, 2] * e3
    f['3dx2y2'] = (R[:, 0] ** 2 - R[:, 1] ** 2) * e3; f['3dz2'] = (3 * R[:, 2] ** 2 - r ** 2) * e3
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100000); ap.add_argument('--a0', type=float, default=3.0)
    ap.add_argument('--alpha', type=float, default=0.125); ap.add_argument('--phi', type=float, default=0.70)
    ap.add_argument('--passi', type=int, default=3000); ap.add_argument('--out', default='idrogeno'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--reticolo', default=None, help='file _reticolo.npz gia\' calcolato, per saltare la generazione')
    ap.add_argument('--tile', type=int, default=1, help='affianca il reticolo caricato tile x tile x tile volte (resta in equilibrio)')
    ap.add_argument('--parete', type=float, default=0.0, help='parete morbida: potenziale W*((r-0.8 L/2)/(0.2 L/2))^2 oltre 0.8 L/2, in unita\' di Rydberg (0 = periodico puro)')
    ap.add_argument('--rumore', type=float, default=0.0, help='rumore relativo gaussiano sui pesi dei varchi (es. 0.10)')
    ap.add_argument('--nucleo', type=float, default=0.5, help='raggio di ammorbidimento del nucleo, in passi (default 0.5)')
    ap.add_argument('--salva_varchi', default=None, help='salva celle e varchi (laplaciano) in questo file .npz, per riusarli con altri raggi')
    ap.add_argument('--varchi', default=None, help='carica celle e varchi da un file .npz gia\' salvato (salta Delaunay)')
    A = ap.parse_args()
    mt.PHI = A.phi; mt.RHO = A.phi / (np.pi / 6)
    # ---- 1. reticolo in equilibrio
    if A.reticolo:
        z = np.load(A.reticolo); X, L = z['X'], float(z['L']); log(f"reticolo caricato: {len(X)} celle, lato {L:.1f}")
        if A.tile > 1:
            X = np.vstack([X + np.array(sh) * L for sh in itertools.product(range(A.tile), repeat=3)]); L = L * A.tile
            log(f"affiancato {A.tile}x{A.tile}x{A.tile}: {len(X)} celle, lato {L:.1f} (equilibrio conservato)")
            np.savez(A.out + '_reticolo.npz', X=X, L=L)
    else:
        rng = np.random.default_rng(A.seed); X, L = mt.ret_casuale(A.n, rng); log(f"reticolo casuale: {len(X)} celle, lato {L:.1f}; assestamento ({A.passi} passi)...")
        X, fn = mt.rilassa(X, L, passi=A.passi); log(f"equilibrio raggiunto: forza residua {fn:.1e}")
        np.savez(A.out + '_reticolo.npz', X=X, L=L)
    n = len(X)
    # ---- 2. varchi e laplaciano
    if A.varchi:
        z = np.load(A.varchi); Lap = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=(n, n)); V0, edges, w, passo = z['V0'], z['edges'], z['w'], float(z['passo'])
        log(f"varchi caricati da {A.varchi}: {len(edges)} varchi")
    else:
        Lap, V0, edges, w, passo = celle_e_varchi(X, L)
        if A.salva_varchi:
            np.savez(A.salva_varchi, data=Lap.data, indices=Lap.indices, indptr=Lap.indptr, V0=V0, edges=edges, w=w, passo=passo); log(f"varchi salvati in {A.salva_varchi}")
    if A.rumore > 0:
        rng2 = np.random.default_rng(A.seed + 100); w = w * np.clip(1 + A.rumore * rng2.normal(size=len(w)), 0.1, None)
        d0 = sp.csr_matrix((np.r_[-np.ones(len(edges)), np.ones(len(edges))], (np.r_[np.arange(len(edges)), np.arange(len(edges))], np.r_[edges[:, 0], edges[:, 1]])), shape=(len(edges), n))
        D0 = sp.diags(np.sqrt(w)) @ d0 @ sp.diags(1 / np.sqrt(V0)); Lap = (D0.T @ D0).tocsr(); log(f"pesi dei varchi perturbati del {100*A.rumore:.0f}%")
    # ---- 3. l'equazione di MT
    alpha, a0 = A.alpha, A.a0; mu = 1 / (alpha * a0)
    c = np.array([L / 2] * 3); ic = int(np.argmin(np.linalg.norm(mi(X - c, L), axis=1)))
    R = mi(X - X[ic], L); r = np.linalg.norm(R, axis=1)
    U = -alpha / np.maximum(r, A.nucleo * passo)                  # nucleo ammorbidito
    Ry = mu * alpha ** 2 / 2
    if A.parete > 0:
        Rw = 0.8 * L / 2; U = U + A.parete * Ry * np.clip((r - Rw) / (0.2 * L / 2), 0, None) ** 2; log(f"parete morbida da {Rw/a0:.1f} a0, altezza {A.parete} Ry al bordo")
    H = (Lap / (2 * mu) + sp.diags(U)).tocsr()
    log(f"alpha = {alpha}, mu = {mu:.4f}, a0 = {a0} celle, scatola = {L/a0:.1f} a0, Rydberg = {Ry:.6f}")
    # ---- 4. livelli: LOBPCG con ipotesi di partenza analitiche e precondizionatore diagonale
    F = analitiche(R, r, a0); nomi = list(F.keys())
    X0 = np.stack([np.sqrt(V0) * F[k] for k in nomi], 1); X0 /= np.linalg.norm(X0, axis=0)
    Minv = sp.diags(1 / (H.diagonal() - H.diagonal().min() + 0.05 * Ry))
    log(f"diagonalizzazione (LOBPCG, {len(nomi)} stati)...")
    vals, vecs = lobpcg(H, X0, M=Minv, largest=False, tol=1e-7, maxiter=600)
    o = np.argsort(vals); vals, vecs = vals[o], vecs[:, o]
    log("fatto")
    # ---- 5. analisi
    def aniso(v):
        p = v * v / np.sum(v * v); Q = (p[:, None, None] * (R[:, :, None] * R[:, None, :])).sum(0); lam = np.linalg.eigvalsh(Q); return (lam.max() - lam.min()) / lam.mean()
    def rmed(v):
        p = v * v / np.sum(v * v); return np.sum(p * r) / a0
    righe = []
    righe.append(f"reticolo {n} celle, lato {L:.1f} = {L/a0:.1f} a0; alpha = {alpha}, mu = {mu:.4f}, a0 = {a0} celle; Rydberg di MT = {Ry:.6f}")
    righe.append(f"vicine per cella {2*len(edges)/n:.2f}; rigidita' dei varchi: mediana {np.median(w):.3f}")
    # identificazione per sovrapposizione con le funzioni analitiche (insiemi: 1s, 2s, 2p, 3s, 3p, 3d)
    insiemi = {'1s': ['1s'], '2s': ['2s'], '2p': ['2px', '2py', '2pz'], '3s': ['3s'], '3p': ['3px', '3py', '3pz'], '3d': ['3dxy', '3dyz', '3dxz', '3dx2y2', '3dz2']}
    base = {k: (np.sqrt(V0) * F[k]) / np.linalg.norm(np.sqrt(V0) * F[k]) for k in nomi}
    righe.append("\n k   E reticolo      <r>/a0   anisotropia   identificato (sovrapposizione^2)")
    tipo = []
    for k in range(len(vals)):
        ov = {nm: sum(np.dot(vecs[:, k], base[c]) ** 2 for c in comp) for nm, comp in insiemi.items()}
        best = max(ov, key=ov.get); tipo.append(best)
        righe.append(f" {k:2d}  {vals[k]:12.7f}   {rmed(vecs[:, k]):6.2f}    {aniso(vecs[:, k]):6.2f}       {best} ({ov[best]:.2f})")
    idx = lambda nm: [k for k in range(len(vals)) if tipo[k] == nm]
    s_idx = idx('1s'); e1 = vals[s_idx[0]] if s_idx else np.nan
    e2s = vals[idx('2s')[0]] if idx('2s') else np.nan
    p2 = sorted(idx('2p'), key=lambda k: vals[k])[:3]; e2p = np.mean(vals[p2]) if p2 else np.nan; sp2 = np.std(vals[p2]) if p2 else np.nan
    righe.append(f"\n1s: {e1:.7f}  contro Rydberg {-Ry:.7f}: scarto {100*(e1/(-Ry)-1):+.2f}%   <r>/a0 = {rmed(vecs[:, s_idx[0]]):.3f} (continuo 1,500)")
    righe.append(f"2p (tre stati): {e2p:.7f} ± {sp2:.1e}  contro {-Ry/4:.7f}: scarto {100*(e2p/(-Ry/4)-1):+.2f}%;  degenerazione fra i tre: {sp2/abs(e2p):.1e}")
    righe.append(f"2s: {e2s:.7f}  contro {-Ry/4:.7f}: scarto {100*(e2s/(-Ry/4)-1):+.2f}%;  separazione 2s-2p: {100*(e2s/e2p-1):+.2f}% (continuo 0)")
    righe.append(f"rapporto E(1s)/E(2p) = {e1/e2p:.4f} (Rydberg 4)")
    # Lyman-alfa: Sum_m,a |<2p_m|x_a|1s>|^2 in a0^2 (continuo 1,6653), Delta E in hartree (0,375), f = (2/3) Delta E Sum (0,4162)
    g1 = vecs[:, s_idx[0]]
    S = sum(np.sum(vecs[:, k] * R[:, a] * g1) ** 2 for k in p2 for a in range(3)) / a0 ** 2
    dE = (e2p - e1) / (mu * alpha ** 2); f_osc = 2 / 3 * dE * S
    # controllo di quadratura con le funzioni analitiche
    f1 = F['1s'] / np.sqrt(np.sum(V0 * F['1s'] ** 2)); S_an = 0
    for ax in 'xyz':
        fp = F['2p' + ax] / np.sqrt(np.sum(V0 * F['2p' + ax] ** 2)); S_an += sum(np.sum(V0 * fp * R[:, a] * f1) ** 2 for a in range(3))
    righe.append(f"\nLyman-alfa: Sum|<2p|r|1s>|^2 = {S:.4f} a0^2 (continuo 1,6653; quadratura sul reticolo con funzioni analitiche {S_an/a0**2:.4f})")
    righe.append(f"            Delta E = {dE:.5f} hartree (continuo 0,37500);  forza d'oscillatore f = {f_osc:.4f}  (osservato 0,4162; scarto {100*(f_osc/0.4162-1):+.1f}%)")
    # altre righe: f(i->f) = (2/3) Delta E (1/g_i) Sum_{m_i,m_f,a} |<f|x_a|i>|^2
    def forza(ini, fin, f_noto, nome):
        I = idx(ini); Fs = idx(fin)
        if not I or not Fs: righe.append(f"{nome}: stati non trovati"); return
        I = sorted(I, key=lambda k: vals[k])[:(1 if ini.endswith('s') else (3 if ini.endswith('p') else 5))]
        Fs = sorted(Fs, key=lambda k: vals[k])[:(1 if fin.endswith('s') else (3 if fin.endswith('p') else 5))]
        Ssum = sum(np.sum(vecs[:, kf] * R[:, a] * vecs[:, ki]) ** 2 for ki in I for kf in Fs for a in range(3)) / a0 ** 2
        dEl = (np.mean(vals[Fs]) - np.mean(vals[I])) / (mu * alpha ** 2); fl = 2 / 3 * dEl * Ssum / len(I)
        righe.append(f"{nome}: f = {fl:.4f}  (noto {f_noto}; scarto {100*(fl/f_noto-1):+.1f}%)   Delta E = {dEl:.4f} hartree, {len(I)}->{len(Fs)} stati")
    righe.append("")
    forza('1s', '3p', 0.0791, 'Lyman-beta 1s->3p')
    forza('2s', '3p', 0.4349, 'Balmer 2s->3p ')
    forza('2p', '3s', 0.01359, 'Balmer 2p->3s ')
    forza('2p', '3d', 0.6958, 'Balmer 2p->3d ')
    # somme sulla varieta' n=3 (robuste alle mescolanze fra stati degeneri): dal 1s solo il 3p (0,0791), dal 2s solo il 3p (0,4349), dal 2p 3s+3d (0,7094)
    n3 = [k for k in range(len(vals)) if tipo[k] in ('3s', '3p', '3d')]
    if len(n3) >= 9:
        righe.append(f"somme sulla varieta' n=3 ({len(n3)} stati, E media {np.mean(vals[n3]):.7f} contro {-Ry/9:.7f}, scarto {100*(np.mean(vals[n3])/(-Ry/9)-1):+.2f}%):")
        for ini, noto, nome in [('1s', 0.0791, '1s -> n=3'), ('2s', 0.4349, '2s -> n=3'), ('2p', 0.7094, '2p -> n=3')]:
            I = sorted(idx(ini), key=lambda k: vals[k])[:(1 if ini.endswith('s') else 3)]
            if not I: continue
            Ssum = sum(np.sum(vecs[:, kf] * R[:, a] * vecs[:, ki]) ** 2 for ki in I for kf in n3 for a in range(3)) / a0 ** 2
            dEl = (np.mean(vals[n3]) - np.mean(vals[I])) / (mu * alpha ** 2); fl = 2 / 3 * dEl * Ssum / len(I)
            righe.append(f"   {nome}: f = {fl:.4f}  (noto {noto}; scarto {100*(fl/noto-1):+.1f}%)")
    testo = "\n".join(righe); print("\n" + testo)
    open(A.out + '.txt', 'w', encoding='utf-8').write(testo + "\n")
    np.savez(A.out + '_stati.npz', vals=vals, vecs=vecs.astype(np.float32), ic=ic, a0=a0, alpha=alpha, mu=mu, L=L)
    log(f"risultati in {A.out}.txt, stati in {A.out}_stati.npz")

if __name__ == '__main__':
    main()
