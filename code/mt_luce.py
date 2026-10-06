#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt_luce.py  —  Max Theory: la luce attraversa il vuoto di celle senza sfocarsi?

La luce e' rappresentata come campo elettromagnetico sul reticolo di celle:
  - le celle sono i vertici di una triangolazione di Delaunay (senza bordo);
    due celle sono collegate quando le loro regioni di Voronoi condividono una faccia;
  - il campo elettrico vive sui collegamenti fra celle, il campo magnetico sui
    triangoli formati da tre celle vicine;
  - le equazioni di Maxwell usano solo chi confina con chi e due rapporti fra misure duali:
        area della faccia di Voronoi / lunghezza del collegamento
        lunghezza del lato di Voronoi / area del triangolo
    (calcolo esterno discreto). Questa costruzione non ha vibrazioni spurie che
    trasportano energia: i soli modi spuri sono campi statici.
  - l'evoluzione e' a passi alternati (campo magnetico, poi elettrico): conserva
    l'energia esattamente.

Si fanno viaggiare onde piane polarizzate e si misura quanta coerenza perdono.
Si separano due cose:
  scarto iniziale  lo stato di partenza non coincide esattamente con un'onda del reticolo:
                   la piccola differenza si stacca subito e poi resta costante (non e' diffusione)
  diffusione       la perdita che continua a crescere nel tempo:
  Gamma(k) ~ (k * passo)^p
  p ~ 4  : diffusione da disordine locale, trascurabile per la luce reale
  p ~ 2  : diffusione forte, le galassie lontane apparirebbero sfocate
E si estrapola a celle della lunghezza di Planck.

Uso (i reticoli si generano con le stesse regole di mt_trasparenza.py, che deve
stare nella stessa cartella):
  python mt_luce.py --tipo crescita --n 131072
  python mt_luce.py --carica cresc131k.npz              (reticolo salvato da mt_trasparenza.py)
  python mt_luce.py --confronto --n 32768               (crescita, equilibrio, esclusione, cristallo)

Requisiti: numpy, scipy (matplotlib facoltativo). Memoria: la triangolazione occupa
circa 2-3 GB per milione di celle.
"""

import argparse, json, os, time, itertools
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse.linalg import cg
import mt_trasparenza as mt

log = mt.log
mi = mt.mi


def ret_cristallo_ccc(n, rng):
    """cubico a corpo centrato alla stessa densita'. A differenza del cubico a facce centrate,
    la sua triangolazione e' unica (nessun gruppo di celle sulla stessa sfera): controllo pulito."""
    m = max(2, int(round((n / 2) ** (1 / 3))))
    ac = (2 / mt.RHO) ** (1 / 3)
    base = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    g = np.array(np.meshgrid(range(m), range(m), range(m), indexing='ij')).reshape(3, -1).T
    X = ((g[:, None, :] + base[None, :, :]).reshape(-1, 3)) * ac
    return X, m * ac


GENERATORI = dict(mt.GENERATORI)
GENERATORI['cristallo'] = ret_cristallo_ccc


# ---------------------------------------------------------------------------
# complesso di Delaunay e Voronoi senza bordo
# ---------------------------------------------------------------------------

def circocentri(Q):
    """Q: (m,4,3) tetraedri -> centri delle sfere circoscritte (vertici di Voronoi)"""
    a = Q[:, 0]
    A = 2 * (Q[:, 1:] - a[:, None, :])
    b = (Q[:, 1:] ** 2).sum(2) - (a ** 2).sum(1)[:, None]
    return np.linalg.solve(A, b[..., None])[..., 0]


def complesso(X, L, passo):
    n = len(X)
    m = 3.5 * passo
    ext, orig = [X], [np.arange(n)]
    for sh in itertools.product((-1, 0, 1), repeat=3):
        if sh == (0, 0, 0):
            continue
        mask = np.ones(n, bool)
        for ax in range(3):
            if sh[ax] == 1:
                mask &= X[:, ax] < m
            elif sh[ax] == -1:
                mask &= X[:, ax] > L - m
        idx = np.where(mask)[0]
        ext.append(X[idx] + np.array(sh) * L)
        orig.append(idx)
    P = np.vstack(ext)
    O = np.concatenate(orig)
    log(f"  triangolazione: {n} celle piu' {len(P) - n} immagini")
    D = Delaunay(P)
    S = D.simplices
    Q = P[S]
    cen = Q.mean(1)
    tieni = np.all((cen >= 0) & (cen < L), axis=1)          # un rappresentante per ogni tetraedro
    S, Q = S[tieni], Q[tieni]
    cc = circocentri(Q)
    So = O[S]
    log(f"  {len(S)} tetraedri ({len(S) / n:.2f} per cella)")
    # --- collegamenti (spigoli) ---
    coppie = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    ea = So[:, coppie[:, 0]].ravel(); eb = So[:, coppie[:, 1]].ravel()
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    ekey = lo.astype(np.int64) * n + hi
    ue, einv = np.unique(ekey, return_inverse=True)
    E = len(ue)
    edges = np.stack([ue // n, ue % n], 1)
    dvec = mi(X[edges[:, 1]] - X[edges[:, 0]], L)
    elen = np.linalg.norm(dvec, axis=1)
    emid = X[edges[:, 0]] + dvec / 2
    # faccia di Voronoi duale a ogni collegamento: poligono dei circocentri attorno allo spigolo
    cc_e = np.repeat(cc, 6, axis=0)
    rel = mi(cc_e - emid[einv], L)
    t = dvec[einv] / elen[einv][:, None]
    e1 = np.cross(t, np.where(np.abs(t[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(t, e1)
    x = (rel * e1).sum(1); y = (rel * e2).sum(1); ang = np.arctan2(y, x)
    ordine = np.lexsort((ang, einv))
    x, y, gi = x[ordine], y[ordine], einv[ordine]
    inizio = np.r_[0, np.where(np.diff(gi))[0] + 1]
    nxt = np.arange(len(gi)) + 1
    fine = np.r_[inizio[1:], len(gi)]
    nxt[fine - 1] = inizio
    cr = x * y[nxt] - x[nxt] * y
    Avor = 0.5 * np.abs(np.bincount(gi, cr, minlength=E))
    # --- triangoli ---
    terne = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    T3 = np.sort(So[:, terne].reshape(-1, 3), axis=1)
    tkey = (T3[:, 0].astype(np.int64) * n + T3[:, 1]) * n + T3[:, 2]
    ut, tinv, tcount = np.unique(tkey, return_inverse=True, return_counts=True)
    F = len(ut)
    tris = np.stack([ut // (n * n), (ut // n) % n, ut % n], 1)
    if np.any(tcount != 2):
        log(f"  attenzione: {np.sum(tcount != 2)} triangoli non condivisi da due tetraedri")
    # lato di Voronoi duale: segmento fra i circocentri dei due tetraedri che condividono il triangolo
    cc_t = np.repeat(cc, 4, axis=0)
    o2 = np.argsort(tinv, kind='stable')
    a1 = cc_t[o2[0::2]]; a2 = cc_t[o2[1::2]]
    lvor = np.linalg.norm(mi(a2 - a1, L), axis=1)
    lvor_full = np.zeros(F); lvor_full[tinv[o2[0::2]]] = lvor
    va = mi(X[tris[:, 1]] - X[tris[:, 0]], L); vb = mi(X[tris[:, 2]] - X[tris[:, 0]], L)
    nrm = np.cross(va, vb); tarea = 0.5 * np.linalg.norm(nrm, axis=1); tn = nrm / (2 * tarea[:, None])
    tcen = X[tris[:, 0]] + (va + vb) / 3
    # incidenza triangoli -> collegamenti (orientati dalla cella di indice minore a quella maggiore)
    def eidx(a, b):
        return np.searchsorted(ue, a.astype(np.int64) * n + b)
    rows = np.repeat(np.arange(F), 3)
    cols = np.stack([eidx(tris[:, 0], tris[:, 1]), eidx(tris[:, 1], tris[:, 2]), eidx(tris[:, 0], tris[:, 2])], 1).ravel()
    vals = np.tile([1.0, 1.0, -1.0], F)
    d1 = sp.csr_matrix((vals, (rows, cols)), shape=(F, E))
    # incidenza collegamenti -> celle (per la divergenza)
    d0 = sp.csr_matrix((np.r_[-np.ones(E), np.ones(E)], (np.r_[np.arange(E), np.arange(E)], np.r_[edges[:, 0], edges[:, 1]])), shape=(E, n))
    h1 = Avor / elen
    h2 = lvor_full / tarea
    log(f"  {E} collegamenti ({2 * E / n:.1f} per cella), {F} triangoli; volume dalle facce "
        f"{(Avor * elen).sum() / 3:.1f} su {L ** 3:.1f}; lati di Voronoi nulli {np.mean(lvor_full < 1e-12) * 100:.2f}%")
    return dict(edges=edges, dvec=dvec, elen=elen, emid=emid, Avor=Avor, tris=tris, tn=tn, tarea=tarea,
                tcen=tcen, lvor=lvor_full, d1=d1, d0=d0, h1=h1, h2=h2)


# ---------------------------------------------------------------------------
# onde elettromagnetiche
# ---------------------------------------------------------------------------

def sinc(z):
    return np.sinc(z / np.pi)


def modo(C, X, L, nvec, pol=0):
    """onda piana che viaggia lungo k, polarizzata perpendicolarmente: valori complessi del
    campo elettrico sui collegamenti e del campo magnetico sui triangoli"""
    k = 2 * np.pi * np.asarray(nvec, float) / L
    kk = np.linalg.norm(k); kh = k / kk
    a = np.array([1.0, 0, 0]) if abs(kh[0]) < 0.9 else np.array([0, 1.0, 0])
    e = np.cross(kh, a); e /= np.linalg.norm(e)
    if pol:
        e = np.cross(kh, e)
    b = np.cross(kh, e)
    m_e = (C['dvec'] @ e) * np.exp(1j * (C['emid'] @ k)) * sinc(C['dvec'] @ k / 2)
    m_f = (C['tn'] @ b) * C['tarea'] * np.exp(1j * (C['tcen'] @ k))
    return m_e, m_f, kk


def senza_divergenza(C, Ev):
    """toglie dal campo elettrico la parte con divergenza (cariche fittizie)"""
    d0, h1 = C['d0'], C['h1']
    Lap = (d0.T @ sp.diags(h1) @ d0).tocsr()
    rhs = d0.T @ (h1 * Ev)
    phi, _ = cg(Lap, rhs - rhs.mean(), rtol=1e-10, maxiter=5000)
    return Ev - d0 @ phi


def passo_temporale(C):
    d1, h1, h2 = C['d1'], C['h1'], C['h2']
    v = np.random.default_rng(0).normal(size=d1.shape[1])
    for _ in range(60):
        w = (d1.T @ (h2 * (d1 @ v))) / h1
        lam = np.linalg.norm(w) / np.linalg.norm(v)
        v = w / np.linalg.norm(w)
    return 1.6 / np.sqrt(lam)


def destinazione(C, X, L, nvec, Ev, B, pol=0):
    """dove si trova, alla fine, l'energia dell'onda: stessa onda, stessa direzione con l'altra
    polarizzazione, altre direzioni con la stessa lunghezza d'onda, altrove"""
    h1, h2 = C['h1'], C['h2']
    tot = (h1 * Ev ** 2).sum() + (h2 * B ** 2).sum()
    def quota(nv, pp):
        m_e, m_f, _ = modo(C, X, L, nv, pp)
        nn = (h1 * np.abs(m_e) ** 2).sum() + (h2 * np.abs(m_f) ** 2).sum()
        a = ((h1 * Ev * np.conj(m_e)).sum() + (h2 * B * np.conj(m_f)).sum())
        return 2 * abs(a) ** 2 / nn / tot
    nv0 = np.asarray(nvec)
    stessa = quota(tuple(nv0), pol)
    altra = quota(tuple(nv0), 1 - pol)
    r2 = int((nv0 ** 2).sum())
    R = int(np.ceil(np.sqrt(r2)))
    altre = 0.0
    for v in itertools.product(range(-R, R + 1), repeat=3):
        v = np.array(v)
        if (v ** 2).sum() != r2 or np.all(v == nv0) or np.all(v == -nv0):
            continue
        if tuple(v) < tuple(-v):          # +v e -v contati insieme in quota()
            continue
        altre += quota(tuple(v), 0) + quota(tuple(v), 1)
    return dict(stessa=float(stessa), altra_polarizzazione=float(altra), altre_direzioni=float(altre),
                altrove=float(max(0.0, 1 - stessa - altra - altre)))


def propaga(C, X, L, nvec, tempo, punti=41, pol=0):
    d1, h1, h2 = C['d1'], C['h1'], C['h2']
    m_e, m_f, kk = modo(C, X, L, nvec, pol)
    Ev = senza_divergenza(C, m_e.real)
    dt = C['dt']
    # campo magnetico al mezzo passo precedente (onda che viaggia in avanti, frequenza ~ k)
    B = (m_f * np.exp(-1j * kk * (-dt / 2))).real
    # proiezione sul modo con il prodotto scalare dell'energia
    norma = np.sqrt((h1 * np.abs(m_e) ** 2).sum() + (h2 * np.abs(m_f) ** 2).sum())
    def ampiezza(Ev, Bm):
        return ((h1 * Ev * np.conj(m_e)).sum() + (h2 * Bm * np.conj(m_f)).sum()) / norma
    passi = int(np.ceil(tempo / dt))
    # campionamento abbastanza fitto che la fase avanzi meno di 0,8 radianti fra due misure
    # (altrimenti non si puo' sapere quanti giri ha fatto e la velocita' risulta sbagliata)
    ogni = max(1, min(passi // (punti - 1), int(0.8 / (kk * dt))))
    ts, A = [], []
    Bprev = B.copy()
    for s in range(passi + 1):
        if s % ogni == 0:
            ts.append(s * dt); A.append(ampiezza(Ev, 0.5 * (Bprev + B)))
        Bprev = B
        B = B - dt * (d1 @ Ev)
        Ev = Ev + dt * ((d1.T @ (h2 * B)) / h1)
    ts = np.array(ts); A = np.array(A)
    C['_ultimo'] = (Ev, 0.5 * (Bprev + B))
    return ts, A / A[0], kk


def misura(X, L, passo, kmax_passo=1.25, tempo=None, punti=41, soglia_facce=1e-3, max_onde=14):
    C = complesso(X, L, passo)
    if soglia_facce:
        # facce di Voronoi quasi nulle (quattro celle quasi sulla stessa sfera): area minima pari a
        # soglia_facce volte la mediana. Rende il passo temporale ~30 volte piu' grande e non cambia
        # i risultati (verificato: coerenza e velocita' identiche alla quarta cifra).
        mod = np.mean(C['h1'] < soglia_facce * np.median(C['h1'])) * 100
        C['h1'] = np.maximum(C['h1'], soglia_facce * np.median(C['h1']))
        log(f"  facce quasi nulle portate alla soglia: {mod:.2f}% dei collegamenti")
    C['dt'] = passo_temporale(C)
    log(f"  passo temporale stabile {C['dt']:.4f}")
    if tempo is None:
        tempo = 4.0 * L
    direzioni = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
    dk = 2 * np.pi / L
    lista = []
    for s in range(1, 200):
        for base in direzioni:
            kk = dk * s * np.linalg.norm(base)
            if kk * passo <= kmax_passo:
                lista.append((kk, tuple(int(s * x) for x in base)))
        if dk * s * passo > kmax_passo:
            break
    lista.sort()
    if len(lista) > max_onde:
        idx = np.unique(np.round(np.geomspace(1, len(lista), max_onde)).astype(int) - 1)
        lista = [lista[i] for i in idx]
    out = []
    for kk, nv in lista:
        ts, A, kk = propaga(C, X, L, nv, tempo, punti)
        mag = np.abs(A)
        c = -np.polyfit(ts, np.unwrap(np.angle(A)), 1)[0] / kk
        # diffusione: pendenza di ln|A| dopo lo scarto iniziale (ultimi tre quarti del tempo)
        sel = (ts >= tempo / 4) & (mag > 0.02)
        y = np.log(mag[sel]); tt = ts[sel]
        pend, inter = np.polyfit(tt, y, 1)
        resid = y - (pend * tt + inter)
        # risoluzione: la pendenza piu' piccola distinguibile dalle oscillazioni residue
        risol = 2 * max(np.std(resid), 1e-7) / (tt[-1] - tt[0]) * np.sqrt(12 / max(len(tt), 3))
        gam = -pend
        trasp = bool(gam < 2 * risol)
        iniziale = float(1 - np.exp(inter + pend * tempo / 4))
        dest = destinazione(C, X, L, nv, *C['_ultimo'])
        out.append(dict(k=kk, k_passo=kk * passo, direzione=list(nv), c=float(c),
                        gamma=float(np.nan if trasp else gam), gamma_misurata=float(gam), risoluzione=float(risol),
                        trasparente=trasp, scarto_iniziale=iniziale, coerenza_finale=float(mag[-1]), destinazione=dest,
                        t=ts.tolist(), coerenza=mag.tolist()))
        log(f"  luce k*passo={kk * passo:.3f} {nv}: velocita' {c:.4f}, scarto iniziale {iniziale:.1e}, "
            + (f"diffusione non rilevata (sotto {2 * risol:.1e})" if trasp else f"diffusione {gam:.2e}")
            + f", coerenza finale {mag[-1]:.4f}")
        log(f"      energia alla fine: stessa onda {dest['stessa']:.4f}, altra polarizzazione {dest['altra_polarizzazione']:.1e}, "
            f"altre direzioni {dest['altre_direzioni']:.1e}, altrove {dest['altrove']:.1e}")
    return out


def esegui(tipo, n, rng, args, X=None, L=None):
    log(f"=== reticolo: {tipo} ===")
    if X is None:
        X, L = GENERATORI[tipo](n, rng)
    n = len(X)
    passo = float(np.mean(cKDTree(X, boxsize=L).query(X, k=2)[0][:, 1]))
    log(f"  {n} celle, lato {L:.2f}, passo {passo:.3f}")
    if args.salva_reticolo:
        nome = args.salva_reticolo if not args.confronto else f"{tipo}_{args.salva_reticolo}"
        np.savez_compressed(nome, X=X, L=L, tipo=tipo)
    onde = misura(X, L, passo, kmax_passo=args.kmax, tempo=args.tempo, soglia_facce=args.soglia_facce, max_onde=args.max_onde)
    return dict(tipo=tipo, n=n, L=float(L), passo=passo, onde=onde, analisi=mt.analizza(onde, passo))


def main():
    ap = argparse.ArgumentParser(description="Max Theory: la luce come campo elettromagnetico sul reticolo di celle")
    ap.add_argument('--tipo', default='crescita', choices=list(GENERATORI))
    ap.add_argument('--confronto', action='store_true')
    ap.add_argument('--n', type=int, default=32768)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--kmax', type=float, default=1.25, help="k*passo massimo (1,25: onde lunghe 5 celle)")
    ap.add_argument('--max-onde', type=int, default=14, help="numero massimo di onde misurate")
    ap.add_argument('--soglia-facce', type=float, default=1e-3, help="area minima delle facce, in frazione della mediana (0 = nessuna)")
    ap.add_argument('--tempo', type=float, default=None)
    ap.add_argument('--salva-reticolo', default=None)
    ap.add_argument('--carica', default=None)
    ap.add_argument('--cartella', default='risultati_luce')
    args = ap.parse_args()
    os.makedirs(args.cartella, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ris = []
    if args.carica:
        z = np.load(args.carica)
        ris.append(esegui(str(z['tipo']), len(z['X']), rng, args, X=z['X'], L=float(z['L'])))
    elif args.confronto:
        for t in ['crescita', 'equilibrio', 'esclusione', 'cristallo']:
            ris.append(esegui(t, args.n, rng, args))
    else:
        ris.append(esegui(args.tipo, args.n, rng, args))
    print("\n" + "=" * 78 + "\nRIEPILOGO — luce come campo elettromagnetico\n" + "=" * 78)
    for r in ris:
        a = r['analisi']
        print(f"\n{r['tipo']}: {r['n']} celle, passo {r['passo']:.3f}")
        on = sorted(r['onde'], key=lambda o: o['k_passo'])
        tr = [o for o in on if o['trasparente']]
        if tr:
            kt = max(o['k_passo'] for o in tr)
            print(f"  diffusione non rilevata in {len(tr)} onde su {len(on)} (fino a k*passo = {kt:.2f}, onde lunghe {2 * np.pi / kt:.1f} passi)")
            gb = 2 * [o for o in tr if o['k_passo'] == kt][0]['risoluzione']
            print(f"    limite superiore della diffusione per l'onda piu' corta senza diffusione: {gb:.1e}")
            print("    stima prudente: sotto la risoluzione la diffusione cresca appena come k^2, partendo da quel limite")
            for nome, cm, fr in mt.estrapola({'potenza_p': 2.0, 'coeff': gb / kt ** 2}, r['passo']):
                print(f"      {nome:22s} cammino libero almeno {cm:9.2e} m = {fr:9.2e} raggi dell'universo  -> {'nitida' if fr > 1 else 'SFOCATA'}")
        if on and 'destinazione' in on[-1]:
            o = on[-1]['destinazione']
            print(f"  onda piu' corta (k*passo {on[-1]['k_passo']:.2f}), energia alla fine: altra polarizzazione {o['altra_polarizzazione']:.1e}, "
                  f"altre direzioni {o['altre_direzioni']:.1e}, altrove {o['altrove']:.1e}")
        mis = [o for o in on if not o['trasparente']]
        if mis:
            print("  diffusione misurata: " + "  ".join(f"k*passo {o['k_passo']:.2f}: {o['gamma']:.1e}" for o in mis))
        if 'potenza_p' in a:
            print(f"  diffusione misurata: Gamma ~ (k*passo)^p con p = {a['potenza_p']:.2f}"
                  "  (ricavata dalle onde piu' corte, vicino alla scala delle celle)")
            for nome, cm, fr in mt.estrapola(a, r['passo']):
                print(f"    {nome:22s} cammino libero {cm:9.2e} m = {fr:9.2e} raggi dell'universo  -> {'nitida' if fr > 1 else 'SFOCATA'}")
        if 'beta_quadratica' in a:
            print(f"  velocita': c(k) = c0 [1 - beta (k*passo)^2], c0 = {a['c0']:.4f}, beta = {a['beta_quadratica']:.3f}")
    print("\n  Lo scarto iniziale (riportato onda per onda nel registro) non e' diffusione: si stacca subito e resta costante.")
    print("  p ~ 4 o piu': diffusione da disordine locale (trascurabile per la luce reale)")
    print("  p ~ 2: diffusione forte (le galassie lontane apparirebbero sfocate)")
    with open(os.path.join(args.cartella, 'risultati.json'), 'w') as f:
        json.dump(ris, f, indent=1, default=float)
    log(f"risultati in {os.path.join(args.cartella, 'risultati.json')}")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for r in ris:
            on = sorted(r['onde'], key=lambda o: o['k_passo'])
            ok = [o for o in on if np.isfinite(o['gamma']) and o['gamma'] > 0]
            if ok:
                ax[0].loglog([o['k_passo'] for o in ok], [o['gamma'] for o in ok], 'o-', ms=4, label=r['tipo'])
            ax[1].plot([o['k_passo'] for o in on], [o['c'] for o in on], 'o-', ms=4, label=r['tipo'])
        xs = np.array([0.05, 1.0])
        for pp, st in [(2, ':'), (4, '--')]:
            ax[0].loglog(xs, 1e-3 * xs ** pp, 'k' + st, lw=0.8, label=f'pendenza {pp}')
        ax[0].set_xlabel('k * passo'); ax[0].set_ylabel('perdita di coerenza'); ax[0].set_title('diffusione della luce'); ax[0].legend(fontsize=8)
        ax[1].set_xlabel('k * passo'); ax[1].set_ylabel('velocita'); ax[1].set_title('velocita della luce'); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.cartella, 'mt_luce.png'), dpi=130)
        log("grafici salvati")
    except Exception as ex:
        log(f"niente grafici ({ex})")


if __name__ == '__main__':
    main()
