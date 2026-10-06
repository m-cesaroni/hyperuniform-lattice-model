#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt_trasparenza.py  —  Max Theory: il vuoto di celle e' trasparente alla luce?

Domanda: una vibrazione che attraversa un reticolo di celle disposte senza ordine
cristallino viene diffusa dalle irregolarita'? Se si', con quale legge al variare
della lunghezza d'onda? Da quella legge dipende se la luce delle galassie lontane
arriverebbe nitida o sfocata.

Cosa fa lo script
  1. Genera uno o piu' reticoli 3D senza bordo, alla stessa densita':
       casuale     punti lanciati a caso (riferimento peggiore)
       esclusione  punti a caso che non possono avvicinarsi troppo
       equilibrio  punti a caso, poi lasciati spingersi finche' le spinte si annullano
       crescita    regola di MT: ogni cella fa nascere una figlia accanto, lo spazio
                   si dilata, le celle si spingono fino all'equilibrio; si ripete
       ricotto     come equilibrio, ma con un'agitazione che si spegne lentamente
       cristallo   reticolo cubico a facce centrate (riferimento trasparente)
  2. Misura il disordine del reticolo:
       S(k)        fluttuazioni di densita' a scala 2*pi/k (iperuniforme: S -> 0)
       varianza    del numero di celle in sfere di raggio crescente
       anisotropia locale delle direzioni delle vicine
  3. Fa passare onde piane (vibrazione senza massa con spin, come la luce) e misura.
     Due modi di pesare il passaggio fra celle vicine (--pesi):
       voronoi     lo spazio e' diviso fra le celle (ogni punto appartiene alla cella piu'
                   vicina); la vibrazione passa fra due celle in proporzione all'area della
                   faccia che condividono. Il bilanciamento e' esatto e locale (teorema della
                   divergenza: le facce di una regione chiusa si sommano a zero).
       bilanciati  pesi il piu' vicini possibile a 1 con il bilanciamento imposto da
                   un'unica equazione globale (versione precedente).
       entrambi    li misura tutti e due sullo stesso reticolo.
       c(k)        velocita' in funzione del numero d'onda
       Gamma(k)    quanto in fretta l'onda perde coerenza (diffusione)
       p           potenza con cui Gamma cresce con k, per k piccolo
  4. Estrapola a celle della lunghezza di Planck: cammino libero medio della luce
     visibile, dei raggi X e dei raggi gamma, confrontato con il raggio dell'universo.

Uso
  python mt_trasparenza.py --tipo crescita --n 200000
  python mt_trasparenza.py --confronto --n 100000          (esclusione, equilibrio, crescita, cristallo)
  python mt_trasparenza.py --tipo crescita --n 200000 --salva-reticolo ret.npz
  python mt_trasparenza.py --carica ret.npz                (riusa un reticolo gia' generato)
  python mt_trasparenza.py --carica ret.npz --pesi entrambi (confronta i due modi di pesare)

Requisiti: python 3.9+, numpy, scipy; matplotlib per i grafici (facoltativo).

Tempi indicativi su un PC recente (un solo processo):
  n =  50 000   qualche minuto per tipo
  n = 200 000   circa 30-60 minuti per tipo
  n = 1 000 000 diverse ore per tipo, e 8-16 GB di memoria
Il risultato decisivo e' la potenza p: servono onde lunghe molte celle, quindi n grande.
"""

import argparse, json, os, sys, time, itertools
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree, Voronoi
from scipy.sparse.linalg import cg, expm_multiply

SIGMA = 1.0                                   # portata della spinta fra celle
PHI = 0.70                                    # frazione di riempimento di riferimento
RHO = PHI / (np.pi * SIGMA**3 / 6)            # celle per unita' di volume (uguale per tutti i tipi)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def lato(n):
    return (n / RHO) ** (1 / 3)


def mi(d, L):
    """differenza per la via piu' corta: il reticolo non ha bordo"""
    return d - L * np.round(d / L)


# ---------------------------------------------------------------------------
# 1. RETICOLI
# ---------------------------------------------------------------------------

def ret_casuale(n, rng):
    L = lato(n)
    return rng.random((n, 3)) * L, L


def ret_esclusione(n, rng, frazione=0.30):
    """celle lanciate a caso che non possono avvicinarsi piu' di dex"""
    L = lato(n)
    dex = (6 * frazione / (np.pi * RHO)) ** (1 / 3)
    pts = np.empty((0, 3))
    giri = 0
    while len(pts) < n and giri < 400:
        giri += 1
        C = rng.random((max(20000, n // 4), 3)) * L
        if len(pts):
            d, _ = cKDTree(pts, boxsize=L).query(C)
            C = C[d >= dex]
        if not len(C):
            continue
        T = cKDTree(C, boxsize=L)
        presi = np.zeros(len(C), bool)
        tieni = []
        for i in range(len(C)):
            if presi[i]:
                continue
            tieni.append(i)
            presi[T.query_ball_point(C[i], dex)] = True
        pts = np.vstack([pts, C[tieni]])[:n]
        if giri % 10 == 0:
            log(f"  esclusione: {len(pts)}/{n} celle")
    if len(pts) < n:
        log(f"  attenzione: esclusione satura a {len(pts)} celle")
    return pts, L


def rilassa(X, L, passi=4000, tol=1e-5, skin=0.3, rng=None, temperatura=0.0, raffreddamento=0.999):
    """Le celle si spingono quando sono piu' vicine di SIGMA (spinta che si spegne con la
    distanza). Minimizzazione FIRE con lista dei vicini aggiornata solo quando serve.
    Con temperatura > 0 aggiunge un'agitazione che si spegne a ogni passo (ricottura)."""
    X = X.copy()
    V = np.zeros_like(X)
    dt, a, Np = 0.05, 0.1, 0
    ref = X.copy()
    pairs = cKDTree(X, boxsize=L).query_pairs(SIGMA + skin, output_type='ndarray')
    temp = temperatura
    fn = np.inf
    for k in range(passi):
        if np.max(np.linalg.norm(mi(X - ref, L), axis=1)) > skin / 2:
            pairs = cKDTree(X, boxsize=L).query_pairs(SIGMA + skin, output_type='ndarray')
            ref = X.copy()
        d = mi(X[pairs[:, 1]] - X[pairs[:, 0]], L)
        r = np.linalg.norm(d, axis=1)
        m = r < SIGMA
        f = np.zeros(len(r))
        f[m] = (1 - r[m] / SIGMA) / SIGMA
        fv = (f / np.maximum(r, 1e-12))[:, None] * d
        F = np.zeros_like(X)
        np.add.at(F, pairs[:, 0], -fv)
        np.add.at(F, pairs[:, 1], fv)
        fn = np.sqrt((F * F).sum() / len(X))
        if temp > 1e-9 and rng is not None:
            F = F + np.sqrt(2 * temp / dt) * rng.normal(size=F.shape)
            temp *= raffreddamento
        elif fn < tol:
            break
        P = (F * V).sum()
        nF = np.linalg.norm(F)
        V = (1 - a) * V + a * F * np.linalg.norm(V) / max(nF, 1e-30)
        if P > 0:
            Np += 1
            if Np > 5:
                dt = min(dt * 1.1, 0.3)
                a *= 0.99
        else:
            Np = 0
            dt *= 0.5
            V[:] = 0
            a = 0.1
        V += dt * F
        X = (X + dt * V) % L
        if k % 500 == 0 and k:
            log(f"    rilassamento: passo {k}, forza residua {fn:.1e}")
    return X, fn


def ret_equilibrio(n, rng, passi=6000):
    X, L = ret_casuale(n, rng)
    X, fn = rilassa(X, L, passi=passi)
    log(f"  equilibrio: forza residua {fn:.1e}")
    return X, L


def ret_ricotto(n, rng, passi=8000):
    X, L = ret_casuale(n, rng)
    X, _ = rilassa(X, L, passi=passi // 2, rng=rng, temperatura=2e-3, raffreddamento=0.9985)
    X, fn = rilassa(X, L, passi=passi)
    log(f"  ricotto: forza residua {fn:.1e}")
    return X, L


def ret_crescita(n, rng, passi_gen=1500, passi_fin=6000):
    """regola di MT: nascite accanto a celle esistenti, dilatazione, spinte fino all'equilibrio"""
    N = 64
    Lg = lato(N)
    X = rng.random((N, 3)) * Lg
    X, _ = rilassa(X, Lg, passi=800)
    while N < n:
        s = 2 ** (1 / 3)
        X *= s
        Lg *= s
        u = rng.normal(size=X.shape)
        u /= np.linalg.norm(u, axis=1)[:, None]
        X = np.vstack([X, (X + 0.5 * SIGMA * u) % Lg])
        N = len(X)
        X, fn = rilassa(X, Lg, passi=passi_fin if N >= n else passi_gen)
        log(f"  crescita: {N} celle, forza residua {fn:.1e}")
    return X, Lg


def ret_cristallo(n, rng, tremolio=0.0):
    """cubico a facce centrate, alla stessa densita' (riferimento trasparente)"""
    m = max(2, int(round((n / 4) ** (1 / 3))))
    ac = (4 / RHO) ** (1 / 3)
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    g = np.array(np.meshgrid(range(m), range(m), range(m), indexing='ij')).reshape(3, -1).T
    X = ((g[:, None, :] + base[None, :, :]).reshape(-1, 3)) * ac
    L = m * ac
    if tremolio > 0:
        X = (X + tremolio * ac * rng.normal(size=X.shape)) % L
    return X, L


GENERATORI = {
    'casuale': ret_casuale,
    'esclusione': ret_esclusione,
    'equilibrio': ret_equilibrio,
    'crescita': ret_crescita,
    'ricotto': ret_ricotto,
    'cristallo': ret_cristallo,
}


# ---------------------------------------------------------------------------
# 2. DISORDINE DEL RETICOLO
# ---------------------------------------------------------------------------

def fattore_struttura(X, L, rng, kmax=3.0, per_guscio=48):
    """S(k) per k piccolo; si campionano al piu' per_guscio vettori per ogni guscio"""
    N = len(X)
    dk = 2 * np.pi / L
    nmax = int(kmax / dk) + 1
    risultati = []
    for s in range(1, nmax + 1):
        # vettori interi con |n| in [s-0.5, s+0.5)
        cand = rng.integers(-s - 1, s + 2, size=(per_guscio * 40, 3))
        nn = np.linalg.norm(cand, axis=1)
        cand = cand[(nn >= s - 0.5) & (nn < s + 0.5)]
        cand = np.unique(cand, axis=0)[:per_guscio]
        if not len(cand):
            continue
        ks = dk * cand
        S = []
        for kv in ks:
            ph = X @ kv
            S.append((np.cos(ph).sum() ** 2 + np.sin(ph).sum() ** 2) / N)
        risultati.append((dk * s, float(np.mean(S))))
    return risultati


def varianza_sfere(X, L, rng, raggi=(1, 1.5, 2, 3, 4, 6, 8), campioni=4000):
    T = cKDTree(X, boxsize=L)
    C = rng.random((campioni, 3)) * L
    out = []
    for R in raggi:
        if R > L / 3:
            break
        c = np.array([len(x) for x in T.query_ball_point(C, R)])
        out.append((R, float(c.var() / c.mean())))
    return out


def spettro_disordine(X, L, pr, u, r, coef, rng, kmax=2.0, per_guscio=32):
    """Disordine residuo della facilita' di passaggio, cella per cella.
    M_i = somma coef |d| n n^T, normalizzato alla media della traccia:
      parte isotropa  (traccia - 1): quanto la cella fa passare la vibrazione in generale
      parte anisotropa (senza traccia): quanto la fa passare meglio in certe direzioni
    Si misurano gli spettri a k piccolo e la correlazione nello spazio della parte
    anisotropa, da cui la lunghezza di correlazione xi. La teoria della diffusione in un
    mezzo disordinato: per k*xi << 0,5 la perdita cresce come k^4, per k*xi >~ 0,5 come k^2."""
    n = len(X)
    M = np.zeros((n, 3, 3))
    for i in range(3):
        for j in range(3):
            v = coef * r * u[:, i] * u[:, j]
            np.add.at(M[:, i, j], pr[:, 0], v)
            np.add.at(M[:, i, j], pr[:, 1], v)
    tr = np.trace(M, axis1=1, axis2=2)
    M /= tr.mean() / 3
    f = np.trace(M, axis1=1, axis2=2) / 3 - 1
    D = M - (np.trace(M, axis1=1, axis2=2) / 3)[:, None, None] * np.eye(3)
    comp = np.stack([D[:, 0, 0] - D[:, 1, 1], D[:, 2, 2], D[:, 0, 1], D[:, 0, 2], D[:, 1, 2]], 1)
    dk = 2 * np.pi / L
    out = []
    for s in range(1, int(kmax / dk) + 1):
        cand = rng.integers(-s - 1, s + 2, size=(per_guscio * 40, 3))
        nn = np.linalg.norm(cand, axis=1)
        cand = np.unique(cand[(nn >= s - 0.5) & (nn < s + 0.5)], axis=0)[:per_guscio]
        if not len(cand):
            continue
        vi, va = [], []
        for kv in dk * cand:
            ph = X @ kv
            c, sn = np.cos(ph), np.sin(ph)
            vi.append(((f * c).sum() ** 2 + (f * sn).sum() ** 2) / n)
            va.append((((comp * c[:, None]).sum(0) ** 2 + (comp * sn[:, None]).sum(0) ** 2).sum()) / n)
        out.append((dk * s, float(np.mean(vi)), float(np.mean(va))))
    # correlazione nello spazio della parte anisotropa
    T = cKDTree(X, boxsize=L)
    m = min(n, 4000)
    idx = rng.choice(n, m, replace=False)
    var = float((comp * comp).sum(1).mean())
    raggi = np.arange(0.5, min(8.0, L / 3), 0.5)
    corr = []
    for R0, R1 in zip(raggi[:-1], raggi[1:]):
        acc, cnt = 0.0, 0
        for i, nb in zip(idx, T.query_ball_point(X[idx], R1)):
            nb = np.asarray(nb)
            if not len(nb):
                continue
            d = np.linalg.norm(mi(X[nb] - X[i], L), axis=1)
            nb = nb[(d >= R0) & (d < R1)]
            if len(nb):
                acc += (comp[nb] @ comp[i]).sum()
                cnt += len(nb)
        corr.append(((R0 + R1) / 2, acc / max(cnt, 1) / var))
    xi = np.nan
    for (R, c) in corr:
        if c < np.exp(-1):
            xi = R
            break
    return dict(spettro=out, correlazione=corr, xi=float(xi),
                dev_isotropa=float(f.std()), dev_anisotropa=float(np.sqrt(var)))


def anisotropia_locale(X, L, cut):
    T = cKDTree(X, boxsize=L)
    pr = T.query_pairs(cut, output_type='ndarray')
    d = mi(X[pr[:, 1]] - X[pr[:, 0]], L)
    u = d / np.linalg.norm(d, axis=1)[:, None]
    M = np.zeros((len(X), 3, 3))
    for i in range(3):
        for j in range(3):
            np.add.at(M[:, i, j], pr[:, 0], u[:, i] * u[:, j])
            np.add.at(M[:, i, j], pr[:, 1], u[:, i] * u[:, j])
    ev = np.linalg.eigvalsh(M)
    return float(np.mean((ev[:, 2] - ev[:, 0]) / np.maximum(ev.mean(1), 1e-12)))


# ---------------------------------------------------------------------------
# 3. ONDE: vibrazione senza massa con spin (operatore di Weyl sul reticolo)
# ---------------------------------------------------------------------------

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.array([[1, 0], [0, -1]], complex)


def grafo(X, L, vicini=14):
    cut = (3 * vicini / (4 * np.pi * RHO)) ** (1 / 3)
    T = cKDTree(X, boxsize=L)
    pr = T.query_pairs(cut, output_type='ndarray')
    d = mi(X[pr[:, 1]] - X[pr[:, 0]], L)
    r = np.linalg.norm(d, axis=1)
    return pr, d / r[:, None], r, cut


def pesi_bilanciati(n, pr, u, r):
    """pesi dei collegamenti il piu' vicini possibile a 1, con il vincolo che in ogni cella
    le direzioni pesate delle vicine si annullino (nessun verso privilegiato)"""
    E = len(pr)
    a, b = pr[:, 0], pr[:, 1]
    v = u / r[:, None]
    rows = np.concatenate([3 * a + k for k in range(3)] + [3 * b + k for k in range(3)])
    cols = np.concatenate([np.arange(E)] * 6)
    vals = np.concatenate([v[:, k] for k in range(3)] + [-v[:, k] for k in range(3)])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(3 * n, E))
    squil0 = np.linalg.norm(A @ np.ones(E)) / np.sqrt(n)
    lam, _ = cg((A @ A.T).tocsr(), A @ np.ones(E), rtol=1e-12, maxiter=20000)
    w = np.ones(E) - A.T @ lam
    squil1 = np.linalg.norm(A @ w) / np.sqrt(n)
    return w, squil0, squil1


def operatore_weyl(n, pr, u, coef):
    """passaggio della vibrazione fra celle vicine con coefficiente coef per collegamento"""
    a, b = pr[:, 0], pr[:, 1]
    S = (u[:, 0, None, None] * SX + u[:, 1, None, None] * SY + u[:, 2, None, None] * SZ) * (-1j * coef)[:, None, None]
    rows, cols, vals = [], [], []
    for p in range(2):
        for q in range(2):
            rows += [2 * a + p, 2 * b + p]
            cols += [2 * b + q, 2 * a + q]
            vals += [S[:, p, q], -S[:, p, q]]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n))


def voronoi_facce(X, L, passo):
    """Tassellazione di Voronoi senza bordo. Restituisce le coppie di celle che condividono
    una faccia, la direzione fra i centri, la distanza, l'area della faccia e il volume
    di ogni cella (somma delle piramidi faccia-centro: V = sum A |d| / 6)."""
    n = len(X)
    m = 3.0 * passo
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
    log(f"  Voronoi: {n} celle piu' {len(P) - n} immagini oltre il bordo")
    vor = Voronoi(P)
    rp = vor.ridge_points
    rv = vor.ridge_vertices
    lun = np.array([len(v) for v in rv])
    buone = ((rp[:, 0] < n) | (rp[:, 1] < n)) & np.array([-1 not in v for v in rv])
    a0, b0 = O[rp[:, 0]], O[rp[:, 1]]
    buone &= a0 != b0
    area = np.zeros(len(rp))
    for k in np.unique(lun[buone]):
        sel = np.where(buone & (lun == k))[0]
        Vv = vor.vertices[np.array([rv[i] for i in sel])]           # (m, k, 3)
        c = Vv.mean(1, keepdims=True)
        nrm = P[rp[sel, 1]] - P[rp[sel, 0]]
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        e1 = np.cross(nrm, np.where(np.abs(nrm[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(nrm, e1)
        x = ((Vv - c) * e1[:, None, :]).sum(2)
        y = ((Vv - c) * e2[:, None, :]).sum(2)
        o = np.argsort(np.arctan2(y, x), axis=1)
        x = np.take_along_axis(x, o, 1)
        y = np.take_along_axis(y, o, 1)
        area[sel] = 0.5 * np.abs((x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1))
    # una sola faccia per coppia di celle
    lo, hi = np.minimum(a0, b0), np.maximum(a0, b0)
    chiave = lo.astype(np.int64) * n + hi
    sel = np.where(buone & (area > 1e-12))[0]
    _, prima = np.unique(chiave[sel], return_index=True)
    sel = sel[prima]
    pr = np.stack([lo[sel], hi[sel]], 1)
    A = area[sel]
    d = mi(X[pr[:, 1]] - X[pr[:, 0]], L)
    r = np.linalg.norm(d, axis=1)
    u = d / r[:, None]
    V = np.zeros(n)
    np.add.at(V, pr[:, 0], A * r / 6)
    np.add.at(V, pr[:, 1], A * r / 6)
    # controllo: le facce di ogni cella, orientate verso l'esterno, si sommano a zero
    chi = np.zeros((n, 3))
    np.add.at(chi, pr[:, 0], A[:, None] * u)
    np.add.at(chi, pr[:, 1], -A[:, None] * u)
    chiusura = float(np.linalg.norm(chi, axis=1).mean() / A.mean())
    log(f"  Voronoi: {len(pr)} facce, {2 * len(pr) / n:.1f} facce per cella; volume totale {V.sum():.1f} su {L**3:.1f}; "
        f"chiusura delle celle {chiusura:.1e}")
    return pr, u, r, A, V, chiusura


def onda_piana(X, L, nvec, ampiezza=None):
    k = 2 * np.pi * np.asarray(nvec, float) / L
    kk = np.linalg.norm(k)
    kh = k / kk
    th = np.arccos(np.clip(kh[2], -1, 1))
    ph = np.arctan2(kh[1], kh[0])
    chi = np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)])      # elica positiva
    env = np.exp(1j * (X @ k)) if ampiezza is None else ampiezza * np.exp(1j * (X @ k))
    psi = (env[:, None] * chi).ravel()
    return psi / np.linalg.norm(psi), kk


def coerenza(H, psi0, T, punti):
    """|<psi0|psi(t)>| e fase, per t da 0 a T"""
    ts = np.linspace(0, T, punti)
    A = [1.0 + 0j]
    psi = psi0.copy()
    for i in range(1, punti):
        psi = expm_multiply(-1j * H * (ts[i] - ts[i - 1]), psi)
        A.append(np.vdot(psi0, psi))
    return ts, np.array(A)


def misura_onde(X, L, passo, kmax_passo=1.0, tempo=None, punti=41, rng=None, pesi='voronoi'):
    n = len(X)
    amp = None
    if pesi == 'voronoi':
        pr, u, r, A, V, chiusura = voronoi_facce(X, L, passo)
        deg = 2 * len(pr) / n
        # passaggio attraverso la faccia comune, con le celle pesate per il loro volume
        coef = A / (2 * np.sqrt(V[pr[:, 0]] * V[pr[:, 1]]))
        amp = np.sqrt(V)
        s0 = chiusura
    else:
        pr, u, r, cut = grafo(X, L)
        deg = np.bincount(pr.ravel(), minlength=n).mean()
        log(f"  grafo: raggio {cut:.3f}, vicine per cella {deg:.1f}")
        w, s0, s1 = pesi_bilanciati(n, pr, u, r)
        log(f"  pesi bilanciati: squilibrio {s0:.2e} -> {s1:.1e}; pesi negativi {np.mean(w < 0) * 100:.2f}%")
        coef = w / r
    dis = spettro_disordine(X, L, pr, u, r, coef, rng if rng is not None else np.random.default_rng(0))
    log(f"  disordine del passaggio: isotropo {dis['dev_isotropa']:.3f}, anisotropo {dis['dev_anisotropa']:.3f}; "
        f"lunghezza di correlazione dell'anisotropia xi = {dis['xi']:.2f} (passo {passo:.2f})")
    log("    correlazione dell'anisotropia a distanza r: " + "  ".join(f"{R:.2f}:{c:+.3f}" for R, c in dis['correlazione'][:10]))
    log("    spettro dell'anisotropia a k piccolo: " + "  ".join(f"{k:.2f}:{va:.4f}" for k, vi, va in dis['spettro'][:8]))
    H = operatore_weyl(n, pr, u, coef)
    # normalizzazione: velocita' della luce = 1 per l'onda piu' lunga
    psi, kk = onda_piana(X, L, (1, 0, 0), amp)
    ts, A = coerenza(H, psi, 4.0, 9)
    c0 = -np.polyfit(ts, np.unwrap(np.angle(A)), 1)[0] / kk
    H = H / abs(c0)
    if tempo is None:
        tempo = 1.5 * L                      # l'onda attraversa il reticolo una volta e mezza
    # onde lungo assi, diagonali di faccia e di corpo, dalla piu' lunga possibile fino a kmax
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
    # al piu' 20 onde, distribuite in modo uniforme in log(k)
    if len(lista) > 20:
        idx = np.unique(np.round(np.geomspace(1, len(lista), 20)).astype(int) - 1)
        lista = [lista[i] for i in idx]
    out = []
    for kk, nv in lista:
        psi, kk = onda_piana(X, L, nv, amp)
        ts, A = coerenza(H, psi, tempo, punti)
        mag = np.abs(A)
        c = -np.polyfit(ts, np.unwrap(np.angle(A)), 1)[0] / kk
        # perdita di coerenza a lungo termine: pendenza di ln|A| nell'ultima parte
        sel = (ts >= tempo / 3) & (mag > 0.02)
        if sel.sum() < 4:
            sel = mag > 0.02
        gam = -np.polyfit(ts[sel], np.log(mag[sel]), 1)[0] if sel.sum() >= 3 else np.nan
        # sotto la soglia del rumore numerico l'onda non e' diffusa: nessuna perdita rilevata
        trasp = bool(mag.min() > 0.995)
        if trasp:
            gam = np.nan
        out.append(dict(k=kk, k_passo=kk * passo, direzione=list(nv), c=float(c),
                        gamma=float(gam), trasparente=trasp, coerenza_finale=float(mag[-1]),
                        t=ts.tolist(), coerenza=mag.tolist()))
        log(f"  onda k*passo={kk * passo:.3f} {nv}: velocita' {c:.4f}, "
            + ("nessuna perdita rilevata" if trasp else f"perdita {gam:.2e}") + f", coerenza finale {mag[-1]:.4f}")
    return dict(pesi=pesi, vicine=float(deg), squilibrio=float(s0), disordine=dis, onde=out, tempo=tempo)


# ---------------------------------------------------------------------------
# 4. ANALISI E ESTRAPOLAZIONE A CELLE DI PLANCK
# ---------------------------------------------------------------------------

L_PLANCK = 1.616e-35
E_PLANCK_EV = 1.22e28
RAGGIO_UNIVERSO = 1.4e26


def analizza(onde, passo):
    ok = [o for o in onde if np.isfinite(o['gamma']) and o['gamma'] > 0]
    res = {'onde_trasparenti': sum(1 for o in onde if o.get('trasparente')), 'onde_totali': len(onde)}
    if len(ok) >= 3:
        x = np.log([o['k_passo'] for o in ok])
        y = np.log([o['gamma'] for o in ok])
        # potenza dalle onde piu' lunghe (meta' inferiore), dove conta
        m = max(3, len(ok) // 2)
        p, lnA = np.polyfit(x[:m], y[:m], 1)
        res['potenza_p'] = float(p)
        res['coeff'] = float(np.exp(lnA))           # gamma = coeff * (k*passo)^p, in unita' c/passo... (c=1)
        p_tutte, _ = np.polyfit(x, y, 1)
        res['potenza_p_tutte'] = float(p_tutte)
    cv = [(o['k_passo'], o['c']) for o in onde]
    if len(cv) >= 3:
        xs = np.array([a for a, _ in cv]); cs = np.array([b for _, b in cv])
        A = np.vstack([np.ones_like(xs), xs**2]).T
        c0, b2 = np.linalg.lstsq(A, cs, rcond=None)[0]
        res['c0'] = float(c0)
        res['beta_quadratica'] = float(-b2 / c0)
    return res


def estrapola(res, passo):
    """gamma(k) = coeff * (k a)^p in unita' di tempo in cui c=1 e le lunghezze sono in unita' del
    reticolo; cammino libero = c/gamma. Con a = lunghezza di Planck e k a = E / E_Planck."""
    if 'potenza_p' not in res:
        return []
    p, C = res['potenza_p'], res['coeff']
    righe = []
    for nome, E in [("luce visibile, 2 eV", 2.0), ("raggi X, 10 keV", 1e4),
                    ("raggi gamma, 1 GeV", 1e9), ("raggi gamma, 1 TeV", 1e12)]:
        ka = E / E_PLANCK_EV
        gamma = C * ka**p                                  # per unita' di tempo (lunghezze in unita' del reticolo)
        cammino_celle = (1.0 / gamma) / passo              # in passi del reticolo
        cammino_m = cammino_celle * L_PLANCK
        righe.append((nome, cammino_m, cammino_m / RAGGIO_UNIVERSO))
    return righe


# ---------------------------------------------------------------------------
# esecuzione
# ---------------------------------------------------------------------------

def esegui_tipo(tipo, n, rng, args, X=None, L=None):
    log(f"=== reticolo: {tipo} ===")
    if X is None:
        X, L = GENERATORI[tipo](n, rng)
    n = len(X)
    passo = float(np.mean(cKDTree(X, boxsize=L).query(X, k=2)[0][:, 1]))
    log(f"  {n} celle, lato {L:.2f}, passo medio (vicina piu' prossima) {passo:.3f}")
    if args.salva_reticolo:
        nome = args.salva_reticolo if not args.confronto else f"{tipo}_{args.salva_reticolo}"
        np.savez_compressed(nome, X=X, L=L, tipo=tipo)
        log(f"  reticolo salvato in {nome}")
    S = fattore_struttura(X, L, rng)
    log("  S(k): " + "  ".join(f"{k:.2f}:{s:.4f}" for k, s in S[:8]))
    var = varianza_sfere(X, L, rng)
    log("  varianza/media in sfere: " + "  ".join(f"R={R}:{v:.3f}" for R, v in var))
    an = anisotropia_locale(X, L, grafo(X, L)[3])
    log(f"  anisotropia locale delle vicine: {an:.3f}")
    misure = []
    if not args.solo_reticolo:
        modi = ['voronoi', 'bilanciati'] if args.pesi == 'entrambi' else [args.pesi]
        for modo in modi:
            log(f"  --- onde, pesi: {modo} ---")
            onde = misura_onde(X, L, passo, kmax_passo=args.kmax, tempo=args.tempo, rng=rng, pesi=modo)
            misure.append(dict(onde=onde, analisi=analizza(onde['onde'], passo)))
    return dict(tipo=tipo, n=n, L=float(L), passo=passo, S=S, varianza=var,
                anisotropia=an, misure=misure)


def grafici(risultati, cartella):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        log("matplotlib non disponibile: niente grafici")
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for r in risultati:
        k = [a for a, _ in r['S']]; s = [b for _, b in r['S']]
        ax[0].loglog(k, np.maximum(s, 1e-6), 'o-', ms=3, label=r['tipo'])
        for mis in r['misure']:
            lab = f"{r['tipo']}, {mis['onde']['pesi']}"
            tutte = sorted(mis['onde']['onde'], key=lambda o: o['k_passo'])
            ok = [o for o in tutte if np.isfinite(o['gamma']) and o['gamma'] > 0]
            if ok:
                ax[1].loglog([o['k_passo'] for o in ok], [o['gamma'] for o in ok], 'o-', ms=4, label=lab)
            ax[2].plot([o['k_passo'] for o in tutte], [o['c'] for o in tutte], 'o-', ms=4, label=lab)
    ax[0].set_xlabel('k (unita' + "' di sigma)"); ax[0].set_ylabel('S(k)'); ax[0].set_title('fluttuazioni di densita')
    ax[1].set_xlabel('k * passo'); ax[1].set_ylabel('perdita di coerenza Gamma'); ax[1].set_title('diffusione delle onde')
    xs = np.array([0.05, 1.0])
    for pp, st in [(2, ':'), (4, '--')]:
        ax[1].loglog(xs, 1e-2 * xs**pp, 'k' + st, lw=0.8, label=f'pendenza {pp}')
    ax[2].set_xlabel('k * passo'); ax[2].set_ylabel('velocita'); ax[2].set_title('velocita della luce')
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    nome = os.path.join(cartella, 'mt_trasparenza.png')
    fig.savefig(nome, dpi=130)
    log(f"grafici salvati in {nome}")


def main():
    ap = argparse.ArgumentParser(description="Max Theory: trasparenza del vuoto di celle")
    ap.add_argument('--tipo', default='crescita', choices=list(GENERATORI))
    ap.add_argument('--confronto', action='store_true', help='esegue esclusione, equilibrio, crescita, cristallo')
    ap.add_argument('--n', type=int, default=100000, help='numero di celle')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--kmax', type=float, default=1.0, help='k*passo massimo delle onde')
    ap.add_argument('--tempo', type=float, default=None, help='durata della propagazione (predefinita: 1,5 volte il lato)')
    ap.add_argument('--solo-reticolo', action='store_true', help='solo le misure del reticolo, niente onde')
    ap.add_argument('--pesi', default='voronoi', choices=['voronoi', 'bilanciati', 'entrambi'],
                    help="come pesare il passaggio fra celle vicine")
    ap.add_argument('--salva-reticolo', default=None)
    ap.add_argument('--carica', default=None, help='file .npz di un reticolo gia' + "' generato")
    ap.add_argument('--cartella', default='risultati_mt')
    args = ap.parse_args()
    os.makedirs(args.cartella, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    log(f"densita' {RHO:.4f} celle per unita' di volume, portata della spinta {SIGMA}")

    risultati = []
    if args.carica:
        z = np.load(args.carica)
        risultati.append(esegui_tipo(str(z['tipo']), len(z['X']), rng, args, X=z['X'], L=float(z['L'])))
    elif args.confronto:
        for t in ['esclusione', 'equilibrio', 'crescita', 'cristallo']:
            risultati.append(esegui_tipo(t, args.n, rng, args))
    else:
        risultati.append(esegui_tipo(args.tipo, args.n, rng, args))

    print("\n" + "=" * 78)
    print("RIEPILOGO")
    print("=" * 78)
    for r in risultati:
        print(f"\n{r['tipo']}: {r['n']} celle, passo {r['passo']:.3f}, anisotropia locale {r['anisotropia']:.3f}")
        if r['S']:
            print(f"  S(k) alle due scale piu' grandi: {r['S'][0][1]:.4f}, {r['S'][1][1] if len(r['S']) > 1 else float('nan'):.4f}"
                  "   (casuale ~1; iperuniforme -> 0)")
        for mis in r['misure']:
            a, o = mis['analisi'], mis['onde']
            dis = o['disordine']
            sp_ = dis['spettro']
            print(f"  [pesi {o['pesi']}] facce o vicine per cella {o['vicine']:.1f}; disordine del passaggio: "
                  f"isotropo {dis['dev_isotropa']:.3f}, anisotropo {dis['dev_anisotropa']:.3f}")
            if sp_:
                print(f"    lunghezza di correlazione dell'anisotropia xi = {dis['xi']:.2f} = {dis['xi'] / r['passo']:.1f} passi; "
                      f"spettro dell'anisotropia alla scala piu' grande / piu' piccola: {sp_[0][2]:.4f} / {sp_[-1][2]:.4f}")
                if 'potenza_p' in a and np.isfinite(dis['xi']):
                    kmin = min(o2['k'] for o2 in o['onde'])
                    print(f"    onde misurate: k*xi da {kmin * dis['xi']:.2f} in su "
                          f"(il regime k^4 richiede k*xi molto sotto 0,5)")
            if a.get('onde_totali') and a['onde_trasparenti'] >= a['onde_totali'] / 2:
                print(f"    diffusione: nessuna perdita di coerenza rilevata in {a['onde_trasparenti']} onde su {a['onde_totali']}"
                      " -> trasparente alla risoluzione di questa simulazione")
            elif 'potenza_p' in a:
                print(f"    diffusione: Gamma ~ (k*passo)^p con p = {a['potenza_p']:.2f} (onde lunghe), {a['potenza_p_tutte']:.2f} (tutte)")
                for nome, cm, fr in estrapola(a, r['passo']):
                    verdetto = "nitida" if fr > 1 else "SFOCATA"
                    print(f"      {nome:22s} cammino libero {cm:9.2e} m = {fr:9.2e} raggi dell'universo  -> {verdetto}")
            if 'beta_quadratica' in a:
                print(f"    velocita': c(k) = c0 [1 - beta (k*passo)^2] con beta = {a['beta_quadratica']:.3f}")
    print("\n  p ~ 4: diffusione da disordine locale (trascurabile per la luce reale)")
    print("  p ~ 2: diffusione forte (la luce delle galassie lontane sarebbe sfocata)")
    print("  xi finito e spettro dell'anisotropia piatto a k piccolo: disordine locale, k^4 per onde lunghe")
    print("  spettro dell'anisotropia che cresce verso k piccolo: disordine correlato a lunga distanza")
    print("  (le estrapolazioni valgono solo se p resta stabile per onde sempre piu' lunghe)")
    with open(os.path.join(args.cartella, 'risultati.json'), 'w') as f:
        json.dump(risultati, f, indent=1, default=float)
    log(f"risultati completi in {os.path.join(args.cartella, 'risultati.json')}")
    grafici(risultati, args.cartella)


if __name__ == '__main__':
    main()
