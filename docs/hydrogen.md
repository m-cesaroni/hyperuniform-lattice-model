# Hydrogen on a hyperuniform random lattice

Numerical study of the hydrogen atom on a three-dimensional **random, relaxed (hyperuniform) point lattice**, using a discrete Laplacian built from the Delaunay–Voronoi dual (face area / edge length weights). The code generates the lattice, builds the operator, solves for the lowest states and compares energies, degeneracies and dipole transition strengths with the exact continuum values.

Everything here runs on a laptop: a 100 000-cell lattice in ~10 minutes, a 2.7-million-cell box (built by tiling a relaxed lattice) in under an hour with ~20 GB of RAM.

## What is computed

* Lattice: `N` points with hard minimum distance, relaxed to mechanical equilibrium (closed pushes) with a finite-range repulsion; the resulting point set is hyperuniform and has no preferred axes.
* Operator: `H = (1/2μ) L − α/r`, with `L` the symmetrized graph Laplacian whose link weights are `A_ij / ℓ_ij` (Voronoi face area over link length) and node weights the Voronoi volumes — i.e. the 0–1 sector of discrete exterior calculus on the Voronoi complex. `μ` and `α` are chosen so that the Bohr radius `a0 = 1/(αμ)` spans a given number of lattice cells; the nucleus is softened at half a cell.
* Solver: LOBPCG with analytic hydrogen functions as initial guesses, 14 lowest states, identified by overlap with the analytic 1s, 2s, 2p, 3s, 3p, 3d sets.
* Observables: level energies vs Rydberg `E_n = −μα²/2n²`; degeneracy of the 2p triplet and the 3d quintet; the accidental 2s–2p and 3s–3p–3d degeneracies; oscillator strengths of Lyman-α, Lyman-β and three Balmer components from dipole matrix elements, plus manifold-summed strengths that are robust to mixing within degenerate groups.

## Main results (2.7·10⁶ cells, box 126 cells)

| quantity | lattice | continuum |
|---|---|---|
| 1s energy (a0 = 4 cells) | −1.2 % | — |
| 2p energy, triplet splitting | +0.17 %, 4·10⁻⁵ | 0 |
| 2s–2p separation | 0.45 % | 0 |
| 3s, 3p, 3d (box 50 a0, a0 = 2.5 cells) | within 0.4 % of each other and of Rydberg/9 | degenerate |
| Lyman-α oscillator strength (a0 = 4 cells, box 32 a0) | 0.425 | 0.4162 |
| Lyman-α, extrapolated to infinite resolution and box (6 runs) | 0.410 – 0.414 | 0.4162 |
| Balmer 2p→3d oscillator strength | 0.690 | 0.6958 |
| manifold-summed strengths, box 50 a0: 1s→n=3, 2s→n=3, 2p→n=3 | 0.082, 0.468, 0.699 | 0.0791, 0.4349, 0.7094 |

Error model: `f = f∞ + c_a/a0² + c_b/box^n` fits six runs with residuals ≤ 0.003 and predicted the largest run (0.4247) before it was made (0.4253).

Controls (160 000-cell tiled lattices): packing fraction 0.60–0.80 changes the levels by ≤ 0.3 % and Lyman-α by ≤ 1 %; periodic boundaries and a spherical soft wall give identical n = 2 results to 0.05 %; a 10 % random perturbation of the link weights changes the levels by ≤ 0.4 %; a lattice relaxed ten times less well, ≤ 0.3 %. The small 2+3 splitting of the 3d quintet in periodic boxes is the cubic field of the periodic images: it disappears under a spherical wall.

Full outputs of every run are in `results/`.

## How to run

```
pip install -r requirements.txt
python mt_idrogeno_pc.py --n 100000 --a0 3.0 --out test                       # ~10 min, ~2 GB
python mt_idrogeno_pc.py --reticolo test_reticolo.npz --tile 2 --a0 3.0 --out box28    # tile the relaxed lattice 2×2×2
python mt_idrogeno_pc.py --reticolo test_reticolo.npz --tile 3 --a0 4.0 --salva_varchi links3.npz --out big   # 2.7 M cells
python mt_idrogeno_pc.py --reticolo big_reticolo.npz --varchi links3.npz --a0 2.5 --out big25          # reuse the links
```

Options: `--phi` packing fraction, `--parete` soft spherical wall, `--rumore` weight noise, `--nucleo` core softening, `--passi` relaxation steps, `--seed`.

## Limitations

The 1s error scales as `1/a0²` (nucleus resolution); box effects scale as an inverse power of the box size; n = 3 states need ≥ 50 Bohr radii of box. The solver can return mixtures within degenerate groups: use the manifold-summed oscillator strengths for those. This is a non-relativistic Schrödinger calculation; spin, fine structure and the Dirac operator on the same lattice are not included here.

## Motivation

The lattice and the operator come from a discrete model of space in which cells carry states and the space between them carries the rules of passage. The present repository only asks whether such a random, equilibrium lattice reproduces ordinary atomic physics without tuning; the model itself is not needed to run or read the code.

## Citation

See `CITATION.cff`. Archived releases carry a DOI on Zenodo.
