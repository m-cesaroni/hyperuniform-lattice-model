# Max Theory (MT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23189702.svg)](https://doi.org/10.5281/zenodo.23189702)

A discrete model of space: a three-dimensional random lattice of cells that do not touch, held in mechanical equilibrium (every cell's pushes close), with the space between cells — the *interstice* — carrying the rules of passage. Light, matter, time and gravity are read as ways of using the cells and the passages between them: light as a phase shift that runs through the faces and twists around the edges of the Voronoi complex; mass as the reflection of a vibration at the interstice; proper time as the count of those reflections; gravity as the fraction of a cell's steps spent on engagements launched by stationing matter.

The name honours Max Planck and his *mathematical artifice* of 1900 — the first acceptance that nature counts in steps. It has nothing to do with the author's name, which is why the author signs with an initial.

**This is a working model, not an established theory.** Everything in this repository is labelled with one of three statuses: *demonstrated* (a theorem or a reproducible numerical result), *coherent* (a mechanism that reproduces data but still depends on a choice or a scale not yet derived), *open* (a problem to solve, with its target number). The status map is the heart of `docs/riepilogo.html`.

## What is verified (reproducible with the code in `code/`)

| result | number | reference |
|---|---|---|
| Speed of light on a random equilibrium lattice (discrete exterior calculus) | 1.0002, isotropic to 1 % | 1 |
| Light dispersion | quadratic in E/E_Planck, scale 2.7 E_Planck; GRB 090510 delay 2·10⁻¹⁹ s | linear excluded |
| Vacuum birefringence; scattering of long waves | < 10⁻⁷; undetected (< 1.2·10⁻⁷), onset ∝ k⁶ | transparent vacuum |
| Kähler–Dirac vibration: E² = K² + m², I² + T² = 1 | exact to 10⁻¹⁴; no spurious modes; two tastes split by 0.1–0.3 % | theorem + measured |
| Uncertainty relations from the spectrum of K | Δx·Δk = 0.49 (≥ 1 cell), Δω·N½ = 0.80 ≥ π/4 | Robertson, Mandelstam–Tamm |
| Hydrogen (Schrödinger of MT, up to 2.7·10⁶ cells) | 1s −1.2 %; 2p +0.17 % (triplet 4·10⁻⁵); 2s–2p 0.45 %; n = 3 manifold 0.4 % | Rydberg, accidental degeneracies |
| Lyman-α oscillator strength (direct / extrapolated) | 0.425 / 0.410–0.414 | 0.4162 |
| Balmer 2p→3d; manifold sums 1s, 2s, 2p → n = 3 | 0.690; 0.082, 0.468, 0.699 | 0.6958; 0.0791, 0.4349, 0.7094 |
| Gravitational profile from critical contagion; perihelion of Mercury from saturation | r^(−0.986); 42.99″/century | 1/r; 42.98″ |
| Channel geometry: three faces around a Voronoi edge | three generations, Koide Q = 2/3 as identity | 0.666661 ± 7·10⁻⁶ |

Details, methods and the full list of controls are in `docs/nota-tecnica.html` (Italian) and `docs/hydrogen.md` (English).

## Falsifiable predictions

Dark energy constant (w = −1); black-hole shadow 4.6 % larger than in general relativity; no fourth generation; neutrino mass sum 60 meV, normal ordering, Majorana; no vacuum birefringence and only quadratic light dispersion. Each with the measurement expected to decide it, in `docs/nota-tecnica.html`.

## What is not derived

Four scales are taken from experiment, not derived: the electron's inversion amplitude (4·10⁻²³ per step), the rigidity of the vacuum against phase windings (the hadronic scale Λ, measured four ways), the birth rate of cells (4·10⁻⁶¹ per cell per step), and the weak scale (2·10⁻¹⁷ E_Planck). The two copies of the spin-½ vibration, the accord operator (weak sector), the dynamics of gravitational waves on the lattice and the growth of cosmic structure are open constructions. The list, with target numbers, closes `docs/riepilogo.html`.

## Repository layout

```
docs/          technical-note.html (technical note, English) · nota-tecnica.html (the same, Italian) · overview.html (full working document, English) · riepilogo.html (the same, Italian) · hydrogen.md (English)
docs/capitoli/ the working document split into six chapters, each with its status legend and open items, published as linked deposits:
               1 Il tessuto · 2 La vibrazione · 3 Quark e nuclei · 4 Le generazioni e la mano · 5 La gravità · 6 L'inizio e il cosmo
code/          mt_trasparenza.py (lattice generation, equilibrium, light scattering) · mt_luce.py (electromagnetic field, DEC)
               kdlib.py (Kähler–Dirac operator on the Voronoi complex) · mt_idrogeno_pc.py (hydrogen)
code/analysis/ the small scripts behind specific numbers in the note (channel geometry, uncertainty, commutator, sectors, pushes, DESI fit), with their own README
results/       outputs of the hydrogen runs
```

To run the hydrogen calculation:

```
pip install -r requirements.txt
cd code
python mt_idrogeno_pc.py --n 100000 --a0 3.0 --out test
```

## License and citation

Code: MIT. Documents: CC BY 4.0. See `CITATION.cff`; archived releases carry a Zenodo DOI.
