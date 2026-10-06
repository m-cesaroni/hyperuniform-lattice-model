# Analysis scripts

Small scripts that produced specific numbers quoted in `docs/nota-tecnica.html`. Run `python prepara_reticoli.py` first (it writes the lattices they read into `reticoli/`), then any of:

| script | result in the note |
|---|---|
| `canali_koide.py` | typical imbalance of the three apertures around a channel vs packing fraction (Koide angle candidate) |
| `vicine_riempimento.py` | number of rigid neighbours N and channel imbalance vs packing fraction (α fixes the filling) |
| `indeterminazione.py` | Δx·Δk from ⟨K²⟩ and Δω·N½ from survival under e^{−iK} |
| `commutatore_lyman.py` | check of [X_a,K] = iV_a and Lyman-α quadrature with analytic vs lattice states |
| `settori_urto.py` | sector decomposition of a push on a face / channel / cell / void (light sector share) |
| `urto_meccanico.py` | conservative push dynamics: equilibrium vs random lattice, face/channel/void directions |
| `desi_geometrico.py` | geometric fit of DESI DR2 BAO + CMB acoustic scale with Λ and with a varying birth rate |
| `figures.py` | regenerates the four figures of `docs/figures/` from the lattices, the hydrogen outputs and the saved S(k) data |

The light-propagation results (speed, dispersion, scattering) come from `../mt_trasparenza.py` and `../mt_luce.py`; the hydrogen results from `../mt_idrogeno_pc.py`. The Kähler–Dirac spectral checks use `../kdlib.py`. The gravity numbers quoted in the note (1/r profile from the critical contagion, Mercury from saturation) were obtained with scripts from an earlier phase that are not yet consolidated here; they will be added in a later release.
