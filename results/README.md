# Hydrogen runs

All runs: α = 1/8, nucleus softened at half a cell, periodic box unless stated. `a0` is the Bohr radius in lattice cells; the box size is given in Bohr radii.

| file | cells | a0 (cells) | box (a0) | notes |
|---|---|---|---|---|
| `prova.txt` | 100 000 | 3.0 | 14 | first test; box too small for n = 2 |
| `idrogeno.txt` | 800 000 | 3.0 | 28 | lattice tiled 2×2×2; Lyman-α 0.431 |
| `balmer.txt` | 800 000 | 2.0 | 42 | n = 3 states enter; 2p→3d 0.678 |
| `balmer_parete.txt` | 800 000 | 2.0 | 42 | same with a spherical soft wall at 17 a0: n = 2 identical to periodic, 3d quintet degenerate |
| `lyman4.txt` | 800 000 | 4.0 | 21 | resolution test; box-limited |
| `grande.txt` | 2 700 000 | 4.0 | 32 | tiled 3×3×3; Lyman-α 0.425 (predicted 0.4247 by the error model) |
| `grande3.txt` | 2 700 000 | 3.0 | 42 | same lattice, reused links; Lyman-α 0.429, 2p→3d 0.683 |
| `grande25.txt` | 2 700 000 | 2.5 | 50 | n = 3 manifold within 0.4 %; manifold-summed strengths 0.082 / 0.468 / 0.699 |

The first five runs predate the manifold-summed strengths and the full 3d basis in the script; their n = 3 lines are box-limited or affected by mixing and should be read through the manifold sums of `grande25.txt`.
