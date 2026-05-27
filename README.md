# PureKLU.jl

[![Build Status](https://github.com/SciML/PureKLU.jl/workflows/CI/badge.svg)](https://github.com/SciML/PureKLU.jl/actions)

A pure-Julia direct port of [SuiteSparse's KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/KLU) sparse LU
solver — no SuiteSparse binary dependency. The API mirrors
[`KLU.jl`](https://github.com/JuliaSparse/KLU.jl) so the two are usually
drop-in compatible.

```julia
using PureKLU, SparseArrays
A = sparse([1.0 0.0; 1.0 2.0])
K = klu(A)
K \ [3.0, 5.0]  # solves A * x = b
```

## What's ported

A direct, line-by-line port of the following SuiteSparse C sources:

  - **BTF** — `btf_maxtrans`, `btf_strongcomp`, `btf_order`
  - **AMD** — `amd_order`, `amd_1`, `amd_2`, `amd_aat`, `amd_postorder`,
    `amd_post_tree`
  - **KLU** — `klu_analyze` / `klu_analyze_given`, `klu_scale`,
    `klu_kernel` (Gilbert-Peierls left-looking LU with symmetric pruning
    and diagonal preference), `klu_factor`, `klu_refactor`, `klu_solve`,
    `klu_tsolve`, `klu_sort`, `klu_extract`

The full pipeline runs entirely in Julia code; no `ccall`s to SuiteSparse.

## API compatibility with KLU.jl

| KLU.jl                  | PureKLU                       |
|-------------------------|-------------------------------|
| `klu(A)`                | `klu(A)`                      |
| `klu!(K, A)`            | `klu!(K, A)`                  |
| `klu_factor!(K)`        | `klu_factor!(K)`              |
| `klu_analyze!(K)`       | `klu_analyze!(K)`             |
| `K \ b`, `ldiv!(K, b)`  | same                          |
| `K.L`, `K.U`, `K.F`     | same                          |
| `K.p`, `K.q`, `K.R`     | same (1-based)                |
| `K.Rs`                  | same                          |
| `K.symbolic.nblocks`    | same                          |
| `K.numeric.lnz`         | same                          |

## Bit-for-bit equivalence

`PureKLU.klu(A; use_fma=Val(false))` produces results that are
**bit-for-bit identical** to `KLU.jl` for every field (`L`, `U`, `F`,
`p`, `q`, `R`, `Rs`, solve output, refactor output) across the test
suite (~1950 cases on real and complex matrices). The pure-Julia
implementation carefully matches SuiteSparse's exact arithmetic
choices, including:

  - Smith's ACM Algorithm 116 for complex division (not Julia's scaled
    Smith);
  - SuiteSparse's `x * sqrt(1 + r²)` complex magnitude (not Julia's
    `hypot`, which is 1 ULP more accurate);
  - No FMA fusion (`SuiteSparse_jll`'s `libklu.so` ships an SSE2-only
    baseline build, so its `c -= a*b` is `mulsd` + `subsd`, never
    `vfnmadd*sd`).

## FMA fusion toggle

By default (`use_fma=Val(true)`), the kernel multiply-subtract loops
compute `a - b*c` via `muladd(-b, c, a)`, which LLVM lowers to a fused
`vfnmadd*sd` on any x86 CPU with FMA. Same numerical result up to one
ULP at most; faster.

```julia
klu(A)                     # FMA on (default)
klu(A; use_fma=true)       # same
klu(A; use_fma=Val(true))  # same, fully type-stable
klu(A; use_fma=false)      # SSE2-equivalent, bit-for-bit KLU.jl match
klu(A; use_fma=Val(false)) # same
```

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Test suite covers:

  - All upstream `KLU.jl/test/runtests.jl` tests (144 cases)
  - Per-phase bit-for-bit comparisons against `KLU.jl` (BTF, AMD, scale,
    symbolic + numeric struct fields, L/U/F/Rs, refactor, solve, tsolve)
  - Matrix zoo: 2D Laplacians up to 60×60, banded, arrow, block-diag,
    structurally singular, complex Hermitian-ish, etc.
  - Both `use_fma=Val(true)` (≈) and `use_fma=Val(false)` (==) modes
  - Multi-iteration refactor sequences with strict `==`
  - Direct AMD comparison against `AMD.jl` (SuiteSparse AMD)

## Benchmarks

See [`benchmark/README.md`](benchmark/README.md). The suite is an
[AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl)-compatible
`BenchmarkGroup`.

## License

KLU/BTF (from SuiteSparse) are LGPL-2.1+. AMD is BSD-3-Clause. The
PureKLU port preserves the SuiteSparse upstream copyright notices in
each ported file. This package is released under the LGPL-2.1+ to
match the original KLU/BTF licensing.

## Acknowledgements

  - SuiteSparse / KLU: Timothy A. Davis and Ekanathan Palamadai.
  - BTF: Timothy A. Davis.
  - AMD: Timothy A. Davis, Patrick R. Amestoy, and Iain S. Duff.
  - `KLU.jl`: Raye Kimmerer.
