# PureKLU.jl

PureKLU.jl is a pure-Julia direct port of [SuiteSparse's KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/KLU)
sparse LU solver, with no SuiteSparse binary dependency. The API mirrors
[`KLU.jl`](https://github.com/JuliaSparse/KLU.jl) so the two are usually drop-in
compatible.

Because it is written entirely in Julia, it works with generic element types
(`Float32`, `BigFloat`, complex numbers, `ForwardDiff.Dual`, …) and supports an
allocation-free refactorization path for repeated solves with the same sparsity
pattern.

## Installation

```julia
using Pkg
Pkg.add("PureKLU")
```

## Quick start

```@example quickstart
using PureKLU, SparseArrays

A = sparse([1.0 0.0; 1.0 2.0])
K = klu(A)
K \ [3.0, 5.0]  # solves A * x = b
```

Once factored, `K` can be reused to solve against new right-hand sides, and
[`klu!`](@ref) / [`klu_refactor!`](@ref) refactor it in place when only the
numeric values change (same sparsity pattern).

## Bit-for-bit equivalence with KLU.jl

`klu(A; use_fma = Val(false))` reproduces `KLU.jl`'s results bit-for-bit for every
field (`L`, `U`, `F`, `p`, `q`, `R`, `Rs`, solve and refactor output) on the general
factorization. The default `use_fma = Val(true)` fuses the kernel's
multiply-subtract loops into FMA instructions: same result up to one ULP, and
faster on hardware with FMA.

```julia
klu(A)                       # FMA on (default)
klu(A; use_fma = false)      # SSE2-equivalent, bit-for-bit KLU.jl match
klu(A; use_fma = Val(false)) # same, fully type-stable
```

!!! note "Banded fast path and bit-for-bit equivalence"
    `detect_banded` (**on by default**) factors narrow-band blocks in their
    natural order, skipping AMD. This is faster but uses a different elimination
    order, so the result is *not* bit-identical to `KLU.jl` (it is a different
    but equally valid factorization that solves to full accuracy). Pass
    `klu(A; detect_banded = false)` to force the general AMD path, which *is*
    byte-for-byte identical to libklu.

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

See the [API reference](@ref api) for the full list of exported functions.
