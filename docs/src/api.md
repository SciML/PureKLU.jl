# [API reference](@id api)

## Factorization

```@docs
klu
klu_analyze!
klu_factor!
```

`klu` returns a `LinearAlgebra.Factorization`. Use the generic
`F \\ b`, `LinearAlgebra.ldiv!(F, b)`, `size(F)`, and
`LinearAlgebra.issuccess(F)` interfaces rather than relying on factorization
storage. The documented KLU-compatible properties (`L`, `U`, `F`, `p`, `q`,
`R`, and `Rs`) are available after the appropriate analysis or factorization
phase.

```@docs
PureKLU.KLUFactorization
```

## In-place refactorization

```@docs
klu!
klu_refactor!
```

## Solving

```@docs
solve!
```

## Pivot growth

```@docs
PureKLU.rgrowth
```

## SparseArrays integration

```@docs
SparseArrays.nonzeros(::PureKLU.AbstractKLUFactorization)
```

## Internals

These describe the internal ports that back the public API.

```@docs
PureKLU
PureKLU.BTF
PureKLU.AMD
PureKLU.AMD.amd_order!
```

## Updating values with new numerical pivots

`klu!(factor, values; reuse_pivots = false)` accepts a vector of replacement CSC
values or a sparse matrix with the same pattern. It retains symbolic analysis and
numeric workspace while selecting numerical pivots again. Use this option when
changing matrix values can make the old pivots unstable. With the default
`reuse_pivots = true`, the caller remains responsible for checking the accuracy of
refactorization, as described in the [KLU user guide](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/KLU/Doc/KLU_UserGuide.tex).
[`PureKLU.rgrowth`](@ref) is that check: refactor with the pivots reused, and
redo the factorization with `klu_factor!` only when the growth it reports says
the old pivots have gone unstable. That keeps the cheap refactorization on the
common path and pays for new pivots only when they are needed.
