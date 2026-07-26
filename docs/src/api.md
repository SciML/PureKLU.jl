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
