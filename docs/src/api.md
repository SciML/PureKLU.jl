# [API reference](@id api)

## Factorization

```@docs
klu
klu_analyze!
klu_factor!
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

## Accessors

```@docs
nonzeros(::PureKLU.AbstractKLUFactorization)
```

## Internals

These describe the internal ports that back the public API.

```@docs
PureKLU
PureKLU.BTF
PureKLU.AMD
PureKLU.AMD.amd_order!
```
