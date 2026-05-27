# PureKLU benchmarks

This folder follows the [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl)
convention: `benchmarks.jl` exposes a top-level `SUITE::BenchmarkGroup`
that AsV / `PkgBenchmark.jl` can pick up automatically.

## Quick run (no AsV install needed)

```bash
julia --project=benchmark benchmark/benchmarks.jl
```

This executes the suite end-to-end and prints a side-by-side comparison
table for each phase (`analyze_factor`, `analyze`, `factor_only`,
`refactor`, `solve`, `tsolve`) with both `pureklu_fma` (FMA on, default)
and `pureklu_nofma` (FMA off, bit-for-bit `KLU.jl`-equivalent) columns.

## AirspeedVelocity (compare branches / tags)

Install AsV once:

```bash
julia -e 'using Pkg; Pkg.add("AirspeedVelocity")'
```

Then from the repo root:

```bash
benchpkg PureKLU --rev=main,HEAD --bench-on=HEAD
benchpkgtable PureKLU --rev=main,HEAD
benchpkgplot  PureKLU --rev=main,HEAD
```

Or compare against a baseline JSON:

```bash
benchpkg PureKLU --rev=HEAD --output-dir=results --tune
```

## Suite layout

The `SUITE` is a nested `BenchmarkGroup`:

```
SUITE
├── analyze_factor
│   ├── laplacian_20x20
│   │   ├── klu_jl
│   │   ├── pureklu_fma
│   │   └── pureklu_nofma
│   ├── ...
├── analyze       (klu_analyze! only)
├── factor_only   (klu_factor! given an already-analysed K)
├── refactor      (klu!(K, A) re-using the existing pattern)
├── solve         (K \ b, vector RHS)
└── tsolve        (K' \ b, adjoint solve)
```

Backends inside each leaf:

  - `klu_jl`         — SuiteSparse `libklu.so` via `KLU.jl` (reference)
  - `pureklu_fma`    — PureKLU with `use_fma=Val(true)` (default; FMA fused)
  - `pureklu_nofma`  — PureKLU with `use_fma=Val(false)` (bit-for-bit
                        identical to `klu_jl`)

The `analyze` group omits `pureklu_fma`/`pureklu_nofma` because the
analyze phase has no FMA-sensitive arithmetic.
