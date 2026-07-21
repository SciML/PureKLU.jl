using SciMLTesting, PureKLU, Test
using JET

run_qa(
    PureKLU;
    explicit_imports = true,
    ei_kwargs = (;
        no_stale_explicit_imports = (;
            # `@muladd` is imported from MuladdMacro but currently unused; kept as a
            # deliberate dependency for future opt-in FMA (see src/Kernel.jl).
            ignore = (Symbol("@muladd"),),
        ),
        all_qualified_accesses_are_public = (;
            # Cross-package non-public names accessed qualified and needed:
            #   AdjointFactorization / TransposeFactorization (LinearAlgebra) are
            #     used under `isdefined` guards for cross-version compatibility.
            #   RefValue (Base) is used as a struct field type.
            ignore = (:AdjointFactorization, :TransposeFactorization, :RefValue),
        ),
    ),
)
