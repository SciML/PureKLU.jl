using SciMLTesting, PureKLU

# ExplicitImports only sees an extension module once its trigger package is loaded, so the
# weakdep has to be loaded here for `PureKLUForwardDiffExt` to be scanned at all.
using ForwardDiff

# ForwardDiff declares no name `public` and exports none of its dual-number interface, so
# the only spelling available for the types and accessors an AD extension must use is the
# non-public one.
const FORWARDDIFF_NONPUBLIC = (:Dual, :Partials, :value, :partials)

run_qa(
    PureKLU;
    ei_kwargs = (;
        all_explicit_imports_are_public = (; ignore = FORWARDDIFF_NONPUBLIC),
        all_qualified_accesses_are_public = (; ignore = FORWARDDIFF_NONPUBLIC),
    )
)
