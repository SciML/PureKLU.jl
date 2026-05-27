# Per-phase bit-for-bit tests against the SuiteSparse reference.
#
# Each phase of the pipeline is checked independently:
#   - BTF (maxtrans + strongcomp + order)            -> P, Q, R, structural_rank
#   - AMD direct comparison vs SuiteSparse           -> permutation match
#   - klu_scale (Rs computation, every scale mode)
#   - Symbolic struct (nz, nblocks, maxblock, lnz/unz estimates, structural_rank)
#   - Numeric struct (lnz, unz, max_lnz_block, max_unz_block, noffdiag)
#   - Pnum (final global pivot permutation)
#   - L/U/F structure + values
#   - solve/tsolve (vector + matrix, real + complex)
#   - klu_refactor numerical equivalence

using Test
using SparseArrays
using LinearAlgebra
using PureKLU
using Random
import KLU
import AMD as SuiteSparseAMD

# Pull the not-exported BTF/AMD modules so we can hit the internals.
const PKBTF = PureKLU.BTF
const PKAMD = PureKLU.AMD

# ---------- helpers ---------------------------------------------------------

"""
    klu_internal_btf(A::SparseMatrixCSC)

Read KLU.jl's symbolic struct directly to get the BTF permutation
(`P_btf`), column permutation (`Q_btf`), block boundaries (`R`) and the
structural rank. KLU.jl exposes them only through pointer-loaded
`klu_symbolic` / `klu_l_symbolic` C structs.
"""
function klu_internal_btf(A::SparseMatrixCSC)
    F = KLU.klu(A)
    Sym = F.symbolic
    n = Int(Sym.n)
    Q_btf = Vector{Int}(undef, n)
    P_btf = Vector{Int}(undef, n)
    R = Vector{Int}(undef, Int(Sym.nblocks) + 1)
    # The Symbolic struct holds Q, P, R as Ptr{Int32} or Ptr{Int64}.
    Tptr = eltype(Sym.Q)  # Int32 or Int64 (Ptr base)
    Ti = Tptr === Ptr{Int64} ? Int64 : Int32
    Qarr = unsafe_wrap(Array, Sym.Q, n; own=false)
    Parr = unsafe_wrap(Array, Sym.P, n; own=false)
    Rarr = unsafe_wrap(Array, Sym.R, Int(Sym.nblocks)+1; own=false)
    Q_btf .= Int.(Qarr)
    P_btf .= Int.(Parr)
    R .= Int.(Rarr)
    return P_btf, Q_btf, R, Int(Sym.structural_rank), Int(Sym.nblocks),
           Int(Sym.maxblock), Int(Sym.nz), Int(Sym.nzoff)
end

"""
    pureklu_internal_btf(A)

Same fields, read from PureKLU's symbolic struct.
"""
function pureklu_internal_btf(A::SparseMatrixCSC)
    F = PureKLU.klu(A; use_fma=false)
    Sym = getfield(F, :symbolic)
    return Int.(Sym.P), Int.(Sym.Q), Int.(Sym.R[1:Int(Sym.nblocks)+1]),
           Int(Sym.structural_rank), Int(Sym.nblocks),
           Int(Sym.maxblock), Int(Sym.nz), Int(Sym.nzoff)
end

# ---------- BTF.order! direct comparison -----------------------------------

@testset "BTF.order!: structural fields match KLU.jl symbolic" begin
    test_matrices = [
        sparse([2.0 3.0 0; 1.0 0 0; 0 0 1.0]),
        sparse([1.0 0 0 0; 0 2.0 0 0; 1.0 1.0 3.0 0; 0 0 1.0 4.0]),
        let
            Ap = [0,4,1,1,2,2,0,1,2,3,4,4] .+ 1
            Ai = [0,4,0,2,1,2,1,4,3,2,1,2] .+ 1
            Ax = [2.,1.,3.,4.,-1.,-3.,3.,6.,2.,1.,4.,2.]
            sparse(Ap, Ai, Ax)
        end,
    ]
    Random.seed!(1)
    for n in (5, 10, 20, 50)
        push!(test_matrices, sprand(n, n, 0.1) + n*I)
    end
    for (idx, A) in enumerate(test_matrices)
        klu_data = klu_internal_btf(A)
        pj_data  = pureklu_internal_btf(A)
        @testset "matrix #$idx" begin
            # KLU.jl's Symbolic stores the combined BTF+AMD permutations in P, Q,
            # not the raw BTF outputs, so we just compare those combined results.
            @test klu_data[1] == pj_data[1]            # P
            @test klu_data[2] == pj_data[2]            # Q
            @test klu_data[3] == pj_data[3]            # R
            @test klu_data[4] == pj_data[4]            # structural_rank
            @test klu_data[5] == pj_data[5]            # nblocks
            @test klu_data[6] == pj_data[6]            # maxblock
            @test klu_data[7] == pj_data[7]            # nz
            @test klu_data[8] == pj_data[8]            # nzoff
        end
    end
end

@testset "BTF.order!: matches BTF directly on structurally singular matrices" begin
    # square structurally rank-deficient matrix
    A = sparse([1 2; 0 0])  # row 2 is all-zero -> structurally singular
    A = SparseMatrixCSC{Float64}(A)
    # KLU.jl doesn't expose full_factor; build factorisation objects and run
    # analyze only.
    F_ref = KLU.KLUFactorization(A); KLU.klu_analyze!(F_ref)
    F_pj  = PureKLU.KLUFactorization(A); F_pj.common.use_fma = Val(false)
    PureKLU.klu_analyze!(F_pj)
    Sym_ref = F_ref.symbolic
    Sym_pj  = getfield(F_pj, :symbolic)
    @test Sym_ref.nblocks == Sym_pj.nblocks
    @test Sym_ref.structural_rank == Sym_pj.structural_rank
    @test Sym_ref.nz == Sym_pj.nz
    @test Sym_ref.nzoff == Sym_pj.nzoff
end

# ---------- AMD direct comparison vs SuiteSparse AMD ------------------------

@testset "AMD.amd_order!: matches SuiteSparse AMD on diverse blocks" begin
    Random.seed!(0)
    sizes = (4, 5, 10, 15, 25, 50, 100)
    for n in sizes, density in (0.1, 0.2, 0.4)
        # symmetric pattern: A + A'
        Asym = sprand(n, n, density)
        Asym = Asym + Asym' + 2n*I
        dropzeros!(Asym)
        Ap0 = Vector{Int64}(Asym.colptr .- 1)
        Ai0 = Vector{Int64}(Asym.rowval .- 1)

        # SuiteSparse AMD via AMD.jl
        ref_perm = Int64.(SuiteSparseAMD.amd(Asym)) .- 1  # 1-based -> 0-based

        # Mine
        my_perm = Vector{Int64}(undef, n)
        status = PKAMD.amd_order!(n, Ap0, Ai0, my_perm)
        @test status == PKAMD.AMD_OK
        @test my_perm == ref_perm
    end
end

@testset "AMD.amd_order!: matches SuiteSparse AMD on small known patterns" begin
    # 1x1 trivial
    @test PKAMD.amd_order!(1, Int64[0, 0], Int64[], Vector{Int64}(undef, 1)) ==
          PKAMD.AMD_OK

    # diagonal-only
    n = 5
    Ap = zeros(Int64, n+1)
    Ai = Int64[]
    P = Vector{Int64}(undef, n)
    PKAMD.amd_order!(n, Ap, Ai, P)
    ref_perm = Int64.(SuiteSparseAMD.amd(sparse(I, n, n))) .- 1
    @test P == ref_perm

    # chain
    n = 6
    A = spdiagm(-1 => ones(n-1), 1 => ones(n-1))
    A = A + n*I
    dropzeros!(A)
    Ap = Vector{Int64}(A.colptr .- 1)
    Ai = Vector{Int64}(A.rowval .- 1)
    P_mine = Vector{Int64}(undef, n)
    PKAMD.amd_order!(n, Ap, Ai, P_mine)
    ref_perm = Int64.(SuiteSparseAMD.amd(A)) .- 1
    @test P_mine == ref_perm
end

# ---------- klu_scale -------------------------------------------------------

@testset "klu_scale!: every scale mode matches KLU.jl's Rs" begin
    Random.seed!(42)
    for scale in (0, 1, 2), n in (5, 10, 25, 50)
        A = sprand(n, n, 0.2) + n*I
        # configure both with the same scale mode
        K_ref = KLU.klu(A; check=false)
        K_pj  = PureKLU.klu(A; check=false, use_fma=false)
        K_ref.common.scale = Int32(scale)
        K_pj.common.scale  = Int32(scale)
        KLU.klu_factor!(K_ref)
        PureKLU.klu_factor!(K_pj)
        @test K_ref.Rs ≈ K_pj.Rs
    end
end

# ---------- symbolic struct equality ---------------------------------------

@testset "Symbolic struct fields: nz, nblocks, maxblock, nzoff, structural_rank" begin
    Random.seed!(11)
    test_cases = [
        sprand(5, 5, 0.3) + 5*I,
        sprand(15, 15, 0.2) + 15*I,
        sprand(50, 50, 0.05) + 50*I,
        sprand(100, 100, 0.05) + 100*I,
    ]
    for A in test_cases
        K_ref = KLU.klu(A)
        K_pj  = PureKLU.klu(A; use_fma=false)
        @test K_ref.symbolic.nz == getfield(K_pj, :symbolic).nz
        @test K_ref.symbolic.nzoff == getfield(K_pj, :symbolic).nzoff
        @test K_ref.symbolic.nblocks == getfield(K_pj, :symbolic).nblocks
        @test K_ref.symbolic.maxblock == getfield(K_pj, :symbolic).maxblock
        @test K_ref.symbolic.structural_rank ==
              getfield(K_pj, :symbolic).structural_rank
    end
end

# ---------- numeric struct equality ----------------------------------------

@testset "Numeric struct fields: lnz, unz, max_lnz_block, max_unz_block, noffdiag" begin
    Random.seed!(13)
    test_cases = [
        sprand(5, 5, 0.3) + 5*I,
        sprand(15, 15, 0.2) + 15*I,
        sprand(50, 50, 0.05) + 50*I,
        sprand(100, 100, 0.05) + 100*I,
    ]
    for A in test_cases
        K_ref = KLU.klu(A)
        K_pj  = PureKLU.klu(A; use_fma=false)
        @test K_ref.numeric.lnz == getfield(K_pj, :numeric).lnz
        @test K_ref.numeric.unz == getfield(K_pj, :numeric).unz
        @test K_ref.numeric.max_lnz_block ==
              getfield(K_pj, :numeric).max_lnz_block
        @test K_ref.numeric.max_unz_block ==
              getfield(K_pj, :numeric).max_unz_block
        @test K_ref.numeric.nzoff == getfield(K_pj, :numeric).nzoff
        @test K_ref.common.noffdiag == K_pj.common.noffdiag
    end
end

# ---------- L, U, F and permutations exact-equality ------------------------

function strict_match_all(A::SparseMatrixCSC)
    K_ref = KLU.klu(A)
    K_pj  = PureKLU.klu(A; use_fma=false)
    @test K_ref.p == K_pj.p
    @test K_ref.q == K_pj.q
    @test K_ref.R == K_pj.R
    @test K_ref.Rs == K_pj.Rs
    @test K_ref.L == K_pj.L
    @test K_ref.U == K_pj.U
    @test K_ref.F == K_pj.F
end

# ---------- sparse matrix zoo: structure variety ---------------------------

@testset "Matrix zoo: arrow patterns" begin
    for n in (10, 25, 50)
        # arrow: dense first row + dense first col + diagonal
        I_idx = vcat(fill(1, n-1), 2:n, 1:n)
        J_idx = vcat(2:n, fill(1, n-1), 1:n)
        V     = vcat(ones(n-1), ones(n-1), Float64(n) .* ones(n))
        A = sparse(I_idx, J_idx, V, n, n)
        strict_match_all(A)
    end
end

@testset "Matrix zoo: banded matrices of varying bandwidth" begin
    for n in (20, 50, 100), bw in (1, 2, 3, 5)
        diags = Dict{Int, Vector{Float64}}()
        diags[0] = Float64(n) .* ones(n)
        for d in 1:bw
            diags[ d] = -ones(n-d)
            diags[-d] = -ones(n-d)
        end
        A = spdiagm((k => v for (k, v) in diags)...)
        dropzeros!(A)
        strict_match_all(A)
    end
end

@testset "Matrix zoo: block-diagonal matrices" begin
    Random.seed!(99)
    block_sizes_list = [(3, 5, 7), (10, 10, 10), (1, 1, 1, 1, 5), (8, 1, 8)]
    for sizes in block_sizes_list
        n = sum(sizes)
        blocks = Any[]
        for sz in sizes
            push!(blocks, sprand(sz, sz, 0.5) + sz*I)
        end
        A = blockdiag(blocks...)
        strict_match_all(A)
    end
end

@testset "Matrix zoo: upper triangular + sparse lower entries" begin
    Random.seed!(7)
    for n in (10, 30, 60)
        U = triu(sprand(n, n, 0.3)) + n*I
        # add a few sparse lower entries
        for _ in 1:max(1, n ÷ 4)
            i = rand(2:n); j = rand(1:i-1)
            U[i, j] = randn()
        end
        strict_match_all(U)
    end
end

@testset "Matrix zoo: very sparse matrices (near singletons)" begin
    Random.seed!(55)
    for n in (15, 30, 60)
        density = 1.5 / n  # roughly 1.5 entries per row
        A = sprand(n, n, density)
        A = A + (n+1) * I
        dropzeros!(A)
        strict_match_all(A)
    end
end

@testset "Matrix zoo: complex Hermitian-ish with imaginary off-diag" begin
    Random.seed!(31)
    for n in (10, 20, 50)
        Are = sprand(n, n, 0.2)
        Aim = sprand(n, n, 0.2)
        A = Are + im*Aim + n*I
        K_ref = KLU.klu(A); K_pj = PureKLU.klu(A; use_fma=false)
        @test K_ref.p == K_pj.p
        @test K_ref.q == K_pj.q
        @test K_ref.L ≈ K_pj.L
        @test K_ref.U ≈ K_pj.U
        @test K_ref.F ≈ K_pj.F
    end
end

# ---------- solve / tsolve variations --------------------------------------

@testset "solve: vector, matrix, complex RHS all match" begin
    Random.seed!(7)
    A = sprand(20, 20, 0.2) + 20*I
    K_ref = KLU.klu(A); K_pj = PureKLU.klu(A; use_fma=false)

    # vector RHS
    b = randn(20)
    @test K_ref \ b ≈ K_pj \ b

    # matrix RHS
    B = randn(20, 5)
    @test K_ref \ B ≈ K_pj \ B

    # complex RHS on real factor
    bc = randn(ComplexF64, 20)
    x_ref = K_ref \ bc
    x_pj  = K_pj  \ bc
    @test x_ref ≈ x_pj
end

@testset "tsolve / adjoint solve match" begin
    Random.seed!(8)
    A = sprand(20, 20, 0.2) + 20*I
    K_ref = KLU.klu(A); K_pj = PureKLU.klu(A; use_fma=false)
    b = randn(20)
    B = randn(20, 4)

    @test K_ref' \ b ≈ K_pj' \ b
    @test transpose(K_ref) \ b ≈ transpose(K_pj) \ b
    @test K_ref' \ B ≈ K_pj' \ B
    @test transpose(K_ref) \ B ≈ transpose(K_pj) \ B
end

@testset "Complex transpose vs adjoint: both match" begin
    Random.seed!(9)
    A = sprand(ComplexF64, 15, 15, 0.3)
    A = A + 15*I
    K_ref = KLU.klu(A); K_pj = PureKLU.klu(A; use_fma=false)
    b = randn(ComplexF64, 15)

    # adjoint: A^H x = b
    x_ref_adj = K_ref' \ b
    x_pj_adj  = K_pj' \ b
    @test x_ref_adj ≈ x_pj_adj
    @test A' * x_pj_adj ≈ b

    # transpose: A^T x = b
    x_ref_tr = transpose(K_ref) \ b
    x_pj_tr  = transpose(K_pj) \ b
    @test x_ref_tr ≈ x_pj_tr
    @test transpose(A) * x_pj_tr ≈ b
end

# ---------- klu_refactor numerical equivalence -----------------------------

@testset "klu_refactor: bit-for-bit L/U/F/Rs match after refactor" begin
    Random.seed!(17)
    # build a pattern and two value sets sharing it
    for n in (10, 20, 50)
        proto = sprand(n, n, 0.2) + n*I
        I_idx, J_idx, _ = findnz(proto)
        V1 = randn(length(I_idx)) .+ 0.5
        V2 = randn(length(I_idx)) .+ 0.5
        A = sparse(I_idx, J_idx, V1, n, n)
        B = sparse(I_idx, J_idx, V2, n, n)
        K_ref = KLU.klu(A); KLU.klu!(K_ref, B)
        K_pj  = PureKLU.klu(A; use_fma=false); PureKLU.klu!(K_pj, B)
        @test K_ref.L == K_pj.L
        @test K_ref.U == K_pj.U
        @test K_ref.F == K_pj.F
        @test K_ref.Rs == K_pj.Rs
        # refactor with just nzval
        K_ref2 = KLU.klu(A); KLU.klu!(K_ref2, B.nzval)
        K_pj2  = PureKLU.klu(A; use_fma=false); PureKLU.klu!(K_pj2, B.nzval)
        @test K_ref2.L == K_pj2.L
        @test K_ref2.U == K_pj2.U
        @test K_ref2.F == K_pj2.F
    end
end

# ---------- klu_analyze_given path -----------------------------------------

@testset "Custom P/Q via klu_analyze_given path" begin
    Random.seed!(33)
    for n in (5, 10, 20)
        A = sprand(n, n, 0.3) + n*I
        # When btf=0 and ordering=2, KLU uses identity-ish processing.
        K_ref = KLU.klu(A; check=false)
        K_pj  = PureKLU.klu(A; check=false, use_fma=false)
        K_ref.common.btf = Int32(0); K_pj.common.btf = Int32(0)
        K_ref.common.ordering = Int32(2); K_pj.common.ordering = Int32(2)
        KLU.klu_factor!(K_ref)
        PureKLU.klu_factor!(K_pj)
        @test K_ref.p == K_pj.p
        @test K_ref.q == K_pj.q
        @test K_ref.L == K_pj.L
        @test K_ref.U == K_pj.U
        b = randn(n)
        @test K_ref \ b ≈ K_pj \ b
    end
end

# ---------- bigger zoo: 2D Laplacians up to 30x30 grid ---------------------

@testset "Matrix zoo: 2D Laplacians of various sizes" begin
    for k in (3, 5, 8, 12, 16, 20, 25)
        n = k * k
        diag_main = 4.0 * ones(n)
        diag_off1 = -1.0 * ones(n-1)
        for i in 1:n-1
            if i % k == 0
                diag_off1[i] = 0.0
            end
        end
        diag_offk = -1.0 * ones(n-k)
        A = spdiagm(-k => diag_offk, -1 => diag_off1, 0 => diag_main,
                    1 => diag_off1, k => diag_offk)
        dropzeros!(A)
        strict_match_all(A)
    end
end

@testset "Matrix zoo: circuit-style sparse with off-diagonal blocks" begin
    Random.seed!(42)
    for n in (15, 30, 60)
        # Build a matrix with sparse off-diagonal pattern resembling a circuit
        Idx = Int[]
        Jdx = Int[]
        for i in 1:n
            push!(Idx, i); push!(Jdx, i)               # diagonal
            j = mod1(i + 1, n)
            push!(Idx, i); push!(Jdx, j)
            push!(Idx, j); push!(Jdx, i)
            if i + 3 <= n
                push!(Idx, i); push!(Jdx, i+3)
                push!(Idx, i+3); push!(Jdx, i)
            end
        end
        V = randn(length(Idx)) .+ 1.5
        # accumulate diagonal mass
        A = sparse(Idx, Jdx, V, n, n) + n*I
        dropzeros!(A)
        strict_match_all(A)
    end
end

# ---------- large-scale strict-equality matrix zoo -------------------------

@testset "Matrix zoo: 200×200 and 300×300 random (strict)" begin
    Random.seed!(2024)
    for n in (200, 300), density in (0.01, 0.02, 0.05)
        A = sprand(n, n, density) + n*I
        dropzeros!(A)
        strict_match_all(A)
    end
end

@testset "Matrix zoo: 5x5 block-sparse pattern (many small blocks)" begin
    Random.seed!(101)
    nblocks_list = (3, 5, 10, 20)
    for nb in nblocks_list
        block_sz = 4
        n = nb * block_sz
        Idx = Int[]; Jdx = Int[]
        for b in 1:nb
            base = (b-1)*block_sz + 1
            for i in base:(base+block_sz-1), j in base:(base+block_sz-1)
                push!(Idx, i); push!(Jdx, j)
            end
            if b < nb
                push!(Idx, base);                  push!(Jdx, base + block_sz) # interblock
            end
        end
        V = randn(length(Idx)) .+ 1.5
        A = sparse(Idx, Jdx, V, n, n) + n*I
        dropzeros!(A)
        strict_match_all(A)
    end
end

# ---------- multi-refactor stress test -------------------------------------

@testset "klu_refactor: multiple refactors with different values in a row" begin
    # Strict bit-for-bit equality on every iteration. (Earlier versions
    # tolerated ~eps drift here because my internal Li/Lx storage stayed
    # unsorted while KLU.jl's klu_sort -- triggered by accessing K.L --
    # mutates KLU's internal storage in place. PureKLU now ports klu_sort
    # and invokes it from L/U extraction, restoring identical FP-
    # accumulation order in subsequent refactors.)
    Random.seed!(73)
    n = 30
    proto = sprand(n, n, 0.2) + n*I
    I_idx, J_idx, _ = findnz(proto)
    K_ref = KLU.klu(proto)
    K_pj  = PureKLU.klu(proto; use_fma=false)
    for iter in 1:5
        V = randn(length(I_idx)) .+ 0.7
        B = sparse(I_idx, J_idx, V, n, n)
        KLU.klu!(K_ref, B)
        PureKLU.klu!(K_pj, B)
        @test K_ref.L == K_pj.L
        @test K_ref.U == K_pj.U
        @test K_ref.F == K_pj.F
        @test K_ref.Rs == K_pj.Rs
        b = randn(n)
        @test K_ref \ b == K_pj \ b
    end
end

# ---------- consistency invariants on both implementations -----------------

@testset "Consistency: L*U + F = Rs \\ A[p,q] for both implementations" begin
    Random.seed!(123)
    for n in (10, 30, 60, 100)
        A = sprand(n, n, 0.15) + n*I
        dropzeros!(A)
        for K in (KLU.klu(A), PureKLU.klu(A; use_fma=false))
            Rs = Diagonal(K.Rs)
            @test Rs \ A[K.p, K.q] ≈ K.L * K.U + K.F
        end
    end
end

# ---------- ldiv! vs \ consistency -----------------------------------------

@testset "ldiv! vs \\\\: both implementations consistent" begin
    Random.seed!(81)
    A = sprand(40, 40, 0.15) + 40*I
    K_pj = PureKLU.klu(A; use_fma=false)
    K_ref = KLU.klu(A)
    b = randn(40)
    x1 = K_pj \ b
    x2 = copy(b); ldiv!(K_pj, x2)
    x3 = K_ref \ b
    x4 = copy(b); ldiv!(K_ref, x4)
    @test x1 ≈ x2 ≈ x3 ≈ x4
end

# ---------- analyze_given with user permutation ----------------------------

# `klu_analyze!(K, P, Q)` exercises KLU's `klu_analyze_given` C path; KLU.jl
# does no index translation, so callers must pass 0-based permutations.
# We just verify both implementations produce the same factorisation when
# given the identity permutation in 0-based form (we don't compare to a
# completely different P/Q because KLU.jl's API path through
# klu_l_analyze_given is fragile -- it segfaults on malformed input
# regardless of implementation).
@testset "klu_analyze! with user-given P=Q=identity (analyze_given path)" begin
    Random.seed!(91)
    n = 12
    A = sprand(n, n, 0.3) + n*I
    P_user = Vector{Int64}(0:(n-1))     # 0-based identity
    Q_user = Vector{Int64}(0:(n-1))
    K_ref = KLU.KLUFactorization(A)
    KLU.klu_analyze!(K_ref, copy(P_user), copy(Q_user))
    K_pj = PureKLU.KLUFactorization(A)
    K_pj.common.use_fma = Val(false)
    PureKLU.klu_analyze!(K_pj, copy(P_user), copy(Q_user))
    @test K_ref.symbolic.nz == getfield(K_pj, :symbolic).nz
    @test K_ref.symbolic.nblocks == getfield(K_pj, :symbolic).nblocks
    KLU.klu_factor!(K_ref); PureKLU.klu_factor!(K_pj)
    @test K_ref.p == K_pj.p
    @test K_ref.q == K_pj.q
    @test K_ref.L == K_pj.L
    @test K_ref.U == K_pj.U
end

# ---------- 1×1 / 2×2 degenerate matrices ----------------------------------

@testset "Degenerate sizes 1x1, 2x2" begin
    # 1x1
    A1 = sparse(reshape([3.0], 1, 1))
    K_ref = KLU.klu(A1); K_pj = PureKLU.klu(A1; use_fma=false)
    @test K_ref.p == K_pj.p
    @test K_ref.q == K_pj.q
    @test K_ref.U == K_pj.U
    @test K_ref \ [6.0] == K_pj \ [6.0]

    # 2x2 various
    for M in (
        Float64[1 0; 0 2],
        Float64[2 1; 0 3],
        Float64[1 0; 1 2],
        Float64[3 1; 1 3],
    )
        A = sparse(M)
        K_ref = KLU.klu(A); K_pj = PureKLU.klu(A; use_fma=false)
        @test K_ref.p == K_pj.p
        @test K_ref.q == K_pj.q
        @test K_ref.L == K_pj.L
        @test K_ref.U == K_pj.U
    end
end

# ---------- structurally symmetric vs unsymmetric --------------------------

@testset "Structurally symmetric & unsymmetric matches" begin
    Random.seed!(202)
    for n in (20, 40, 80)
        # structurally symmetric pattern with unsymmetric values
        Asym = sprand(n, n, 0.15)
        Asym = Asym + Asym' + n*I
        dropzeros!(Asym)
        I_idx, J_idx, _ = findnz(Asym)
        A1 = sparse(I_idx, J_idx, randn(length(I_idx)) .+ 1.5, n, n) + n*I
        dropzeros!(A1)
        strict_match_all(A1)

        # structurally unsymmetric
        A2 = tril(sprand(n, n, 0.2)) + triu(sprand(n, n, 0.05)) + n*I
        dropzeros!(A2)
        strict_match_all(A2)
    end
end
