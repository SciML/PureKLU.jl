"""
    KLUSymbolic{Ti}

Pure-Julia analogue of SuiteSparse's `klu_symbolic`. Stores the
fill-reducing symbolic ordering (`P` for rows, `Q` for columns) together
with the block-triangular-form boundaries `R`. The `Lnz` array holds
per-block estimates of nonzeros in `L`.
"""
mutable struct KLUSymbolic{Ti <: Integer}
    symmetry::Float64
    est_flops::Float64
    lnz::Float64
    unz::Float64
    Lnz::Vector{Float64}
    n::Ti
    nz::Ti
    P::Vector{Ti}
    Q::Vector{Ti}
    R::Vector{Ti}
    nzoff::Ti
    nblocks::Ti
    maxblock::Ti
    ordering::Ti
    do_btf::Ti
    structural_rank::Ti
end

# Empty placeholder; `n == 0` is the "not yet analyzed" sentinel checked
# throughout the package.  Lets `KLUFactorization.symbolic` be a concrete
# non-Union field.
KLUSymbolic{Ti}() where {Ti <: Integer} = KLUSymbolic{Ti}(
    EMPTY_FLOAT, EMPTY_FLOAT, EMPTY_FLOAT, EMPTY_FLOAT,
    Float64[],
    Ti(0), Ti(0),
    Ti[], Ti[], Ti[],
    Ti(0), Ti(0), Ti(0),
    Ti(0), Ti(0), Ti(EMPTY),
)

"""
    klu_analyze(n, Ap, Ai, common; given_P=nothing, given_Q=nothing) -> KLUSymbolic

Symbolic analysis routine. Computes the BTF permutation (if enabled),
then for each diagonal block applies AMD/COLAMD or, for blocks of size
≤3, natural ordering. Returns the full symbolic factorization.

When `given_P` and `given_Q` are supplied, BTF is skipped and the given
permutations are used directly (mirrors `klu_analyze_given`).
"""
function klu_analyze(
        n::Integer, Ap::Vector{Ti}, Ai::Vector{Ti},
        common::KLUCommon{Ti};
        given_P::Union{Nothing, Vector{Ti}} = nothing,
        given_Q::Union{Nothing, Vector{Ti}} = nothing
    ) where {Ti <: Integer}
    common.status = KLU_OK
    common.structural_rank = Ti(EMPTY)

    if given_P !== nothing || given_Q !== nothing
        return _analyze_given(n, Ap, Ai, given_P, given_Q, common)
    elseif common.ordering == 2
        return _analyze_given(n, Ap, Ai, nothing, nothing, common)
    else
        return _order_and_analyze(n, Ap, Ai, common)
    end
end

function _allocate_symbolic(
        n::Ti, Ap::Vector{Ti}, Ai::Vector{Ti},
        common::KLUCommon{Ti}
    ) where {Ti}
    if n <= 0 || Ap[1] != 0 || Ap[n + 1] < 0
        common.status = KLU_INVALID
        return nothing
    end
    nz = Ap[n + 1]
    # Column-pointer monotonicity. Once this passes, `Ap` is non-decreasing,
    # so the row indices read below cover exactly Ai[1:nz] (Ap[1]==0), the
    # same access set the per-column loop would visit.
    @inbounds for j in 1:n
        if Ap[j] > Ap[j + 1]
            common.status = KLU_INVALID
            return nothing
        end
    end
    # Row-index bounds as a single flat vectorizable scan. `unsigned(i) >= un`
    # is bit-exact equivalent to `i < 0 || i >= n` (negative i wraps to a huge
    # unsigned), collapsing the two-branch range test into one comparison the
    # compiler can SIMD.
    un = unsigned(n)
    @inbounds for p in 1:nz
        if unsigned(Ai[p]) >= un
            common.status = KLU_INVALID
            return nothing
        end
    end

    return KLUSymbolic{Ti}(
        EMPTY_FLOAT, EMPTY_FLOAT, EMPTY_FLOAT, EMPTY_FLOAT,
        Vector{Float64}(undef, n),
        n, nz,
        Vector{Ti}(undef, n),
        Vector{Ti}(undef, n),
        Vector{Ti}(undef, n + 1),
        Ti(0), Ti(0), Ti(1),
        Ti(common.ordering), Ti(common.btf), Ti(EMPTY),
    )
end

function _order_and_analyze(
        n, Ap::Vector{Ti}, Ai::Vector{Ti},
        common::KLUCommon{Ti}
    ) where {Ti}
    Sym = _allocate_symbolic(Ti(n), Ap, Ai, common)
    Sym === nothing && return nothing

    if common.ordering ∉ (0, 1, 3)
        common.status = KLU_INVALID
        return nothing
    end

    Pbtf = Vector{Ti}(undef, n)
    Qbtf = Vector{Ti}(undef, n)

    # One scratch buffer serves two disjoint phases.  BTF.order! consumes its
    # first 5n entries; after it returns that storage is dead, so the per-block
    # worker reuses the buffer for its own Pinv (n) / Cp (maxblock+1 ≤ n+1) /
    # Ci (nz+1) scratch via views.  Sizing it to cover both uses (5n for BTF,
    # 2n+nz+2 for the worker since maxblock ≤ n) replaces the worker's three
    # separate allocations with zero, matching C's single preallocated Common
    # workspace.
    nz = Int(Sym.nz)
    Work = Vector{Ti}(undef, max(5n, 2n + nz + 2))

    common.work = 0.0
    do_btf = common.btf != 0
    Sym.do_btf = do_btf ? Ti(1) : Ti(0)
    Sym.ordering = Ti(common.ordering)

    if do_btf
        nblocks, nmatch, work = BTF.order!(
            Int(n), Ap, Ai, common.maxwork,
            Pbtf, Qbtf, Sym.R, Work
        )
        Sym.structural_rank = Ti(nmatch)
        common.structural_rank = Sym.structural_rank
        common.work += work
        if Sym.structural_rank < n
            @inbounds for k in 1:n
                Qbtf[k] = Ti(BTF.btf_unflip(Int(Qbtf[k])))
            end
        end
        maxblock = 1
        @inbounds for b in 1:nblocks
            nk = Int(Sym.R[b + 1]) - Int(Sym.R[b])
            maxblock = max(maxblock, nk)
        end
        Sym.nblocks = Ti(nblocks)
        Sym.maxblock = Ti(maxblock)
    else
        Sym.R[1] = Ti(0)
        Sym.R[2] = Ti(n)
        @inbounds for k in 1:n
            Pbtf[k] = Ti(k - 1)
            Qbtf[k] = Ti(k - 1)
        end
        Sym.nblocks = Ti(1)
        Sym.maxblock = Ti(n)
    end

    _analyze_worker!(Sym, Int(n), Ap, Ai, Pbtf, Qbtf, Int(common.ordering), common, Work)
    return Sym
end

function _analyze_given(
        n_in, Ap::Vector{Ti}, Ai::Vector{Ti},
        P_in::Union{Nothing, Vector{Ti}},
        Q_in::Union{Nothing, Vector{Ti}},
        common::KLUCommon{Ti}
    ) where {Ti}
    Sym = _allocate_symbolic(Ti(n_in), Ap, Ai, common)
    Sym === nothing && return nothing
    n = Int(n_in)
    nz = Int(Sym.nz)

    Sym.ordering = Ti(2)
    do_btf = common.btf != 0
    Sym.do_btf = do_btf ? Ti(1) : Ti(0)

    # Q = Quser or identity (in 0-based)
    if Q_in === nothing
        @inbounds for k in 1:n
            Sym.Q[k] = Ti(k - 1)
        end
    else
        @inbounds for k in 1:n
            Sym.Q[k] = Q_in[k]
        end
    end

    nblocks = 1
    nzoff = 0
    maxblock = 1

    if do_btf
        # If a user P is given, compute B = Puser * A by rewriting Ai through Puser's inverse.
        Pinv = Vector{Ti}(undef, n)
        if P_in !== nothing
            @inbounds for k in 1:n
                Pinv[Int(P_in[k]) + 1] = Ti(k - 1)
            end
            Bi = Vector{Ti}(undef, max(nz, 1))
            @inbounds for p in 1:nz
                Bi[p] = Pinv[Int(Ai[p]) + 1]
            end
        else
            Bi = Ai
        end

        Work = Vector{Ti}(undef, 4n)
        # strongcomp modifies Q in place and writes Sym.P, Sym.R
        nblocks = BTF.strongcomp!(n, Ap, Bi, Sym.Q, Sym.P, Sym.R, Work)

        # P = P_btf * Puser
        if P_in !== nothing
            Wk = Vector{Ti}(undef, n)
            @inbounds for k in 1:n
                Wk[k] = P_in[Int(Sym.P[k]) + 1]
            end
            @inbounds for k in 1:n
                Sym.P[k] = Wk[k]
            end
        end

        # Pinv = inverse of Sym.P
        @inbounds for k in 1:n
            Pinv[Int(Sym.P[k]) + 1] = Ti(k - 1)
        end

        # count nzoff and maxblock per block
        for block in 0:(nblocks - 1)
            k1 = Int(Sym.R[block + 1])
            k2 = Int(Sym.R[block + 2])
            nk = k2 - k1
            maxblock = max(maxblock, nk)
            for k in k1:(k2 - 1)
                oldcol = Int(Sym.Q[k + 1])
                pend = Int(Ap[oldcol + 2])
                for p in Int(Ap[oldcol + 1]):(pend - 1)
                    if Int(Pinv[Int(Ai[p + 1]) + 1]) < k1
                        nzoff += 1
                    end
                end
            end
            Sym.Lnz[block + 1] = EMPTY_FLOAT
        end
    else
        Sym.R[1] = Ti(0)
        Sym.R[2] = Ti(n)
        if P_in === nothing
            @inbounds for k in 1:n
                Sym.P[k] = Ti(k - 1)
            end
        else
            @inbounds for k in 1:n
                Sym.P[k] = P_in[k]
            end
        end
        maxblock = n
        nblocks = 1
        Sym.Lnz[1] = EMPTY_FLOAT
    end

    Sym.nblocks = Ti(nblocks)
    Sym.maxblock = Ti(maxblock)
    Sym.nzoff = Ti(nzoff)
    Sym.lnz = EMPTY_FLOAT
    Sym.unz = EMPTY_FLOAT
    Sym.est_flops = EMPTY_FLOAT
    return Sym
end

function _analyze_worker!(
        Sym::KLUSymbolic{Ti}, n::Int,
        Ap::Vector{Ti}, Ai::Vector{Ti},
        Pbtf::Vector{Ti}, Qbtf::Vector{Ti},
        ordering::Int, common::KLUCommon{Ti},
        Work::Vector{Ti}
    ) where {Ti}
    P = Sym.P; Q = Sym.Q; R = Sym.R; Lnz = Sym.Lnz
    nblocks = Int(Sym.nblocks)
    maxblock = Int(Sym.maxblock)

    # Pinv (n), Cp (maxblock+1) and Ci (nz+1) are carved from the dead BTF
    # scratch.  Work was sized to hold all three disjoint slices (the caller
    # guarantees length ≥ 2n+nz+2 ≥ n+(maxblock+1)+(nz+1)), and BTF's use of
    # the buffer is fully complete before the worker runs, so this is safe.
    nz = Int(Sym.nz)
    Pinv = view(Work, 1:n)
    Cp = view(Work, (n + 1):(n + maxblock + 1))
    Ci = view(Work, (n + maxblock + 2):(n + maxblock + nz + 2))
    @inbounds for k in 1:n
        Pinv[Int(Pbtf[k]) + 1] = Ti(k - 1)
    end

    nzoff = 0
    lnz = 0.0
    flops = 0.0
    maxnz = 0
    Sym.symmetry = EMPTY_FLOAT

    Pblk = Vector{Ti}(undef, maxblock)

    @inbounds for block in 0:(nblocks - 1)
        k1 = Int(R[block + 1])
        k2 = Int(R[block + 2])
        nk = k2 - k1

        # build the block's CSC pattern
        Lnz[block + 1] = EMPTY_FLOAT
        pc = 0
        for k in k1:(k2 - 1)
            Cp[k - k1 + 1] = Ti(pc)
            oldcol = Int(Qbtf[k + 1])
            pend = Int(Ap[oldcol + 2])
            for p in (Int(Ap[oldcol + 1]) + 1):pend
                newrow = Int(Pinv[Int(Ai[p]) + 1])
                if newrow < k1
                    nzoff += 1
                else
                    Ci[pc + 1] = Ti(newrow - k1)
                    pc += 1
                end
            end
        end
        Cp[nk + 1] = Ti(pc)
        maxnz = max(maxnz, pc)

        # order the block
        lnz1 = 0.0
        flops1 = 0.0
        ok = true
        if nk <= 3
            for k in 1:nk
                Pblk[k] = Ti(k - 1)
            end
            lnz1 = nk * (nk + 1) / 2
            flops1 = nk * (nk - 1) / 2 + (nk - 1) * nk * (2nk - 1) / 6
        elseif ordering == 0
            # For a block whose nonzero pattern is contained in a narrow band,
            # the natural ordering already yields band-confined fill that AMD
            # cannot improve on (any permutation of a within-band-dense block
            # produces >= band fill). Detecting this cheaply (one O(nnz) pass)
            # lets us skip the AMD ordering entirely for true banded systems.
            banded, bw1 = common.detect_banded ? _block_bandwidth(nk, Cp, Ci) : (false, 0)
            if banded
                @inbounds for k in 1:nk
                    Pblk[k] = Ti(k - 1)
                end
                # Exact off-diagonal fill of a banded LU under natural order:
                # column k (0-based) has min(bw, nk-1-k) sub-diagonal L entries.
                lnz1 = _band_lnz(nk, bw1)
                flops1 = EMPTY_FLOAT
            else
                ok, lnz1, flops1 = _amd_or_natural!(Pblk, nk, Cp, Ci, common)
            end
        elseif ordering == 1
            # COLAMD - fallback to natural for now
            for k in 1:nk
                Pblk[k] = Ti(k - 1)
            end
            lnz1 = EMPTY_FLOAT
            flops1 = EMPTY_FLOAT
        else
            common.status = KLU_INVALID
            return
        end
        if !ok
            common.status = KLU_INVALID
            return
        end

        Lnz[block + 1] = lnz1
        lnz = (lnz == EMPTY_FLOAT || lnz1 == EMPTY_FLOAT) ? EMPTY_FLOAT : lnz + lnz1
        flops = (flops == EMPTY_FLOAT || flops1 == EMPTY_FLOAT) ? EMPTY_FLOAT : flops + flops1

        # combine Pblk with BTF P,Q
        for k in 0:(nk - 1)
            pk = Int(Pblk[k + 1]) + k1 + 1
            Q[k + k1 + 1] = Qbtf[pk]
            P[k + k1 + 1] = Pbtf[pk]
        end
    end

    Sym.lnz = lnz
    Sym.unz = lnz
    Sym.nzoff = Ti(nzoff)
    Sym.est_flops = flops
    common.status = KLU_OK
    return Sym
end

# Largest half-bandwidth for which the natural ordering is preferred over AMD.
# Natural-order banded LU fill is O(nk*bw); keeping `bw` tiny ensures the fill
# stays at-or-below what AMD produces and that the band detector never fires on
# matrices (e.g. 2D Laplacians, bw≈sqrt(n)) where the band is sparse and natural
# order would fill it in. True hardware/PDE band systems have bw of a few units.
const BAND_BW_MAX = 8

# Scan a block's CSC pattern (`Cp[1:nk+1]` / `Ci`) and return `(banded, bw)`
# where `bw` is the half-bandwidth max|i-j| over all entries.  Bails out as soon
# as any entry exceeds `BAND_BW_MAX` (so the common non-banded case costs only a
# partial scan).  One O(nnz) pass, no allocation.
#
# Beyond the narrow-bandwidth test, two guards ensure the natural ordering is
# chosen only when it is genuinely at-or-better than AMD (otherwise we would
# diverge from AMD's ordering for no -- or negative -- benefit):
#   1. size: the band must be narrow *relative to* the block (`nk >= 4(bw+1)`).
#      Without this an 8x8 dense block reads as bw<=7<=BAND_BW_MAX "banded" and
#      gets reordered, needlessly diverging from AMD (and from KLU bit-for-bit).
#   2. density: the band must be (near-)full (`2*nnz >= nk*(2bw+1)`).  A sparse
#      band -- e.g. a 2D Laplacian with only a few occupied diagonals -- would be
#      *filled in* by natural-order LU, which AMD avoids; those keep the AMD path.
#      A genuine PDE/hardware band block fills its band, so this passes for them.
@inline function _block_bandwidth(nk::Int, Cp::AbstractVector{Ti}, Ci::AbstractVector{Ti}) where {Ti}
    bw = 0
    @inbounds for j in 0:(nk - 1)
        p1 = Int(Cp[j + 1])
        p2 = Int(Cp[j + 2])
        for p in p1:(p2 - 1)
            d = Int(Ci[p + 1]) - j
            d = ifelse(d < 0, -d, d)
            if d > BAND_BW_MAX
                return (false, 0)
            end
            bw = ifelse(d > bw, d, bw)
        end
    end
    @inbounds pc = Int(Cp[nk + 1])
    if nk < 4 * (bw + 1) || 2 * pc < nk * (2 * bw + 1)
        return (false, 0)
    end
    return (true, bw)
end

# Exact off-diagonal L nonzero count of an `nk`-column banded LU with
# half-bandwidth `bw` under the natural ordering (no fill outside the band):
# column k (0-based) contributes min(bw, nk-1-k) sub-diagonal entries.
@inline function _band_lnz(nk::Int, bw::Int)
    if nk <= bw + 1
        # Whole trailing block is within the band: dense lower triangle.
        return Float64(nk * (nk - 1) ÷ 2)
    end
    # Columns 0 .. nk-1-bw each have `bw` sub-diagonals; the last `bw` columns
    # taper as bw-1, bw-2, ..., 0.
    full = (nk - bw) * bw
    taper = bw * (bw - 1) ÷ 2
    return Float64(full + taper)
end

# AMD ordering for blocks larger than 3. The block's CSC pattern already
# lives contiguously in `Cp[1:nk+1]` / `Ci[1:pc]`, and `amd_order!` only ever
# reads its `Ap`/`Ai` (the matrix is never mutated), so we pass views directly
# rather than copying into fresh `Ap_blk`/`Ai_blk`. The permutation is written
# into the first `nk` entries of `Pblk` (which `amd_order!` is the sole writer
# of), eliminating a third scratch vector + copy.
function _amd_or_natural!(
        Pblk::Vector{Ti}, nk::Int, Cp::AbstractVector{Ti}, Ci::AbstractVector{Ti},
        common::KLUCommon{Ti}
    ) where {Ti}
    pc = Int(Cp[nk + 1])
    Ap_blk = view(Cp, 1:(nk + 1))
    Ai_blk = view(Ci, 1:max(pc, 1))
    status, lnz1 = AMD.amd_order!(nk, Ap_blk, Ai_blk, Pblk)
    if status < 0
        return false, 0.0, 0.0
    end
    # `lnz1` is AMD's exact off-diagonal L fill count (SuiteSparse Info[AMD_LNZ]);
    # `_block_cap` adds the `nk` diagonal entries.  The flop count is left as the
    # coarse dense estimate (used only for the informational `Sym.est_flops`).
    flops1 = nk * (nk - 1) / 2 + (nk - 1) * nk * (2nk - 1) / 6
    return true, lnz1, flops1
end
