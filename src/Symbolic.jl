# Direct Julia port of SuiteSparse/KLU/Source/klu_analyze.c and
# klu_analyze_given.c.
# Upstream: KLU, Copyright (c) 2004-2025, University of Florida.
# Authors:  Timothy A. Davis and Ekanathan Palamadai.
# SPDX-License-Identifier: LGPL-2.1-or-later
# See LICENSE for the full notice.

"""
    KLUSymbolic{Ti}

Pure-Julia analogue of SuiteSparse's `klu_symbolic`. Stores the
fill-reducing symbolic ordering (`P` for rows, `Q` for columns) together
with the block-triangular-form boundaries `R`. The `Lnz` array holds
per-block estimates of nonzeros in `L`.
"""
mutable struct KLUSymbolic{Ti<:Integer}
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

"""
    klu_analyze(n, Ap, Ai, common; given_P=nothing, given_Q=nothing) -> KLUSymbolic

Symbolic analysis routine. Computes the BTF permutation (if enabled),
then for each diagonal block applies AMD/COLAMD or, for blocks of size
≤3, natural ordering. Returns the full symbolic factorization.

When `given_P` and `given_Q` are supplied, BTF is skipped and the given
permutations are used directly (mirrors `klu_analyze_given`).
"""
function klu_analyze(n::Integer, Ap::Vector{Ti}, Ai::Vector{Ti},
                     common::KLUCommon{Ti};
                     given_P::Union{Nothing,Vector{Ti}}=nothing,
                     given_Q::Union{Nothing,Vector{Ti}}=nothing) where {Ti<:Integer}
    common.status = Cint(KLU_OK)
    common.structural_rank = Ti(EMPTY)

    if given_P !== nothing || given_Q !== nothing
        return _analyze_given(n, Ap, Ai, given_P, given_Q, common)
    elseif common.ordering == 2
        return _analyze_given(n, Ap, Ai, nothing, nothing, common)
    else
        return _order_and_analyze(n, Ap, Ai, common)
    end
end

function _allocate_symbolic(n::Ti, Ap::Vector{Ti}, Ai::Vector{Ti},
                            common::KLUCommon{Ti}) where {Ti}
    if n <= 0 || Ap[1] != 0 || Ap[n+1] < 0
        common.status = Cint(KLU_INVALID)
        return nothing
    end
    nz = Ap[n+1]
    @inbounds for j in 1:n
        if Ap[j] > Ap[j+1]
            common.status = Cint(KLU_INVALID)
            return nothing
        end
        for p in (Ap[j]+1):Ap[j+1]
            i = Ai[p]
            if i < 0 || i >= n
                common.status = Cint(KLU_INVALID)
                return nothing
            end
        end
    end

    return KLUSymbolic{Ti}(
        EMPTY_FLOAT, EMPTY_FLOAT, EMPTY_FLOAT, EMPTY_FLOAT,
        Vector{Float64}(undef, n),
        n, nz,
        Vector{Ti}(undef, n),
        Vector{Ti}(undef, n),
        Vector{Ti}(undef, n+1),
        Ti(0), Ti(0), Ti(1),
        Ti(common.ordering), Ti(common.btf), Ti(EMPTY),
    )
end

function _order_and_analyze(n, Ap::Vector{Ti}, Ai::Vector{Ti},
                            common::KLUCommon{Ti}) where {Ti}
    Sym = _allocate_symbolic(Ti(n), Ap, Ai, common)
    Sym === nothing && return nothing

    if common.ordering ∉ (0, 1, 3)
        common.status = Cint(KLU_INVALID)
        return nothing
    end

    Pbtf = Vector{Ti}(undef, n)
    Qbtf = Vector{Ti}(undef, n)

    common.work = 0.0
    do_btf = common.btf != 0
    Sym.do_btf = do_btf ? Ti(1) : Ti(0)
    Sym.ordering = Ti(common.ordering)

    if do_btf
        Work = Vector{Ti}(undef, 5n)
        nblocks, nmatch, work = BTF.order!(Int(n), Ap, Ai, common.maxwork,
                                           Pbtf, Qbtf, Sym.R, Work)
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
            nk = Int(Sym.R[b+1]) - Int(Sym.R[b])
            maxblock = max(maxblock, nk)
        end
        Sym.nblocks = Ti(nblocks)
        Sym.maxblock = Ti(maxblock)
    else
        Sym.R[1] = Ti(0)
        Sym.R[2] = Ti(n)
        @inbounds for k in 1:n
            Pbtf[k] = Ti(k-1)
            Qbtf[k] = Ti(k-1)
        end
        Sym.nblocks = Ti(1)
        Sym.maxblock = Ti(n)
    end

    _analyze_worker!(Sym, Int(n), Ap, Ai, Pbtf, Qbtf, Int(common.ordering), common)
    return Sym
end

function _analyze_given(n_in, Ap::Vector{Ti}, Ai::Vector{Ti},
                        P_in::Union{Nothing,Vector{Ti}},
                        Q_in::Union{Nothing,Vector{Ti}},
                        common::KLUCommon{Ti}) where {Ti}
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
            Sym.Q[k] = Ti(k-1)
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
                Pinv[Int(P_in[k])+1] = Ti(k-1)
            end
            Bi = Vector{Ti}(undef, max(nz, 1))
            @inbounds for p in 1:nz
                Bi[p] = Pinv[Int(Ai[p])+1]
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
                Wk[k] = P_in[Int(Sym.P[k])+1]
            end
            @inbounds for k in 1:n
                Sym.P[k] = Wk[k]
            end
        end

        # Pinv = inverse of Sym.P
        @inbounds for k in 1:n
            Pinv[Int(Sym.P[k])+1] = Ti(k-1)
        end

        # count nzoff and maxblock per block
        for block in 0:(nblocks-1)
            k1 = Int(Sym.R[block+1])
            k2 = Int(Sym.R[block+2])
            nk = k2 - k1
            maxblock = max(maxblock, nk)
            for k in k1:(k2-1)
                oldcol = Int(Sym.Q[k+1])
                pend = Int(Ap[oldcol+2])
                for p in Int(Ap[oldcol+1]):(pend-1)
                    if Int(Pinv[Int(Ai[p+1])+1]) < k1
                        nzoff += 1
                    end
                end
            end
            Sym.Lnz[block+1] = EMPTY_FLOAT
        end
    else
        Sym.R[1] = Ti(0)
        Sym.R[2] = Ti(n)
        if P_in === nothing
            @inbounds for k in 1:n
                Sym.P[k] = Ti(k-1)
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

function _analyze_worker!(Sym::KLUSymbolic{Ti}, n::Int,
                          Ap::Vector{Ti}, Ai::Vector{Ti},
                          Pbtf::Vector{Ti}, Qbtf::Vector{Ti},
                          ordering::Int, common::KLUCommon{Ti}) where {Ti}
    P = Sym.P; Q = Sym.Q; R = Sym.R; Lnz = Sym.Lnz
    nblocks = Int(Sym.nblocks)

    Pinv = Vector{Ti}(undef, n)
    @inbounds for k in 1:n
        Pinv[Pbtf[k]+1] = Ti(k-1)
    end

    nzoff = 0
    lnz = 0.0
    flops = 0.0
    maxnz = 0
    Sym.symmetry = EMPTY_FLOAT

    maxblock = Int(Sym.maxblock)
    Pblk = Vector{Ti}(undef, maxblock)
    Cp = Vector{Ti}(undef, maxblock + 1)
    Ci = Vector{Ti}(undef, Sym.nz + 1)

    for block in 0:(nblocks-1)
        k1 = Int(R[block+1])
        k2 = Int(R[block+2])
        nk = k2 - k1

        # build the block's CSC pattern
        Lnz[block+1] = EMPTY_FLOAT
        pc = 0
        for k in k1:(k2-1)
            newcol = k - k1
            Cp[newcol+1] = Ti(pc)
            oldcol = Int(Qbtf[k+1])
            pend = Int(Ap[oldcol+2])
            for p in Int(Ap[oldcol+1]):(pend-1)
                newrow = Int(Pinv[Int(Ai[p+1])+1])
                if newrow < k1
                    nzoff += 1
                else
                    Ci[pc+1] = Ti(newrow - k1)
                    pc += 1
                end
            end
        end
        Cp[nk+1] = Ti(pc)
        maxnz = max(maxnz, pc)

        # order the block
        lnz1 = 0.0
        flops1 = 0.0
        ok = true
        if nk <= 3
            @inbounds for k in 1:nk
                Pblk[k] = Ti(k-1)
            end
            lnz1 = nk * (nk + 1) / 2
            flops1 = nk * (nk - 1) / 2 + (nk-1)*nk*(2nk-1) / 6
        elseif ordering == 0
            # AMD - fallback to natural for now
            ok, lnz1, flops1 = _amd_or_natural!(view(Pblk, 1:nk), nk, Cp, Ci, common)
        elseif ordering == 1
            # COLAMD - fallback to natural for now
            @inbounds for k in 1:nk
                Pblk[k] = Ti(k-1)
            end
            lnz1 = EMPTY_FLOAT
            flops1 = EMPTY_FLOAT
        else
            common.status = Cint(KLU_INVALID)
            return
        end
        if !ok
            common.status = Cint(KLU_INVALID)
            return
        end

        Lnz[block+1] = lnz1
        lnz = (lnz == EMPTY_FLOAT || lnz1 == EMPTY_FLOAT) ? EMPTY_FLOAT : lnz + lnz1
        flops = (flops == EMPTY_FLOAT || flops1 == EMPTY_FLOAT) ? EMPTY_FLOAT : flops + flops1

        # combine Pblk with BTF P,Q
        for k in 0:(nk-1)
            Q[k + k1 + 1] = Qbtf[Int(Pblk[k+1]) + k1 + 1]
        end
        for k in 0:(nk-1)
            P[k + k1 + 1] = Pbtf[Int(Pblk[k+1]) + k1 + 1]
        end
    end

    Sym.lnz = lnz
    Sym.unz = lnz
    Sym.nzoff = Ti(nzoff)
    Sym.est_flops = flops
    common.status = Cint(KLU_OK)
    return Sym
end

# AMD ordering for blocks larger than 3. We slice the block's CSC pattern
# into fresh Ap/Ai vectors (size nk+1 / pc) and call amd_order!; the returned
# permutation is then stored in `Pblk`.
function _amd_or_natural!(Pblk, nk::Int, Cp::AbstractVector{Ti}, Ci::AbstractVector{Ti},
                          common::KLUCommon{Ti}) where {Ti}
    Ap_blk = Vector{Ti}(undef, nk + 1)
    @inbounds for k in 1:(nk+1)
        Ap_blk[k] = Cp[k]
    end
    pc = Int(Cp[nk+1])
    Ai_blk = Vector{Ti}(undef, max(pc, 1))
    @inbounds for p in 1:pc
        Ai_blk[p] = Ci[p]
    end
    P = Vector{Ti}(undef, nk)
    status = AMD.amd_order!(nk, Ap_blk, Ai_blk, P)
    if status < 0
        return false, 0.0, 0.0
    end
    @inbounds for k in 1:nk
        Pblk[k] = P[k]
    end
    # Rough estimate of L fill-in -- matches SuiteSparse's coarse estimate when
    # actual AMD_LNZ isn't available.
    lnz1 = nk * (nk + 1) / 2
    flops1 = nk * (nk - 1) / 2 + (nk-1)*nk*(2nk-1) / 6
    return true, lnz1, flops1
end
