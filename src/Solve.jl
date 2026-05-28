# All multiply-subtract loops in this file dispatch on `Val(common.use_fma)`
# via `_mulsub` (defined in `Kernel.jl`). The default `Val(true)` emits a
# fused multiply-add on x86 with FMA. To match SuiteSparse `libklu.so`
# bit-for-bit, set `common.use_fma = false` (or pass `use_fma=false` to
# `klu`/`klu_factor!`) -- see `Kernel.jl` for the full discussion.

"""
    klu_solve!(Sym, Num, B, common; conj_solve=false) -> B

Solve `A * X = B` in place, where `A = P' * R * L * U * Q'` (i.e. the
matrix originally analysed and factored). `B` is a vector or matrix.
Mirrors `klu_solve.c`.
"""
function klu_solve!(Sym::KLUSymbolic{Ti}, Num::KLUNumeric{Tv, Ti, Tr},
                    B::AbstractVecOrMat{Tv}, common::KLUCommon{Ti}) where {Tv, Ti, Tr}
    return _klu_solve_impl!(Sym, Num, B, common, common.use_fma)
end

function _klu_solve_impl!(Sym::KLUSymbolic{Ti}, Num::KLUNumeric{Tv, Ti, Tr},
                          B::AbstractVecOrMat{Tv}, common::KLUCommon{Ti},
                          fma_val::Val) where {Tv, Ti, Tr}
    common.status = Cint(KLU_OK)
    n = Int(Sym.n)
    size(B, 1) == n || throw(DimensionMismatch())
    nrhs = size(B, 2)

    Q = Sym.Q
    R = Sym.R
    nblocks = Int(Sym.nblocks)
    Pnum = Num.Pnum
    Offp = Num.Offp
    Offi = Num.Offi
    Offx = Num.Offx
    Lip = Num.Lip
    Llen = Num.Llen
    Uip = Num.Uip
    Ulen = Num.Ulen
    Udiag = Num.Udiag
    Rs = Num.Rs
    scaled = !isempty(Rs)

    # `Num.Xwork` is preallocated to length `n` at factor time, so
    # `klu_solve!` is heap-allocation-free on subsequent calls. Every
    # element is overwritten before being read (`X[k+1] = B[...]`), so
    # we don't need to clear it.
    X = Num.Xwork
    for col in 1:nrhs
        # X = P * (R \ B[:, col]) -- gather permuted+scaled RHS
        if scaled
            @inbounds for k in 0:(n-1)
                X[k+1] = B[Int(Pnum[k+1])+1, col] / Rs[k+1]
            end
        else
            @inbounds for k in 0:(n-1)
                X[k+1] = B[Int(Pnum[k+1])+1, col]
            end
        end

        # Solve block-by-block, back to front
        for block in nblocks:-1:1
            k1 = Int(R[block])
            k2 = Int(R[block+1])
            nk = k2 - k1
            if nk == 1
                X[k1+1] = _cdiv(X[k1+1], Udiag[k1+1], fma_val)
            else
                _klu_lsolve!(nk, view(Lip, k1+1:k2), view(Llen, k1+1:k2),
                             Num.LUbx[block], view(X, k1+1:k2), fma_val)
                _klu_usolve!(nk, view(Uip, k1+1:k2), view(Ulen, k1+1:k2),
                             Num.LUbx[block], view(Udiag, k1+1:k2),
                             view(X, k1+1:k2), fma_val)
            end
            if block > 1
                @inbounds for k in k1:(k2-1)
                    xk = X[k+1]
                    pstart = Int(Offp[k+1])
                    pend   = Int(Offp[k+2]) - 1
                    # Offi entries are distinct per column of F (sparse CSC
                    # with no duplicate rows), so the scatter is alias-free.
                    @simd ivdep for p in pstart:pend
                        X[Int(Offi[p+1])+1] = _mulsub(X[Int(Offi[p+1])+1],
                                                     Offx[p+1], xk, fma_val)
                    end
                end
            end
        end

        # Scatter X back to B with Q permutation: B[Q[k]+1] = X[k+1]
        @inbounds for k in 0:(n-1)
            B[Int(Q[k+1])+1, col] = X[k+1]
        end
    end
    return B
end

function _klu_lsolve!(n::Int, Lip::AbstractVector{Ti}, Llen::AbstractVector{Ti},
                      block::KLUNumericBlock{Tv, Ti}, X::AbstractVector{Tv},
                      fma_val::Val) where {Tv, Ti}
    @inbounds for k in 0:(n-1)
        xk = X[k+1]
        lip = Int(Lip[k+1])
        len = Int(Llen[k+1])
        # Row indices block.Li[lip+1..lip+len] are pairwise distinct per
        # column (no duplicates in sparse LU storage), so `ivdep` is safe.
        @simd ivdep for p in 0:(len-1)
            i = Int(block.Li[lip+p+1])
            X[i+1] = _mulsub(X[i+1], block.Lx[lip+p+1], xk, fma_val)
        end
    end
end

function _klu_usolve!(n::Int, Uip::AbstractVector{Ti}, Ulen::AbstractVector{Ti},
                      block::KLUNumericBlock{Tv, Ti}, Udiag::AbstractVector{Tv},
                      X::AbstractVector{Tv}, fma_val::Val) where {Tv, Ti}
    @inbounds for k in (n-1):-1:0
        uip = Int(Uip[k+1])
        len = Int(Ulen[k+1])
        xk = _cdiv(X[k+1], Udiag[k+1], fma_val)
        X[k+1] = xk
        # Same as in _klu_lsolve!: distinct row indices, safe to vectorise.
        @simd ivdep for p in 0:(len-1)
            i = Int(block.Ui[uip+p+1])
            X[i+1] = _mulsub(X[i+1], block.Ux[uip+p+1], xk, fma_val)
        end
    end
end

"""
    klu_tsolve!(Sym, Num, B, common; conj_solve=false) -> B

Solve `A' * X = B` (or `A^H X = B` if `conj_solve=true` and `Tv` is
complex) in place. Mirrors `klu_tsolve.c`.
"""
function klu_tsolve!(Sym::KLUSymbolic{Ti}, Num::KLUNumeric{Tv, Ti, Tr},
                     B::AbstractVecOrMat{Tv}, common::KLUCommon{Ti};
                     conj_solve::Bool=false) where {Tv, Ti, Tr}
    return _klu_tsolve_impl!(Sym, Num, B, common, common.use_fma;
                             conj_solve)
end

function _klu_tsolve_impl!(Sym::KLUSymbolic{Ti}, Num::KLUNumeric{Tv, Ti, Tr},
                           B::AbstractVecOrMat{Tv}, common::KLUCommon{Ti},
                           fma_val::Val;
                           conj_solve::Bool=false) where {Tv, Ti, Tr}
    common.status = Cint(KLU_OK)
    n = Int(Sym.n)
    size(B, 1) == n || throw(DimensionMismatch())
    nrhs = size(B, 2)

    Q = Sym.Q
    R = Sym.R
    nblocks = Int(Sym.nblocks)
    Pnum = Num.Pnum
    Offp = Num.Offp
    Offi = Num.Offi
    Offx = Num.Offx
    Lip = Num.Lip
    Llen = Num.Llen
    Uip = Num.Uip
    Ulen = Num.Ulen
    Udiag = Num.Udiag
    Rs = Num.Rs
    scaled = !isempty(Rs)

    # See `_klu_solve_impl!`: reuse the per-numeric solve workspace
    # instead of allocating a fresh `Vector{Tv}(undef, n)` each call.
    X = Num.Xwork
    for col in 1:nrhs
        @inbounds for k in 0:(n-1)
            X[k+1] = B[Int(Q[k+1])+1, col]
        end

        for block in 1:nblocks
            k1 = Int(R[block])
            k2 = Int(R[block+1])
            nk = k2 - k1

            if block > 1
                @inbounds for k in k1:(k2-1)
                    pstart = Int(Offp[k+1])
                    pend   = Int(Offp[k+2]) - 1
                    acc = X[k+1]
                    # Reduction over `acc`.  `@simd` lets the compiler split
                    # this into multiple partial accumulators and recombine,
                    # breaking the serial FMA dependency chain that otherwise
                    # makes tsolve slower with `use_fma=Val(true)`.
                    @simd for p in pstart:pend
                        offik = Offx[p+1]
                        if conj_solve
                            offik = conj(offik)
                        end
                        acc = _mulsub(acc, offik, X[Int(Offi[p+1])+1], fma_val)
                    end
                    X[k+1] = acc
                end
            end

            if nk == 1
                s = Udiag[k1+1]
                if conj_solve
                    s = conj(s)
                end
                X[k1+1] = _cdiv(X[k1+1], s, fma_val)
            else
                _klu_utsolve!(nk, view(Uip, k1+1:k2), view(Ulen, k1+1:k2),
                              Num.LUbx[block], view(Udiag, k1+1:k2),
                              view(X, k1+1:k2), conj_solve, fma_val)
                _klu_ltsolve!(nk, view(Lip, k1+1:k2), view(Llen, k1+1:k2),
                              Num.LUbx[block], view(X, k1+1:k2), conj_solve,
                              fma_val)
            end
        end

        if scaled
            @inbounds for k in 0:(n-1)
                B[Int(Pnum[k+1])+1, col] = X[k+1] / Rs[k+1]
            end
        else
            @inbounds for k in 0:(n-1)
                B[Int(Pnum[k+1])+1, col] = X[k+1]
            end
        end
    end
    return B
end

function _klu_ltsolve!(n::Int, Lip::AbstractVector{Ti}, Llen::AbstractVector{Ti},
                       block::KLUNumericBlock{Tv, Ti}, X::AbstractVector{Tv},
                       conj_solve::Bool, fma_val::Val) where {Tv, Ti}
    @inbounds for k in (n-1):-1:0
        lip = Int(Lip[k+1])
        len = Int(Llen[k+1])
        acc = X[k+1]
        # Inner reduction into `acc`.  `@simd` permits reassociation so the
        # compiler can vectorise (multiple partial accumulators) and break
        # the serial FMA dependency.  See SciML/PureKLU.jl#1 for context.
        @simd for p in 0:(len-1)
            lik = block.Lx[lip+p+1]
            if conj_solve
                lik = conj(lik)
            end
            acc = _mulsub(acc, lik, X[Int(block.Li[lip+p+1])+1], fma_val)
        end
        X[k+1] = acc
    end
end

function _klu_utsolve!(n::Int, Uip::AbstractVector{Ti}, Ulen::AbstractVector{Ti},
                       block::KLUNumericBlock{Tv, Ti}, Udiag::AbstractVector{Tv},
                       X::AbstractVector{Tv}, conj_solve::Bool,
                       fma_val::Val) where {Tv, Ti}
    @inbounds for k in 0:(n-1)
        uip = Int(Uip[k+1])
        len = Int(Ulen[k+1])
        acc = X[k+1]
        @simd for p in 0:(len-1)
            uik = block.Ux[uip+p+1]
            if conj_solve
                uik = conj(uik)
            end
            acc = _mulsub(acc, uik, X[Int(block.Ui[uip+p+1])+1], fma_val)
        end
        ukk = Udiag[k+1]
        if conj_solve
            ukk = conj(ukk)
        end
        X[k+1] = _cdiv(acc, ukk, fma_val)
    end
end

"""
    klu_sort!(Sym, Num) -> Num

Sort the row indices within each column of `L` and `U` to strictly
increasing order, in place. Mirrors `klu_sort.c`. KLU.jl invokes its C
counterpart whenever the user accesses `K.L` / `K.U` (via `klu_extract`);
calling this routine ensures `K.LUbx[block]` matches the post-sort state
that KLU.jl leaves behind, which makes subsequent refactors bit-for-bit
equivalent.
"""
function klu_sort!(Sym::KLUSymbolic{Ti}, Num::KLUNumeric{Tv, Ti, Tr}) where {Tv, Ti, Tr}
    nblocks = Int(Sym.nblocks)
    for block in 1:nblocks
        k1 = Int(Sym.R[block]); k2 = Int(Sym.R[block+1])
        nk = k2 - k1
        nk > 1 || continue
        bk = Num.LUbx[block]
        Lip_b = view(Num.Lip, k1+1:k2)
        Llen_b = view(Num.Llen, k1+1:k2)
        Uip_b = view(Num.Uip, k1+1:k2)
        Ulen_b = view(Num.Ulen, k1+1:k2)
        for kk in 0:(nk-1)
            lip = Int(Lip_b[kk+1]); llen = Int(Llen_b[kk+1])
            _sort_internal_col!(bk.Li, bk.Lx, lip+1, lip+llen)
            uip = Int(Uip_b[kk+1]); ulen = Int(Ulen_b[kk+1])
            _sort_internal_col!(bk.Ui, bk.Ux, uip+1, uip+ulen)
        end
    end
    return Num
end

function _sort_internal_col!(I::AbstractVector, X::AbstractVector, lo::Int, hi::Int)
    lo >= hi && return nothing
    perm = sortperm(view(I, lo:hi))
    Iv = I[lo:hi][perm]
    Xv = X[lo:hi][perm]
    @inbounds for i in 1:length(perm)
        I[lo + i - 1] = Iv[i]
        X[lo + i - 1] = Xv[i]
    end
    return nothing
end

"""
    klu_refactor!(Sym, Num, Ap, Ai, Ax, common; allowsingular=false) -> Num

Re-factor a matrix with the same nonzero pattern as the one used to
build `Num`. No pivoting is performed; the sparsity pattern and pivot
order of L and U are reused. Mirrors `klu_refactor.c`.
"""
function klu_refactor!(Sym::KLUSymbolic{Ti}, Num::KLUNumeric{Tv, Ti, Tr},
                       Ap::Vector{Ti}, Ai::Vector{Ti}, Ax::Vector{Tv},
                       common::KLUCommon{Ti};
                       allowsingular::Bool=false) where {Tv, Ti, Tr}
    scale = Int(common.scale)
    # Function-barrier on `scale_val` so the per-nonzero scale path is
    # specialised per branch (no union-typed `scale_val` flowing through
    # the inner loop).
    if scale > 0
        return _klu_refactor_impl!(Sym, Num, Ap, Ai, Ax, common, common.use_fma,
                                   Val(true); allowsingular)
    else
        return _klu_refactor_impl!(Sym, Num, Ap, Ai, Ax, common, common.use_fma,
                                   Val(false); allowsingular)
    end
end

function _klu_refactor_impl!(Sym::KLUSymbolic{Ti}, Num::KLUNumeric{Tv, Ti, Tr},
                             Ap::Vector{Ti}, Ai::Vector{Ti}, Ax::Vector{Tv},
                             common::KLUCommon{Ti}, fma_val::Val,
                             scale_val::Val;
                             allowsingular::Bool=false) where {Tv, Ti, Tr}
    common.status = Cint(KLU_OK)
    common.numerical_rank = Ti(EMPTY)
    common.singular_col = Ti(EMPTY)

    n = Int(Sym.n)
    Q = Sym.Q
    R = Sym.R
    nblocks = Int(Sym.nblocks)
    maxblock = Int(Sym.maxblock)
    Pnum = Num.Pnum
    Offx = Num.Offx
    Lip = Num.Lip; Llen = Num.Llen
    Uip = Num.Uip; Ulen = Num.Ulen
    Udiag = Num.Udiag
    Pinv = Num.Pinv

    scale = Int(common.scale)
    if scale > 0
        if isempty(Num.Rs)
            Num.Rs = zeros(Tr, n)
        end
        if isempty(Num.Xtmp)
            Num.Xtmp = Vector{Tr}(undef, n)
        end
    elseif !isempty(Num.Rs)
        # Drop stale row-scaling state when user turns scaling off.
        Num.Rs = Tr[]
    end
    Rs = Num.Rs

    if scale > 0
        if !klu_scale!(scale, n, Ap, Ai, Ax, Rs, nothing, common)
            return Num
        end
    end
    # scale == 0: pattern already validated at factor time; skip walk.
    # scale < 0: caller opted out of scaling and validation.

    # Refactor only touches the first `maxblock` entries of `X`, and
    # the algorithm restores each column's scratch positions to zero
    # before moving on -- so we only need to ensure the working prefix
    # starts at zero on entry, and the workspace can be shared with
    # `klu_solve!` via `Num.Xwork` (length `n >= maxblock`).
    X = Num.Xwork
    @inbounds for k in 1:maxblock
        X[k] = zero(Tv)
    end
    poff = 0
    nzoff = Int(Sym.nzoff)

    for block in 1:nblocks
        k1 = Int(R[block])
        k2 = Int(R[block+1])
        nk = k2 - k1

        if nk == 1
            @inbounds oldcol = Int(Q[k1+1])
            @inbounds pend = Int(Ap[oldcol+2])
            s = zero(Tv)
            @inbounds for p in Int(Ap[oldcol+1]):(pend-1)
                oldrow = Int(Ai[p+1])
                newrow = Int(Pinv[oldrow+1]) - k1
                aik = _scale_aik(Ax[p+1],
                                 _rs_at(Rs, oldrow+1, scale_val),
                                 scale_val)
                if newrow < 0 && poff < nzoff
                    Offx[poff+1] = aik
                    poff += 1
                else
                    s = aik
                end
            end
            @inbounds Udiag[k1+1] = s
            if iszero(s)
                common.status = Cint(KLU_SINGULAR)
                if common.numerical_rank == Ti(EMPTY)
                    common.numerical_rank = Ti(k1)
                    common.singular_col = Ti(oldcol)
                end
                if common.halt_if_singular != 0
                    return Num
                end
            end
        else
            bk = Num.LUbx[block]
            Lip_b = view(Lip, k1+1:k2)
            Llen_b = view(Llen, k1+1:k2)
            Uip_b = view(Uip, k1+1:k2)
            Ulen_b = view(Ulen, k1+1:k2)
            @inbounds for k in 0:(nk-1)
                oldcol = Int(Q[k+k1+1])
                pend = Int(Ap[oldcol+2])
                @inbounds for p in Int(Ap[oldcol+1]):(pend-1)
                    oldrow = Int(Ai[p+1])
                    newrow = Int(Pinv[oldrow+1]) - k1
                    aik = _scale_aik(Ax[p+1],
                                     _rs_at(Rs, oldrow+1, scale_val),
                                     scale_val)
                    if newrow < 0 && poff < nzoff
                        Offx[poff+1] = aik
                        poff += 1
                    else
                        X[newrow+1] = aik
                    end
                end

                uip = Int(Uip_b[k+1])
                ulen = Int(Ulen_b[k+1])
                @inbounds for up in 0:(ulen-1)
                    j = Int(bk.Ui[uip+up+1])
                    ujk = X[j+1]
                    X[j+1] = zero(Tv)
                    bk.Ux[uip+up+1] = ujk
                    lip_j = Int(Lip_b[j+1])
                    llen_j = Int(Llen_b[j+1])
                    # Same scatter pattern as the factor / forward solve:
                    # distinct row indices per column, alias-free.
                    @inbounds @simd ivdep for p in 0:(llen_j-1)
                        ii = Int(bk.Li[lip_j+p+1])
                        X[ii+1] = _mulsub(X[ii+1], bk.Lx[lip_j+p+1], ujk, fma_val)
                    end
                end
                ukk = X[k+1]
                X[k+1] = zero(Tv)
                if iszero(ukk)
                    common.status = Cint(KLU_SINGULAR)
                    if common.numerical_rank == Ti(EMPTY)
                        common.numerical_rank = Ti(k + k1)
                        common.singular_col = Ti(oldcol)
                    end
                    if common.halt_if_singular != 0
                        return Num
                    end
                end
                Udiag[k+k1+1] = ukk
                lip_k = Int(Lip_b[k+1])
                llen_k = Int(Llen_b[k+1])
                # Row indices `bk.Li[lip_k+1..lip_k+llen_k]` are pairwise
                # distinct (sparse LU column pattern), so the gather/scatter
                # on `X[i+1]` is alias-free; writes to `bk.Lx[lip_k+p+1]`
                # are to consecutive slots.
                @inbounds @simd ivdep for p in 0:(llen_k-1)
                    i = Int(bk.Li[lip_k+p+1])
                    bk.Lx[lip_k+p+1] = _cdiv(X[i+1], ukk, fma_val)
                    X[i+1] = zero(Tv)
                end
            end
        end
    end

    if scale > 0
        # `Num.Xtmp` is preallocated to length `n` at factor time when
        # scale > 0; reuse it for the Rs permutation shuffle.
        Xt = Num.Xtmp
        @inbounds for k in 1:n
            Xt[k] = Rs[Int(Pnum[k])+1]
        end
        @inbounds for k in 1:n
            Rs[k] = Xt[k]
        end
    end

    return Num
end
