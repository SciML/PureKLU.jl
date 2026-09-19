"""
    rgrowth(K::KLUFactorization) -> Real

Reciprocal pivot growth of the factorization held by `K`, as
`min_j max_i|A_ij| / max_i|U_ij|` over the columns of the BTF diagonal
blocks, with the rows scaled exactly as the factorization scaled them.
`1×1` blocks are skipped, as is any column whose `U` is entirely zero.

A value near `1` means the pivots chosen for `K` are stable for the values
currently stored in it; a tiny value means they are not, and the
factorization should be redone with fresh pivots
(`klu_factor!`) rather than refactored in place ([`klu!`](@ref)). This is
the pivot-growth estimate SuiteSparse exposes as `klu_rgrowth`, and it is
the cheap test that makes `reuse_pivots = true` safe: it costs one pass
over the stored values of `A` and `U`.

The result is also written to `K.common.rgrowth`, matching SuiteSparse,
whenever the element type has a floating-point magnitude.

# Arguments
- `K`: a `KLUFactorization` that has already been factored.

# Returns
- The reciprocal pivot growth, `1` for a factorization with no non-singleton
  block and `0` when some block column has a zero `U` diagonal.

# Examples
```julia
julia > using PureKLU, SparseArrays

julia > F = klu(sparse([2.0 1.0; 1.0 3.0]));

julia > PureKLU.rgrowth(F) > 0.1
true
```
"""
function rgrowth(K::KLUFactorization{Tv, Ti, Tr}) where {Tv, Ti, Tr}
    _is_factored(K) || throw(ArgumentError("KLUFactorization has not been factored yet. Call `klu_factor!` first."))
    Sym = getfield(K, :symbolic)
    Num = getfield(K, :numeric)
    Ap, Ai, Ax = getfield(K, :colptr), getfield(K, :rowval), getfield(K, :nzval)
    Q, R = Sym.Q, Sym.R
    Pinv, Rs = Num.Pinv, Num.Rs
    # `Rs` is permuted into pivotal row order at the tail of the factorization,
    # so it is indexed by the new row, not the original one.
    scaled = !isempty(Rs)
    growth = one(Tr)
    @inbounds for block in 1:Int(Sym.nblocks)
        k1 = Int(R[block])
        nk = Int(R[block + 1]) - k1
        nk == 1 && continue
        blk = Num.LUbx[block]
        for j in 0:(nk - 1)
            oldcol = Int(Q[k1 + j + 1])
            max_a = zero(Tr)
            for p in (Int(Ap[oldcol + 1]) + 1):Int(Ap[oldcol + 2])
                newrow = Int(Pinv[Int(Ai[p]) + 1])
                newrow < k1 && continue   # entry outside the diagonal block
                a = abs(scaled ? Ax[p] / Rs[newrow + 1] : Ax[p])
                a > max_a && (max_a = a)
            end

            uip = Int(Num.Uip[k1 + j + 1])
            max_u = abs(Num.Udiag[k1 + j + 1])
            for p in 1:Int(Num.Ulen[k1 + j + 1])
                u = abs(blk.Ux[uip + p])
                u > max_u && (max_u = u)
            end

            iszero(max_u) && continue
            t = max_a / max_u
            t < growth && (growth = t)
        end
    end
    if growth isa AbstractFloat
        getfield(K, :common).rgrowth = Float64(growth)
    end
    return growth
end
