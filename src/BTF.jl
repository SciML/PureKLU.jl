"""
    BTF

Pure-Julia port of SuiteSparse's BTF (Block Triangular Form) routines:
maximum transversal (`btf_maxtrans`), strongly connected components
(`btf_strongcomp`) and the combined `btf_order`.

All arrays are 0-based to mirror the C implementation; callers from the
KLU driver use 0-based indices throughout.
"""
module BTF

const EMPTY = -1

@inline btf_flip(j) = -j - 2
@inline btf_isflipped(j) = j < -1
@inline btf_unflip(j) = btf_isflipped(j) ? btf_flip(j) : j

"""
    maxtrans!(nrow, ncol, Ap, Ai, maxwork, Match, Work) -> (nmatch, work)

Find a column permutation maximising the number of zero-free diagonal
entries. `Match` (size `nrow`) is filled in-place: `Match[i] = j` if column
`j` is matched with row `i`, `EMPTY` if unmatched. `Work` is workspace
of length `5*ncol`.
"""
function maxtrans!(
        nrow::Int, ncol::Int,
        Ap::Vector{Ti}, Ai::Vector{Ti},
        maxwork::Float64,
        Match::Vector{Ti},
        Work::Vector{Ti}
    ) where {Ti <: Integer}
    Cheap = view(Work, 1:ncol)
    Flag = view(Work, (ncol + 1):2ncol)
    Istack = view(Work, (2ncol + 1):3ncol)
    Jstack = view(Work, (3ncol + 1):4ncol)
    Pstack = view(Work, (4ncol + 1):5ncol)

    @inbounds for j in 1:ncol
        Cheap[j] = Ap[j]
        Flag[j] = Ti(EMPTY)
    end
    @inbounds for i in 1:nrow
        Match[i] = Ti(EMPTY)
    end

    work_done = Ref(0.0)
    mw = maxwork > 0 ? maxwork * Ap[ncol + 1] : 0.0

    nmatch = 0
    work_limit_reached = false
    for k in 0:(ncol - 1)
        result = augment!(
            k, Ap, Ai, Match, Cheap, Flag, Istack, Jstack, Pstack,
            work_done, mw
        )
        if result == 1
            nmatch += 1
        elseif result == EMPTY
            work_limit_reached = true
        end
    end

    if work_limit_reached
        work_done[] = Float64(EMPTY)
    end
    return nmatch, work_done[]
end

"""
    augment!(k, Ap, Ai, Match, Cheap, Flag, Istack, Jstack, Pstack, work, maxwork)

Non-recursive depth-first augmenting-path search starting at column `k`
(0-based). Returns 1 on success, 0 on failure, and `EMPTY` if a work
budget was exhausted. Mirrors `btf_maxtrans.c::augment`.
"""
function augment!(
        k::Int, Ap, Ai, Match, Cheap, Flag, Istack, Jstack, Pstack,
        work::Ref{Float64}, maxwork::Float64
    )
    Ti = eltype(Match)
    quick = maxwork > 0
    found = false
    i = Ti(EMPTY)
    head = 0
    Jstack[1] = Ti(k)

    p = Ti(0)
    pend = Ti(0)
    j = Ti(0)
    while head >= 0
        j = Jstack[head + 1]
        pend = Ap[j + 2]

        if Flag[j + 1] != Ti(k)
            Flag[j + 1] = Ti(k)
            p = Cheap[j + 1]
            while p < pend && !found
                i = Ai[p + 1]
                found = Match[i + 1] == Ti(EMPTY)
                p += 1
            end
            Cheap[j + 1] = p

            if found
                Istack[head + 1] = i
                break
            end
            Pstack[head + 1] = Ap[j + 1]
        end

        if quick && work[] > maxwork
            return EMPTY
        end

        pstart = Pstack[head + 1]
        p = pstart
        broke = false
        j2 = Ti(0)
        while p < pend
            i = Ai[p + 1]
            j2 = Match[i + 1]
            if Flag[j2 + 1] != Ti(k)
                Pstack[head + 1] = p + 1
                Istack[head + 1] = i
                head += 1
                Jstack[head + 1] = j2
                broke = true
                break
            end
            p += 1
        end

        work[] += (p - pstart + 1)

        if !broke && p == pend
            head -= 1
        end
    end

    if found
        for hp in head:-1:0
            jj = Jstack[hp + 1]
            ii = Istack[hp + 1]
            Match[ii + 1] = jj
        end
        return 1
    end
    return 0
end

const UNVISITED = -2
const UNASSIGNED = -1

"""
    strongcomp!(n, Ap, Ai, Q, P, R, Work) -> nblocks

Find strongly connected components of the directed graph encoded by the
sparse matrix (`Ap`, `Ai`). `Q` (size `n`) is the optional input column
permutation (potentially with flipped entries from `maxtrans!`); pass
`nothing` for the identity. `P` (size `n`) and `R` (size `n+1`) receive
the symmetric row/col permutation and block boundaries respectively.
`Work` is workspace of length `4n`. Mirrors `btf_strongcomp.c`.
"""
function strongcomp!(
        n::Int, Ap::Vector{Ti}, Ai::Vector{Ti},
        Q::Union{Vector{Ti}, Nothing},
        P::Vector{Ti}, R::Vector{Ti}, Work::Vector{Ti}
    ) where {Ti <: Integer}
    Time = view(Work, 1:n)
    Flag = view(Work, (n + 1):2n)
    Low = P
    Cstack = R
    Jstack = view(Work, (2n + 1):3n)
    Pstack = view(Work, (3n + 1):4n)

    @inbounds for j in 1:n
        Flag[j] = Ti(UNVISITED)
        Low[j] = Ti(EMPTY)
        Time[j] = Ti(EMPTY)
    end

    timestamp = 0
    nblocks = 0
    for j0 in 0:(n - 1)
        if Flag[j0 + 1] == Ti(UNVISITED)
            timestamp, nblocks = scc_dfs!(
                j0, Ap, Ai, Q, Time, Flag, Low,
                nblocks, timestamp,
                Cstack, Jstack, Pstack
            )
        end
    end

    @inbounds for b in 1:nblocks
        R[b] = 0
    end
    @inbounds for j in 1:n
        R[Flag[j] + 1] += 1
    end
    Time[1] = 0
    @inbounds for b in 2:nblocks
        Time[b] = Time[b - 1] + R[b - 1]
    end
    @inbounds for b in 1:nblocks
        R[b] = Time[b]
    end
    R[nblocks + 1] = n

    @inbounds for j0 in 0:(n - 1)
        bidx = Flag[j0 + 1] + 1
        P[Time[bidx] + 1] = Ti(j0)
        Time[bidx] += 1
    end

    if Q !== nothing
        @inbounds for k in 1:n
            Time[k] = Q[P[k] + 1]
        end
        @inbounds for k in 1:n
            Q[k] = Time[k]
        end
    end

    return nblocks
end

"""
    scc_dfs!(j_start, Ap, Ai, Q, Time, Flag, Low, nblocks, timestamp,
             Cstack, Jstack, Pstack) -> (timestamp, nblocks)

Non-recursive Tarjan DFS rooted at `j_start` (0-based). Mirrors
`btf_strongcomp.c::dfs`.
"""
function scc_dfs!(
        j_start::Int, Ap, Ai, Q::Union{AbstractVector, Nothing},
        Time, Flag, Low, nblocks, timestamp,
        Cstack, Jstack, Pstack
    )
    Ti = eltype(Flag)
    chead = 0
    jhead = 0
    Jstack[1] = Ti(j_start)

    j = Ti(0)
    while jhead >= 0
        j = Jstack[jhead + 1]
        jj = Q === nothing ? Int(j) : Int(btf_unflip(Q[j + 1]))
        pend = Ap[jj + 2]

        if Flag[j + 1] == Ti(UNVISITED)
            chead += 1
            Cstack[chead + 1] = j
            timestamp += 1
            Time[j + 1] = Ti(timestamp)
            Low[j + 1] = Ti(timestamp)
            Flag[j + 1] = Ti(UNASSIGNED)
            Pstack[jhead + 1] = Ap[jj + 1]
        end

        p = Pstack[jhead + 1]
        broke = false
        while p < pend
            i = Ai[p + 1]
            if Flag[i + 1] == Ti(UNVISITED)
                Pstack[jhead + 1] = p + 1
                jhead += 1
                Jstack[jhead + 1] = i
                broke = true
                break
            elseif Flag[i + 1] == Ti(UNASSIGNED)
                if Time[i + 1] < Low[j + 1]
                    Low[j + 1] = Time[i + 1]
                end
            end
            p += 1
        end

        if !broke && p == pend
            jhead -= 1

            if Low[j + 1] == Time[j + 1]
                while true
                    ii = Cstack[chead + 1]
                    chead -= 1
                    Flag[ii + 1] = Ti(nblocks)
                    if ii == j
                        break
                    end
                end
                nblocks += 1
            end
            if jhead >= 0
                parent = Jstack[jhead + 1]
                if Low[j + 1] < Low[parent + 1]
                    Low[parent + 1] = Low[j + 1]
                end
            end
        end
    end
    return timestamp, nblocks
end

"""
    order!(n, Ap, Ai, maxwork, P, Q, R, Work) -> (nblocks, nmatch, work)

Compute row permutation `P`, column permutation `Q` (possibly with flipped
entries when structurally singular) and block boundaries `R` such that
`A(P,Q)` is in upper block triangular form. `Work` must have length
`5n`. Mirrors `btf_order.c`.
"""
function order!(
        n::Int, Ap::Vector{Ti}, Ai::Vector{Ti},
        maxwork::Float64,
        P::Vector{Ti}, Q::Vector{Ti}, R::Vector{Ti},
        Work::Vector{Ti}
    ) where {Ti <: Integer}
    nmatch, work = maxtrans!(n, n, Ap, Ai, maxwork, Q, Work)

    if nmatch < n
        Flag = view(Work, (n + 1):2n)
        @inbounds for j in 1:n
            Flag[j] = 0
        end
        @inbounds for i in 1:n
            j = Q[i]
            if j != Ti(EMPTY)
                Flag[j + 1] = 1
            end
        end
        nbadcol = 0
        @inbounds for j in (n - 1):-1:0
            if Flag[j + 1] == 0
                nbadcol += 1
                Work[nbadcol] = Ti(j)
            end
        end
        @inbounds for i in 1:n
            if Q[i] == Ti(EMPTY) && nbadcol > 0
                jbad = Work[nbadcol]
                nbadcol -= 1
                Q[i] = Ti(btf_flip(Int(jbad)))
            end
        end
    end

    nblocks = strongcomp!(n, Ap, Ai, Q, P, R, Work)
    return nblocks, nmatch, work
end

end # module
