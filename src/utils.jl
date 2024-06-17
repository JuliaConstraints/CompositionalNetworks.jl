"""
    map_tr!(f, x, X, param)
Return an anonymous function that applies `f` to all elements of `x` and store the result in `X`, with a parameter `param` (which is set to `nothing` for function with no parameter).
"""
function map_tr!(f, x, X; p...)
    return ((g, y, Y; params...) -> map!(i -> g(i, y; params...), Y, 1:length(y)))(
        f,
        x,
        X;
        p...,
    )
end
# function map_tr!(f, x, X)
#     return ((g, y, Y; param) -> map!(i -> g(i, y), Y, 1:length(y)))(
#         f,
#         x,
#         X;
#         param=nothing,
#     )
# end

"""
    lazy(funcs::Function...)
Generate methods extended to a vector instead of one of its components. A function `f` should have the following signature: `f(i::Int, x::V)`.
"""
function lazy(funcs::Function...)
    for f in Iterators.map(Symbol, funcs)
        eval(:($f(x::V, X; params...) where {V<:AbstractVector} = map_tr!($f, x, X; params...)))
        eval(:($f(x; params...) = $f(x, similar(x); params...)))
    end
    return nothing
end

"""
    lazy_param(funcs::Function...)
Generate methods extended to a vector instead of one of its components. A function `f` should have the following signature: `f(i::Int, x::V; param)`.
"""
function lazy_param(funcs::Function...)
    for f in Iterators.map(Symbol, funcs)
        eval(:($f(x::V, X; params...) where {V<:AbstractVector} = map_tr!($f, x, X; params...)))
        eval(:($f(x; params...) = $f(x, similar(x); params...)))
    end
    return nothing
end

"""
    as_bitvector(n::Int, max_n::Int = n)
Convert an Int to a BitVector of minimal size (relatively to `max_n`).
"""
function as_bitvector(n::Int, max_n::Int=n)
    nm1 = n - 1
    v = falses(ceil(Int, log2(max_n)))
    i = 0
    @inbounds while !iszero(nm1)
        tz = trailing_zeros(nm1)
        i += (tz + 1)
        v[i] = true
        nm1 >>>= (tz + 1)
    end
    return v
end

"""
    as_int(v::AbstractVector)
Convert a `BitVector` into an `Int`.
"""
function as_int(v::AbstractVector)
    n = 0
    for (i, b) in enumerate(v)
        n += b ? 2^(i - 1) : 0
    end
    return n
end

"""
    reduce_symbols(symbols, sep)
Produce a formatted string that separates the symbols by `sep`. Used internally for `show_composition`.
"""
function reduce_symbols(symbols, sep, parenthesis=true; prefix="")
    str = reduce((x, y) -> "$y$sep$x", map(s -> "$prefix$s", symbols))
    return parenthesis ? "[$str]" : str
end

"""
    tr_in(tr, X, x, param)

Application of an operation from the transformation layer. Used to generate more efficient code for all compositions.
"""
@unroll function tr_in(tr, X, x; params...)
    @unroll for i = 1:length(tr)
        tr[i](x, @view(X[:, i]); params...)
    end
end
