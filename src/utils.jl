"""
    map_tr(f, x, param)
Return an anonymous function that applies `f` to all elements of `x`, with a parameter `param` (which is set to `nothing` for function with no parameter).
"""
map_tr(f, x, p) = ((g, y; param) -> map(i -> g(i, y; param=param), 1:length(y)))(f, x, param=p)

"""
    lazy(funcs::Function...)
Generate methods extended to a vector instead of one of its components. A function `f` should have the following signature: `f(i::Int, x::V; param = nothing)`.
"""
function lazy(funcs::Function...)
    foreach(f -> eval(:($f(x; param=nothing) = map_tr($f, x, param))), map(Symbol, funcs))
end

"""
    lazy_param(funcs::Function...)
Generate methods extended to a vector instead of one of its components. A function `f` should have the following signature: `f(i::Int, x::V; param)`.
"""
function lazy_param(funcs::Function...)
    foreach(f -> eval(:($f(x; param) = map_tr($f, x, param))), map(Symbol, funcs))
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
    v
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

function incsert!(d::Dictionary, ind)
    set!(d, ind, isassigned(d, ind) ? d[ind] + 1 : 1)
end
