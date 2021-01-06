"""
    _map_tr(f, x)
    _map_tr(f, x, param)
Return an anonymous function that applies `f` to all elements of `x`, with an optional parameter `param`.
"""
_map_tr(f, x) = ((g, y) -> map(i -> g(i, y), 1:length(y)))(f, x)
_map_tr(f, x, param) = ((g, y, p) -> map(i -> g(i, y, p), 1:length(y)))(f, x, param)

"""
    lazy(funcs::Function...)
    lazy_param(funcs::Function...)
Generate methods extended to a vector instead of one of its components. For `lazy` (resp. `lazy_param`) a function `f` should have the following signature: `f(i::Int, x::V)` (resp. `f(i::Int, x::V, param::T)`).
"""
function lazy(funcs::Function...)
    foreach(f -> eval(:($f(x) = (y -> _map_tr($f, y))(x))), map(Symbol, funcs))
end
function lazy_param(funcs::Function...)
    foreach(f -> eval(:($f(x, param) = (y -> _map_tr($f, y, param))(x))), map(Symbol, funcs))
end

"""
    _as_bitvector(n::Int, max_n::Int = n)
Convert an Int to a BitVector of minimal size (relatively to `max_n`).
"""
function _as_bitvector(n::Int, max_n::Int = n)
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

# TODO: memory layout stable bitvector for the general individual
# function as_bitvector(n::Int)
#     v = falses(8*sizeof(n))
#     i = 0
#     @inbounds while !iszero(n)
#         tz = trailing_zeros(n)
#         i += (tz + 1)
#         v[i] = true
#         n >>>= (tz + 1)
#     end
#     v
# end

"""
    _as_int(v::AbstractVector)
Convert a `BitVector` into an `Int`.
"""
function _as_int(v::AbstractVector)
    n = 0
    for (i, b) in enumerate(v)
        n += b ? 2^(i - 1) : 0
    end
    return n
end

"""
    _reduce_symbols(symbols, sep)
Produce a formatted string that separates the symbols by `sep`. Used internally for `show_composition`.
"""
function _reduce_symbols(symbols, sep, parenthesis = true)
    str = reduce((x,y) -> "$y$sep$x", symbols)
    return parenthesis ? "[$str]" : str
end
