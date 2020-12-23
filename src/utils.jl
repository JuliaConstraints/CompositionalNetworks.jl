
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
