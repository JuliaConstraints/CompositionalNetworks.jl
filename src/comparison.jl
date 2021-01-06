# doc in transformation.jl
_co_identity(x) = identity(x)

"""
    _abs_diff_val_param(x::T, param::T)
Return the absolute difference between `x` and `param`.
"""
_co_abs_diff_val_param(x, param) = abs(x - param)

"""
    _val_minus_param(x::T, param::T)
    _param_minus_val(x::T, param::T)
Return the difference `x - param` (resp. `param - x`) if positive, `0.0` otherwise.
"""
_co_val_minus_param(x, param) = max(0.0, x - param)
_co_param_minus_val(x, param) = max(0.0, param - x)

"""
    _euclidian_param(x::T, param::T, dom_size::T2)
    _euclidian(x::T, dom_size::T2)
Compute an euclidian norm , possibly weigthed by `param`, on a scalar.
"""
_co_euclidian_param(x, param, d_size) = x == param ? 0.0 : (1.0 + abs(x - param) \ d_size)
_co_euclidian(x, d_size) = _co_euclidian_param(x, 0.0, d_size)

"""
    _abs_diff_val_vars(x::T, nvars::Int)
Return the absolute difference between `x` and the number of variables.
"""
_co_abs_diff_val_vars(x, nvars) = abs(x - nvars)

"""
    _val_minus_vars(x::T, nvars::Int)
    _vars_minus_val(x::T, nvars::Int)
Return the difference `x - nvars` (resp. `nvars - x`) if positive, `0.0` otherwise.
"""
_co_val_minus_vars(x, nvars) = _co_val_minus_param(x, nvars)
_co_vars_minus_val(x, nvars) = _co_param_minus_val(x, nvars)

"""
    comparison_layer(nvars, dom_size, param = nothing)
Generate the layer of transformations functions of the ICN. Iff `param` value is set, also includes all the parametric transformation with that value.
"""
function comparison_layer(nvars, dom_size, param = nothing)
    comparisons = LittleDict{Symbol, Function}(
        :identity => _co_identity,
        :euclidian => (x -> _co_euclidian(x, dom_size)),
        :abs_diff_val_vars => (x -> _co_abs_diff_val_vars(x, nvars)),
        :val_minus_vars => (x -> _co_val_minus_vars(x, nvars)),
        :vars_minus_val => (x -> _co_vars_minus_val(x, nvars)),
    )

    if !isnothing(param)
        comparisons_param = LittleDict{Symbol, Function}(
            :abs_diff_val_param => (x -> _co_abs_diff_val_param(x, param)),
            :val_minus_param => (x -> _co_val_minus_param(x, param)),
            :param_minus_val => (x -> _co_param_minus_val(x, param)),
            :euclidian_param => (x -> _co_euclidian_param(x, param, dom_size)),
        )
        comparisons = LittleDict{Symbol, Function}(union(comparisons, comparisons_param))
    end

    return Layer(comparisons, true)
end
