"""
    _co_identity(x::Number)
Identity function. Already defined in Julia as `identity`, specialized for scalars in the `comparison` layer.
"""
_co_identity(x; param=nothing, dom_size=0, nvars=0) = identity(x)

"""
    _co_abs_diff_val_param(x, param)
Return the absolute difference between `x` and `param`.
"""
_co_abs_diff_val_param(x; param, dom_size=0, nvars=0) = abs(x - param)

"""
    _co_val_minus_param(x, param)
    _co_param_minus_val(x, param)
Return the difference `x - param` (resp. `param - x`) if positive, `0.0` otherwise.
"""
_co_val_minus_param(x; param, dom_size=0, nvars=0) = max(0.0, x - param)
_co_param_minus_val(x; param, dom_size=0, nvars=0) = max(0.0, param - x)

"""
    _co_euclidian_param(x, param, ds)
    _co_euclidian(x, ds)
Compute an euclidian norm with domain size `ds`, possibly weigthed by `param`, on a scalar.
"""
function _co_euclidian_param(x; param, dom_size, nvars=0)
    return x == param ? 0.0 : (1.0 + abs(x - param) \ dom_size)
end
function _co_euclidian(x; param=nothing, dom_size, nvars=0)
    return _co_euclidian_param(x; param=0.0, dom_size=dom_size)
end

"""
    _co_abs_diff_val_vars(x, nvars)
Return the absolute difference between `x` and the number of variables `nvars`.
"""
_co_abs_diff_val_vars(x; param=nothing, dom_size=0, nvars) = abs(x - nvars)

"""
    _co_val_minus_vars(x, nvars)
    _co_vars_minus_val(x, nvars)
Return the difference `x - nvars` (resp. `nvars - x`) if positive, `0.0` otherwise, where `nvars` denotes the numbers of variables.
"""
_co_val_minus_vars(x; param=nothing, dom_size=0, nvars) = _co_val_minus_param(x; param=nvars)
_co_vars_minus_val(x; param=nothing, dom_size=0, nvars) = _co_param_minus_val(x; param=nvars)

"""
    comparison_layer(nvars, dom_size, param = nothing)
Generate the layer of transformations functions of the ICN. Iff `param` value is set, also includes all the parametric transformation with that value.
"""
function comparison_layer(param=false)
    comparisons = LittleDict{Symbol,Function}(
        :identity => _co_identity,
        :euclidian => _co_euclidian,
        :abs_diff_val_vars => _co_abs_diff_val_vars,
        :val_minus_vars => _co_val_minus_vars,
        :vars_minus_val => _co_vars_minus_val,
    )

    if param
        comparisons_param = LittleDict{Symbol,Function}(
            :abs_diff_val_param => _co_abs_diff_val_param,
            :val_minus_param => _co_val_minus_param,
            :param_minus_val => _co_param_minus_val,
            :euclidian_param => _co_euclidian_param,
        )
        comparisons = LittleDict{Symbol,Function}(union(comparisons, comparisons_param))
    end

    return Layer(comparisons, true)
end
