"""
    co_identity(x)
Identity function. Already defined in Julia as `identity`, specialized for scalars in the `comparison` layer.
"""
co_identity(x; param=nothing, dom_size=0, nvars=0) = identity(x)

"""
    co_abs_diff_val_param(x; param)
Return the absolute difference between `x` and `param`.
"""
co_abs_diff_val_param(x; param, dom_size=0, nvars=0) = abs(x - param)

"""
    co_val_minus_param(x; param)
Return the difference `x - param` if positive, `0.0` otherwise.
"""
co_val_minus_param(x; param, dom_size=0, nvars=0) = max(0.0, x - param)

"""
    co_param_minus_val(x; param)
Return the difference `param - x` if positive, `0.0` otherwise.
"""
co_param_minus_val(x; param, dom_size=0, nvars=0) = max(0.0, param - x)

"""
    co_euclidian_param(x; param, dom_size)
Compute an euclidian norm with domain size `dom_size`, weigthed by `param`, of a scalar.
"""
function co_euclidian_param(x; param, dom_size, nvars=0)
    return x == param ? 0.0 : (1.0 + abs(x - param) / dom_size)
end

"""
    co_euclidian(x; dom_size)
Compute an euclidian norm with domain size `dom_size` of a scalar.
"""
function co_euclidian(x; param=nothing, dom_size, nvars=0)
    return co_euclidian_param(x; param=0.0, dom_size=dom_size)
end

"""
    co_abs_diff_val_vars(x; nvars)
Return the absolute difference between `x` and the number of variables `nvars`.
"""
co_abs_diff_val_vars(x; param=nothing, dom_size=0, nvars) = abs(x - nvars)

"""
    co_val_minus_vars(x; nvars)
Return the difference `x - nvars` if positive, `0.0` otherwise, where `nvars` denotes the numbers of variables.
"""
co_val_minus_vars(x; param=nothing, dom_size=0, nvars) = co_val_minus_param(x; param=nvars)

"""
    co_vars_minus_val(x; nvars)
Return the difference `nvars - x` if positive, `0.0` otherwise, where `nvars` denotes the numbers of variables.
"""
co_vars_minus_val(x; param=nothing, dom_size=0, nvars) = co_param_minus_val(x; param=nvars)


# Parametric layers
make_comparisons(param::Symbol) = make_comparisons(Val(param))

function make_comparisons(::Val{:none})
    return LittleDict{Symbol,Function}(
        :identity => co_identity,
        :euclidian => co_euclidian,
        :abs_diff_val_vars => co_abs_diff_val_vars,
        :val_minus_vars => co_val_minus_vars,
        :vars_minus_val => co_vars_minus_val,
    )
end

function make_comparisons(::Val{:val})
    return LittleDict{Symbol,Function}(
        :abs_diff_val_param => co_abs_diff_val_param,
        :val_minus_param => co_val_minus_param,
        :param_minus_val => co_param_minus_val,
        :euclidian_param => co_euclidian_param,
    )
end


"""
    comparison_layer(param = false)
Generate the layer of transformations functions of the ICN. Iff `param` value is set, also includes all the parametric comparison with that value. The operations are mutually exclusive, that is only one will be selected.
"""
function comparison_layer(parameters = Vector{Symbol}())
    comparisons = make_comparisons(:none)

    for p in parameters
        comparisons_param = make_comparisons(p)
        comparisons = LittleDict{Symbol,Function}(union(comparisons, comparisons_param))
    end

    return Layer(true, comparisons, parameters)
end
