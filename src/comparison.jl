# doc in transformation.jl
_identity(x::T) where T <: Number = identity(x)

"""
    _abs_diff_val_param(x::T, param::T)
Return the absolute difference between `x` and `param`.
"""
_abs_diff_val_param(x::T, param::T) where T <: Number = abs(x - param)

"""
    _val_minus_param(x::T, param::T)
    _param_minus_val(x::T, param::T)
Return the difference `x - param` (resp. `param - x`) if positive, `0.0` otherwise.
"""
_val_minus_param(x::T, param::T) where T <: Number = max(0.0, x - param)
_param_minus_val(x::T, param::T) where T <: Number = max(0.0, param - x)

"""
    _euclidian_param(x::T, param::T, dom_size::T2)
    _euclidian(x::T, dom_size::T2)
Compute an euclidian norm , possibly weigthed by `param`, on a scalar.
"""
function _euclidian_param(x::T, param::T, dom_size::T2) where {T <: Number, T2 <: Number}
    return x == param ? 0.0 : (1.0 + abs(x - param) \ dom_size)
end
function _euclidian(x::T, dom_size::T2) where {T <: Number, T2 <: Number}
    return _euclidian_param(x, zero(T), dom_size)
end

"""
    _abs_diff_val_vars(x::T, nvars::Int)
Return the absolute difference between `x` and the number of variables.
"""
function _abs_diff_val_vars(x::T, nvars::Int) where {T <: Number}
    return abs(x - nvars)
end

"""
    _val_minus_vars(x::T, nvars::Int)
    _vars_minus_val(x::T, nvars::Int)
Return the difference `x - nvars` (resp. `nvars - x`) if positive, `0.0` otherwise.
"""
function _val_minus_vars(x::T, nvars::Int) where {T <: Number}
    return _val_minus_param(x, nvars)
end
function _vars_minus_val(x::T, nvars::Int) where {T <: Number}
    return _param_minus_val(x, nvars)
end

"""
    comparison_layer(nvars, dom_size, param = nothing)
Generate the layer of transformations functions of the ICN. Iff `param` value is set, also includes all the parametric transformation with that value.
"""
function comparison_layer(nvars, dom_size, param = nothing)
    comparisons = LittleDict{Symbol, Function}(
        :identity => _identity,
        :euclidian => (x -> _euclidian(x, dom_size)),
        :abs_diff_val_vars => (x -> _abs_diff_val_vars(x, nvars)),
        :val_minus_vars => (x -> _val_minus_vars(x, nvars)),
        :vars_minus_val => (x -> _vars_minus_val(x, nvars)),
    )

    if !isnothing(param)
        comparisons_param = LittleDict{Symbol, Function}(
            :abs_diff_val_param => (x -> _abs_diff_val_param(x, param)),
            :val_minus_param => (x -> _val_minus_param(x, param)),
            :param_minus_val => (x -> _param_minus_val(x, param)),
            :euclidian_param => (x -> _euclidian_param(x, param, dom_size)),
        )
        comparisons = LittleDict{Symbol, Function}(union(comparisons, comparisons_param))
    end

    return Layer(comparisons, true)
end
