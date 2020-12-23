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
    return _euclidian_param(x, 0.0, dom_size)
end

"""
    _abs_diff_val_vars(x::T, vars::V)
Return the absolute difference between `x` and the number of variables.
"""
function _abs_diff_val_vars(x::T, vars::V) where {T <: Number, V <: AbstractVector{T}}
    return abs(x - length(vars))
end

"""
    _val_minus_vars(x::T, vars::V)
    _vars_minus_val(x::T, vars::V)
Return the difference `x - length(vars)` (resp. `length(vars) - x`) if positive, `0.0` otherwise.
"""
function _val_minus_vars(x::T, vars::V) where {T <: Number, V <: AbstractVector{T}}
    return _val_minus_param(x, length(vars))
end
function _vars_minus_val(x::T, vars::V) where {T <: Number, V <: AbstractVector{T}}
    return _param_minus_val(x, length(vars))
end
