"""
    _identity(x::V)
Identity function. Already defined in Julia as `identity`, specialized for vectors.
"""
_identity(x::V) where {T <: Number,V <: AbstractVector{T}} = identity(x)

"""
    _count_eq(i::Int, x::V)
    _count_eq_right(i::Int, x::V)
    _count_eq_left(i::Int, x::V)
Count the number of elements equal to `x[i]` (optionally to the right/left of `x[i]`).
"""
function _count_eq(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> x[i] == y, x)
end
function _count_eq_right(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_eq(i, @view x[(i + 1):end])
end
function _count_eq_left(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_eq(i, @view x[1:(i - 1)])
end

"""
    _count_greater(i::Int, x::V)
    _count_lesser(i::Int, x::V)
    _count_g_left(i::Int, x::V)
    _count_l_left(i::Int, x::V)
    _count_g_right(i::Int, x::V)
    _count_l_right(i::Int, x::V)
Count the number of elements greater/lesser than `x[i]` (optionally to the left/right of `x[i]`).
"""
function _count_greater(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> x[i] < y, x)
end
function _count_lesser(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> x[i] < y, x)
end
function _count_g_left(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_greater(i, @view x[1:(i - 1)])
end
function _count_l_left(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_lesser(i, @view x[1:(i - 1)])
end
function _count_g_right(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_greater(i, @view x[(i + 1):end])
end
function _count_l_right(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_lesser(i, @view x[(i + 1):end])
end

"""
    _count_eq_param(param, i::Int, x::V)
    _count_l_param(param, i::Int, x::V)
    _count_g_param(param, i::Int, x::V)
Count the number of elements equal to (resp. lesser/greater than) `x[i] + param`.
"""
function _count_eq_param(param::T, i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> y == x[i] + param, x)
end
function _count_l_param(param::T, i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> y < x[i] + param, x)
end
function _count_g_param(param::T, i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> y > x[i] + param, x)
end

"""
    _count_eq_param(param, i::Int, x::V)
    _count_l_param(param, i::Int, x::V)
    _count_g_param(param, i::Int, x::V)
Count the number of elements bounded (not strictly) by `x[i]` and `x[i] + param`.
"""
function _count_bounding_param(param::T, i::Int, x::V
) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> x[i] ≤ y ≤ x[i] + param, x)
end

"""
    _val_minus_param(param, i::Int, x::V)
    _param_minus_val(param, i::Int, x::V)
Return the difference `x[i] - param` (resp. `param - x[i]`) if positive, `0.0` otherwise.
"""
function _val_minus_param(param::T, i::Int, x::V
) where {T <: Number,V <: AbstractVector{T}}
    return max(0.0, x[i] - param)
end
function _param_minus_val(param::T, i::Int, x::V
) where {T <: Number,V <: AbstractVector{T}}
    return max(0.0, param - x[i])
end

"""
    _contiguous_vals_minus(i::Int, x::V)
    _contiguous_vals_minus_rev(i::Int, x::V)
Return the difference `x[i] - x[i + 1]` (resp. `x[i + 1] - x[i]`) if positive, `0.0` otherwise.
"""
function _contiguous_vals_minus(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return length(x) == i ? 0.0 : _val_minus_param(x[i + 1], i, x)
end
function _contiguous_vals_minus_rev(i::Int, x::V
) where {T <: Number,V <: AbstractVector{T}}
    return length(x) == i ? 0.0 : _param_minus_val(x[i + 1], i, x)
end
