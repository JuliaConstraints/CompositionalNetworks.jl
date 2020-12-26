"""
    _identity(x::V) where {T <: Number,V <: AbstractVector{T}}
    _identity(x::T) where T <: Number = identity(x)
Identity function. Already defined in Julia as `identity`, specialized for vectors and scalars.
"""
_identity(x::V) where {T <: Number,V <: AbstractVector{T}} = identity(x)

"""
    _count_eq(i::Int, x::V)
    _count_eq_right(i::Int, x::V)
    _count_eq_left(i::Int, x::V)
Count the number of elements equal to `x[i]` (optionally to the right/left of `x[i]`). Extended method to vector `x::V` are generated.
"""
function _count_eq(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> x[i] == y, x) - 1
end
function _count_eq_right(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_eq(1, @view x[i:end])
end
function _count_eq_left(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_eq(i, @view x[1:i])
end
# Generating vetorized versions
lazy(_count_eq, _count_eq_left, _count_eq_right)

"""
    _count_greater(i::Int, x::V)
    _count_lesser(i::Int, x::V)
    _count_g_left(i::Int, x::V)
    _count_l_left(i::Int, x::V)
    _count_g_right(i::Int, x::V)
    _count_l_right(i::Int, x::V)
Count the number of elements greater/lesser than `x[i]` (optionally to the left/right of `x[i]`). Extended method to vector with sig `(x::V)` are generated.
"""
function _count_greater(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> x[i] < y, x)
end
function _count_lesser(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> x[i] > y, x)
end
function _count_g_left(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_greater(i, @view x[1:i])
end
function _count_l_left(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_lesser(i, @view x[1:i])
end
function _count_g_right(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_greater(1, @view x[i:end])
end
function _count_l_right(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return _count_lesser(1, @view x[i:end])
end
# Generating vetorized versions
lazy(_count_greater, _count_g_left, _count_g_right)
lazy(_count_lesser, _count_l_left, _count_l_right)

"""
    _count_eq_param(i::Int, x::V, param::T)
    _count_l_param(i::Int, x::V, param::T)
    _count_g_param(i::Int, x::V, param::T)
Count the number of elements equal to (resp. lesser/greater than) `x[i] + param`. Extended method to vector with sig `(x::V, param::T)` are generated.
"""
function _count_eq_param(i::Int, x::V, param::T) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> y == x[i] + param, x)
end
function _count_l_param(i::Int, x::V, param::T) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> y < x[i] + param, x)
end
function _count_g_param(i::Int, x::V, param::T) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> y > x[i] + param, x)
end
# Generating vetorized versions
lazy_param(_count_eq_param, _count_l_param, _count_g_param)

"""
    _count_bounding_param(i::Int, x::V, param::T)
Count the number of elements bounded (not strictly) by `x[i]` and `x[i] + param`. An extended method to vector with sig `(x::V, param::T)` is generated.
"""
function _count_bounding_param(i::Int, x::V, param::T
) where {T <: Number,V <: AbstractVector{T}}
    return count(y -> x[i] ≤ y ≤ x[i] + param, x)
end
# Generating vetorized versions
lazy_param(_count_bounding_param)

"""
    _val_minus_param(i::Int, x::V, param::T)
    _param_minus_val(i::Int, x::V, param::T)
Return the difference `x[i] - param` (resp. `param - x[i]`) if positive, `0.0` otherwise.  Extended method to vector with sig `(x::V, param::T)` are generated.
"""
function _val_minus_param(i::Int, x::V, param::T
) where {T <: Number,V <: AbstractVector{T}}
    return max(0, x[i] - param)
end
function _param_minus_val(i::Int, x::V, param::T
) where {T <: Number,V <: AbstractVector{T}}
    return max(0, param - x[i])
end
# Generating vetorized versions
lazy_param(_val_minus_param, _param_minus_val)

"""
    _contiguous_vals_minus(i::Int, x::V)
    _contiguous_vals_minus_rev(i::Int, x::V)
Return the difference `x[i] - x[i + 1]` (resp. `x[i + 1] - x[i]`) if positive, `0.0` otherwise. Extended method to vector with sig `(x::V)` are generated.
"""
function _contiguous_vals_minus(i::Int, x::V) where {T <: Number,V <: AbstractVector{T}}
    return length(x) == i ? 0 : _val_minus_param(i, x, x[i + 1])
end
function _contiguous_vals_minus_rev(i::Int, x::V
) where {T <: Number,V <: AbstractVector{T}}
    return length(x) == i ? 0 : _param_minus_val(i, x, x[i + 1])
end
# Generating vetorized versions
lazy(_contiguous_vals_minus, _contiguous_vals_minus_rev)


"""
    transformation_layer(param = nothing)
Generate the layer of transformations functions of the ICN. Iff `param` value is set, also includes all the parametric transformation with that value.
"""
function transformation_layer(param = nothing)
    transformations = Dict{Symbol, Function}(
        :identity => _identity,
        :count_eq => _count_eq,
        :count_eq_left => _count_eq_left,
        :count_eq_right => _count_eq_right,
        :count_greater => _count_greater,
        :count_lesser => _count_lesser,
        :count_g_left => _count_g_left,
        :count_l_left => _count_l_left,
        :count_g_right => _count_g_right,
        :count_l_right => _count_l_right,
        :contiguous_vals_minus => _contiguous_vals_minus,
        :contiguous_vals_minus_rev => _contiguous_vals_minus_rev,
    )
    
    if !isnothing(param)
        transformations_param = Dict{Symbol, Function}(
            :count_eq_param => ((x...) -> _count_eq_param(x..., param)),
            :count_l_param => ((x...) -> _count_l_param(x..., param)),
            :count_g_param => ((x...) -> _count_g_param(x..., param)),
            :count_bounding_param => ((x...) -> _count_bounding_param(x..., param)),
            :val_minus_param => ((x...) -> _val_minus_param(x..., param)),
            :param_minus_val => ((x...) -> _param_minus_val(x..., param)),
        )
        transformations = Dict(union(transformations, transformations_param))
    end

    return transformations
end