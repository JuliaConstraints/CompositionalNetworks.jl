"""
    _tr_identity(x::V) where {T <: Number,V <: AbstractVector{T}}
    _tr_identity(x::T) where T <: Number = identity(x)
Identity function. Already defined in Julia as `identity`, specialized for vectors and scalars.
"""
_tr_identity(x) = identity(x)
_tr_identity(i, x) = identity(i)

"""
    _tr_count_eq(i::Int, x::V)
    _tr_count_eq_right(i::Int, x::V)
    _tr_count_eq_left(i::Int, x::V)
Count the number of elements equal to `x[i]` (optionally to the right/left of `x[i]`). Extended method to vector `x::V` are generated.
"""
_tr_count_eq(i, x) = count(y -> x[i] == y, x) - 1
_tr_count_eq_right(i, x) = _tr_count_eq(1, @view x[i:end])
_tr_count_eq_left(i, x) = _tr_count_eq(i, @view x[1:i])

# Generating vetorized versions
lazy(_tr_count_eq, _tr_count_eq_left, _tr_count_eq_right)

"""
    _tr_count_greater(i::Int, x::V)
    _tr_count_lesser(i::Int, x::V)
    _tr_count_g_left(i::Int, x::V)
    _tr_count_l_left(i::Int, x::V)
    _tr_count_g_right(i::Int, x::V)
    _tr_count_l_right(i::Int, x::V)
Count the number of elements greater/lesser than `x[i]` (optionally to the left/right of `x[i]`). Extended method to vector with sig `(x::V)` are generated.
"""
_tr_count_greater(i, x) = count(y -> x[i] < y, x)
_tr_count_lesser(i, x) = count(y -> x[i] > y, x)
_tr_count_g_left(i, x) = _tr_count_greater(i, @view x[1:i])
_tr_count_l_left(i, x) = _tr_count_lesser(i, @view x[1:i])
_tr_count_g_right(i, x) = _tr_count_greater(1, @view x[i:end])
_tr_count_l_right(i, x) = _tr_count_lesser(1, @view x[i:end])

# Generating vetorized versions
lazy(_tr_count_greater, _tr_count_g_left, _tr_count_g_right)
lazy(_tr_count_lesser, _tr_count_l_left, _tr_count_l_right)

"""
    _tr_count_eq_param(i::Int, x::V, param::T)
    _tr_count_l_param(i::Int, x::V, param::T)
    _tr_count_g_param(i::Int, x::V, param::T)
Count the number of elements equal to (resp. lesser/greater than) `x[i] + param`. Extended method to vector with sig `(x::V, param::T)` are generated.
"""
_tr_count_eq_param(i, x, param) = count(y -> y == x[i] + param, x)
_tr_count_l_param(i, x, param) = count(y -> y < x[i] + param, x)
_tr_count_g_param(i, x, param) = count(y -> y > x[i] + param, x)

# Generating vetorized versions
lazy_param(_tr_count_eq_param, _tr_count_l_param, _tr_count_g_param)

"""
    _tr_count_bounding_param(i::Int, x::V, param::T)
Count the number of elements bounded (not strictly) by `x[i]` and `x[i] + param`. An extended method to vector with sig `(x::V, param::T)` is generated.
"""
_tr_count_bounding_param(i, x, param) = count(y -> x[i] ≤ y ≤ x[i] + param, x)

# Generating vetorized versions
lazy_param(_tr_count_bounding_param)

"""
    _tr_val_minus_param(i::Int, x::V, param::T)
    _tr_param_minus_val(i::Int, x::V, param::T)
Return the difference `x[i] - param` (resp. `param - x[i]`) if positive, `0.0` otherwise.  Extended method to vector with sig `(x::V, param::T)` are generated.
"""
_tr_val_minus_param(i, x, param) = max(0, x[i] - param)
_tr_param_minus_val(i, x, param) = max(0, param - x[i])

# Generating vetorized versions
lazy_param(_tr_val_minus_param, _tr_param_minus_val)

"""
    _tr_contiguous_vals_minus(i::Int, x::V)
    _tr_contiguous_vals_minus_rev(i::Int, x::V)
Return the difference `x[i] - x[i + 1]` (resp. `x[i + 1] - x[i]`) if positive, `0.0` otherwise. Extended method to vector with sig `(x::V)` are generated.
"""
_tr_contiguous_vals_minus(i, x) = length(x) == i ? 0 : _tr_val_minus_param(i, x, x[i + 1])
function _tr_contiguous_vals_minus_rev(i, x)
    return length(x) == i ? 0 : _tr_param_minus_val(i, x, x[i + 1])
end
# Generating vetorized versions
lazy(_tr_contiguous_vals_minus, _tr_contiguous_vals_minus_rev)


"""
    transformation_layer(param = nothing)
Generate the layer of transformations functions of the ICN. Iff `param` value is set, also includes all the parametric transformation with that value.
"""
function transformation_layer(param = nothing)
    transformations = LittleDict{Symbol, Function}(
        :identity => _tr_identity,
        :count_eq => _tr_count_eq,
        :count_eq_left => _tr_count_eq_left,
        :count_eq_right => _tr_count_eq_right,
        :count_greater => _tr_count_greater,
        :count_lesser => _tr_count_lesser,
        :count_g_left => _tr_count_g_left,
        :count_l_left => _tr_count_l_left,
        :count_g_right => _tr_count_g_right,
        :count_l_right => _tr_count_l_right,
        :contiguous_vals_minus => _tr_contiguous_vals_minus,
        :contiguous_vals_minus_rev => _tr_contiguous_vals_minus_rev,
    )

    if !isnothing(param)
        transformations_param = LittleDict{Symbol, Function}(
            :count_eq_param => ((x...) -> _tr_count_eq_param(x..., param)),
            :count_l_param => ((x...) -> _tr_count_l_param(x..., param)),
            :count_g_param => ((x...) -> _tr_count_g_param(x..., param)),
            :count_bounding_param => ((x...) -> _tr_count_bounding_param(x..., param)),
            :val_minus_param => ((x...) -> _tr_val_minus_param(x..., param)),
            :param_minus_val => ((x...) -> _tr_param_minus_val(x..., param)),
        )
        transformations = LittleDict(union(transformations, transformations_param))
    end

    return Layer(transformations, false)
end
