# Identity

"""
    _tr_identity(x)
    _tr_identity(i, x)
Identity function. Already defined in Julia as `identity`, specialized for vectors.
"""
_tr_identity(x; param=nothing) = identity(x)
_tr_identity(i, x; param=nothing) = identity(i)


# Count equalities

"""
    _tr_count_eq(i, x)
    _tr_count_eq(x)
Count the number of elements equal to `x[i]`. Extended method to vector with sig `(x)` are generated.
"""
_tr_count_eq(i, x; param=nothing) = count(y -> x[i] == y, x) - 1

"""
    _tr_count_eq_right(i, x)
    _tr_count_eq_right(x)
Count the number of elements to the right of and equal to `x[i]`. Extended method to vector with sig `(x)` are generated.
"""
_tr_count_eq_right(i, x; param=nothing) = _tr_count_eq(1, @view x[i:end])

"""
    _tr_count_eq_left(i, x)
    _tr_count_eq_left(x)
Count the number of elements to the left of and equal to `x[i]`. Extended method to vector with sig `(x)` are generated.
"""
_tr_count_eq_left(i, x; param=nothing) = _tr_count_eq(i, @view x[1:i])

# Generating vetorized versions
lazy(_tr_count_eq, _tr_count_eq_left, _tr_count_eq_right)

# Count greater/lesser

"""
    _tr_count_greater(i, x)
    _tr_count_greater(x)
Count the number of elements greater than `x[i]`. Extended method to vector with sig `(x)` are generated.
"""
_tr_count_greater(i, x; param=nothing) = count(y -> x[i] < y, x)

"""
    _tr_count_lesser(i, x)
    _tr_count_lesser(x)
Count the number of elements lesser than `x[i]`. Extended method to vector with sig `(x)` are generated.
"""
_tr_count_lesser(i, x; param=nothing) = count(y -> x[i] > y, x)

"""
    _tr_count_g_left(i, x)
    _tr_count_g_left(x)
Count the number of elements to the left of and greater than `x[i]`. Extended method to vector with sig `(x)` are generated.
"""
_tr_count_g_left(i, x; param=nothing) = _tr_count_greater(i, @view x[1:i])

"""
    _tr_count_l_left(i, x)
    _tr_count_l_left(x)
Count the number of elements to the left of and lesser than `x[i]`. Extended method to vector with sig `(x)` are generated.
"""
_tr_count_l_left(i, x; param=nothing) = _tr_count_lesser(i, @view x[1:i])

"""
    _tr_count_g_right(i, x)
    _tr_count_g_right(x)
Count the number of elements to the right of and greater than `x[i]`. Extended method to vector with sig `(x)` are generated.
"""
_tr_count_g_right(i, x; param=nothing) = _tr_count_greater(1, @view x[i:end])

"""
    _tr_count_l_right(i, x)
    _tr_count_l_right(x)
Count the number of elements to the right of and lesser than `x[i]`. Extended method to vector with sig `(x)` are generated.
"""
_tr_count_l_right(i, x; param=nothing) = _tr_count_lesser(1, @view x[i:end])

# Generating vetorized versions
lazy(_tr_count_greater, _tr_count_g_left, _tr_count_g_right)
lazy(_tr_count_lesser, _tr_count_l_left, _tr_count_l_right)

# Count param

"""
    _tr_count_eq_param(i, x; param)
    _tr_count_eq_param(x; param)
Count the number of elements equal to `x[i] + param`. Extended method to vector with sig `(x, param)` are generated.
"""
_tr_count_eq_param(i, x; param) = count(y -> y == x[i] + param, x)

"""
    _tr_count_l_param(i, x; param)
    _tr_count_l_param(x; param)
Count the number of elements lesser than `x[i] + param`. Extended method to vector with sig `(x, param)` are generated.
"""
_tr_count_l_param(i, x; param) = count(y -> y < x[i] + param, x)

"""
    _tr_count_g_param(i, x; param)
    _tr_count_g_param(x; param)
Count the number of elements greater than `x[i] + param`. Extended method to vector with sig `(x, param)` are generated.
"""
_tr_count_g_param(i, x; param) = count(y -> y > x[i] + param, x)

# Generating vetorized versions
lazy_param(_tr_count_eq_param, _tr_count_l_param, _tr_count_g_param)

# Bounding param

"""
    _tr_count_bounding_param(i, x; param)
    _tr_count_bounding_param(x; param)
Count the number of elements bounded (not strictly) by `x[i]` and `x[i] + param`. An extended method to vector with sig `(x, param)` is generated.
"""
_tr_count_bounding_param(i, x; param) = count(y -> x[i] ≤ y ≤ x[i] + param, x)

# Generating vetorized versions
lazy_param(_tr_count_bounding_param)

# Val/param substractions

"""
    _tr_val_minus_param(i, x; param)
    _tr_val_minus_param(x; param)
Return the difference `x[i] - param` if positive, `0.0` otherwise.  Extended method to vector with sig `(x, param)` are generated.
"""
_tr_val_minus_param(i, x; param) = max(0, x[i] - param)

"""
    _tr_param_minus_val(i, x; param)
    _tr_param_minus_val(x; param)
Return the difference `param - x[i]` if positive, `0.0` otherwise.  Extended method to vector with sig `(x, param)` are generated.
"""
_tr_param_minus_val(i, x; param) = max(0, param - x[i])

# Generating vetorized versions
lazy_param(_tr_val_minus_param, _tr_param_minus_val)

# Continuous values substraction
"""
    _tr_contiguous_vals_minus(i, x)
    _tr_contiguous_vals_minus(x)
Return the difference `x[i] - x[i + 1]` if positive, `0.0` otherwise. Extended method to vector with sig `(x)` are generated.
"""
_tr_contiguous_vals_minus(i, x; param=nothing) = length(x) == i ? 0 : _tr_val_minus_param(i, x; param=x[i + 1])

"""
    _tr_contiguous_vals_minus_rev(i, x)
    _tr_contiguous_vals_minus_rev(x)
Return the difference `x[i + 1] - x[i]` if positive, `0.0` otherwise. Extended method to vector with sig `(x)` are generated.
"""
function _tr_contiguous_vals_minus_rev(i, x; param=nothing)
    return length(x) == i ? 0 : _tr_param_minus_val(i, x; param=x[i + 1])
end

# Generating vetorized versions
lazy(_tr_contiguous_vals_minus, _tr_contiguous_vals_minus_rev)


"""
    transformation_layer(param = false)
Generate the layer of transformations functions of the ICN. Iff `param` value is true, also includes all the parametric transformations.
"""
function transformation_layer(param=false)
    transformations = LittleDict{Symbol,Function}(
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

    if param
        transformations_param = LittleDict{Symbol,Function}(
            :count_eq_param => _tr_count_eq_param,
            :count_l_param => _tr_count_l_param,
            :count_g_param => _tr_count_g_param,
            :count_bounding_param => _tr_count_bounding_param,
            :val_minus_param => _tr_val_minus_param,
            :param_minus_val => _tr_param_minus_val,
        )
        transformations = LittleDict(union(transformations, transformations_param))
    end

    return Layer(transformations, false)
end
