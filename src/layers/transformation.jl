# Identity

"""
    tr_identity(i, x)
    tr_identity(x)
    tr_identity(x, X::AbstractVector)

Identity function. Already defined in Julia as `identity`, specialized for vectors.
When `X` is provided, the result is computed without allocations.
"""
tr_identity(i, x) = identity(x[i])
lazy(tr_identity)

# Count equalities

"""
    tr_count_eq(i, x)
    tr_count_eq(x)
    tr_count_eq(x, X::AbstractVector)

Count the number of elements equal to `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_eq(i, x) = count(y -> x[i] == y, x) - 1

"""
    tr_count_eq_right(i, x)
    tr_count_eq_right(x)
    tr_count_eq_right(x, X::AbstractVector)

Count the number of elements to the right of and equal to `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_eq_right(i, x) = tr_count_eq(1, @view x[i:end])

"""
    tr_count_eq_left(i, x)
    tr_count_eq_left(x)
    tr_count_eq_left(x, X::AbstractVector)

Count the number of elements to the left of and equal to `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_eq_left(i, x) = tr_count_eq(i, @view x[1:i])

# Generating vetorized versions
lazy(tr_count_eq, tr_count_eq_left, tr_count_eq_right)

# Count greater/lesser

"""
    tr_count_greater(i, x)
    tr_count_greater(x)
    tr_count_greater(x, X::AbstractVector)

Count the number of elements greater than `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_greater(i, x) = count(y -> x[i] < y, x)

"""
    tr_count_lesser(i, x)
    tr_count_lesser(x)
    tr_count_lesser(x, X::AbstractVector)

Count the number of elements lesser than `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_lesser(i, x) = count(y -> x[i] > y, x)

"""
    tr_count_g_left(i, x)
    tr_count_g_left(x)
    tr_count_g_left(x, X::AbstractVector)

Count the number of elements to the left of and greater than `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_g_left(i, x) = tr_count_greater(i, @view x[1:i])

"""
    tr_count_l_left(i, x)
    tr_count_l_left(x)
    tr_count_l_left(x, X::AbstractVector)

Count the number of elements to the left of and lesser than `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_l_left(i, x) = tr_count_lesser(i, @view x[1:i])

"""
    tr_count_g_right(i, x)
    tr_count_g_right(x)
    tr_count_g_right(x, X::AbstractVector)

Count the number of elements to the right of and greater than `x[i]`. Extended method to vector with sig `(x)` are generated.
"""
tr_count_g_right(i, x) = tr_count_greater(1, @view x[i:end])

"""
    tr_count_l_right(i, x)
    tr_count_l_right(x)
    tr_count_l_right(x, X::AbstractVector)

Count the number of elements to the right of and lesser than `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_l_right(i, x) = tr_count_lesser(1, @view x[i:end])

# Generating vetorized versions
lazy(tr_count_greater, tr_count_g_left, tr_count_g_right)
lazy(tr_count_lesser, tr_count_l_left, tr_count_l_right)

# Count param

"""
    tr_count_eq_param(i, x; param)
    tr_count_eq_param(x; param)
    tr_count_eq_param(x, X::AbstractVector; param)

Count the number of elements equal to `x[i] + param`. Extended method to vector with sig `(x, param)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_eq_param(i, x; param) = count(y -> y == x[i] + param, x)

"""
    tr_count_l_param(i, x; param)
    tr_count_l_param(x; param)
    tr_count_l_param(x, X::AbstractVector; param)

Count the number of elements lesser than `x[i] + param`. Extended method to vector with sig `(x, param)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_l_param(i, x; param) = count(y -> y < x[i] + param, x)

"""
    tr_count_g_param(i, x; param)
    tr_count_g_param(x; param)
    tr_count_g_param(x, X::AbstractVector; param)

Count the number of elements greater than `x[i] + param`. Extended method to vector with sig `(x, param)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_g_param(i, x; param) = count(y -> y > x[i] + param, x)

# Generating vetorized versions
lazy_param(tr_count_eq_param, tr_count_l_param, tr_count_g_param)

# Bounding param

"""
    tr_count_bounding_param(i, x; param)
    tr_count_bounding_param(x; param)
    tr_count_bounding_param(x, X::AbstractVector; param)

Count the number of elements bounded (not strictly) by `x[i]` and `x[i] + param`. An extended method to vector with sig `(x, param)` is generated.
When `X` is provided, the result is computed without allocations.
"""
tr_count_bounding_param(i, x; param) = count(y -> x[i] ≤ y ≤ x[i] + param, x)

# Generating vetorized versions
lazy_param(tr_count_bounding_param)

# Val/param subtractions

"""
    tr_val_minus_param(i, x; param)
    tr_val_minus_param(x; param)
    tr_val_minus_param(x, X::AbstractVector; param)

Return the difference `x[i] - param` if positive, `0.0` otherwise.  Extended method to vector with sig `(x, param)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_val_minus_param(i, x; param) = max(0, x[i] - param)

"""
    tr_param_minus_val(i, x; param)
    tr_param_minus_val(x; param)
    tr_param_minus_val(x, X::AbstractVector; param)

Return the difference `param - x[i]` if positive, `0.0` otherwise.  Extended method to vector with sig `(x, param)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_param_minus_val(i, x; param) = max(0, param - x[i])

# Generating vetorized versions
lazy_param(tr_val_minus_param, tr_param_minus_val)

# Continuous values subtraction
"""
    tr_contiguous_vals_minus(i, x)
    tr_contiguous_vals_minus(x)
    tr_contiguous_vals_minus(x, X::AbstractVector)

Return the difference `x[i] - x[i + 1]` if positive, `0.0` otherwise. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""
tr_contiguous_vals_minus(i, x; param = nothing) =
    length(x) == i ? 0 : tr_val_minus_param(i, x; param = x[i+1])

"""
    tr_contiguous_vals_minus_rev(i, x)
    tr_contiguous_vals_minus_rev(x)
    tr_contiguous_vals_minus_rev(x, X::AbstractVector)

Return the difference `x[i + 1] - x[i]` if positive, `0.0` otherwise. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""
function tr_contiguous_vals_minus_rev(i, x; param = nothing)
    return length(x) == i ? 0 : tr_param_minus_val(i, x; param = x[i+1])
end

# Generating vetorized versions
lazy(tr_contiguous_vals_minus, tr_contiguous_vals_minus_rev)

# Parametric layers
make_transformations(param::Symbol) = make_transformations(Val(param))

function make_transformations(::Val{:none})
    return LittleDict{Symbol,Function}(
        :identity => tr_identity,
        :count_eq => tr_count_eq,
        :count_eq_left => tr_count_eq_left,
        :count_eq_right => tr_count_eq_right,
        :count_greater => tr_count_greater,
        :count_lesser => tr_count_lesser,
        :count_g_left => tr_count_g_left,
        :count_l_left => tr_count_l_left,
        :count_g_right => tr_count_g_right,
        :count_l_right => tr_count_l_right,
        :contiguous_vals_minus => tr_contiguous_vals_minus,
        :contiguous_vals_minus_rev => tr_contiguous_vals_minus_rev,
    )
end

function make_transformations(::Val{:val})
    return LittleDict{Symbol,Function}(
        :count_eq_param => tr_count_eq_param,
        :count_l_param => tr_count_l_param,
        :count_g_param => tr_count_g_param,
        :count_bounding_param => tr_count_bounding_param,
        :val_minus_param => tr_val_minus_param,
        :param_minus_val => tr_param_minus_val,
    )
end

function make_transformations(::Val)
    return LittleDict{Symbol,Function}()
end


"""
    transformation_layer(param = false)
Generate the layer of transformations functions of the ICN. Iff `param` value is true, also includes all the parametric transformations.
"""
function transformation_layer(parameters = Vector{Symbol}())
    transformations = make_transformations(:none)

    for p in parameters
        transformations_param = make_transformations(p)
        transformations = LittleDict(union(transformations, transformations_param))
    end

    return Layer(false, transformations, parameters)
end

## SECTION - Test Items
@testitem "Arithmetic Layer" tags = [:arithmetic, :layer] begin
    CN = CompositionalNetworks

    data = [[1, 5, 2, 4, 3] => 2, [1, 2, 3, 2, 1] => 2]

    # Test transformations without parameters
    funcs = Dict(
        CN.tr_identity => [data[1].first, data[2].first],
        CN.tr_count_eq => [[0, 0, 0, 0, 0], [1, 1, 0, 1, 1]],
        CN.tr_count_eq_right => [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0]],
        CN.tr_count_eq_left => [[0, 0, 0, 0, 0], [0, 0, 0, 1, 1]],
        CN.tr_count_greater => [[4, 0, 3, 1, 2], [3, 1, 0, 1, 3]],
        CN.tr_count_lesser => [[0, 4, 1, 3, 2], [0, 2, 4, 2, 0]],
        CN.tr_count_g_left => [[0, 0, 1, 1, 2], [0, 0, 0, 1, 3]],
        CN.tr_count_l_left => [[0, 1, 1, 2, 2], [0, 1, 2, 1, 0]],
        CN.tr_count_g_right => [[4, 0, 2, 0, 0], [3, 1, 0, 0, 0]],
        CN.tr_count_l_right => [[0, 3, 0, 1, 0], [0, 1, 2, 1, 0]],
        CN.tr_contiguous_vals_minus => [[0, 3, 0, 1, 0], [0, 0, 1, 1, 0]],
        CN.tr_contiguous_vals_minus_rev => [[4, 0, 2, 0, 0], [1, 1, 0, 0, 0]],
    )

    for (f, results) in funcs
        @info f
        for (key, vals) in enumerate(data)
            @test f(vals.first) == results[key]
            foreach(i -> f(i, vals.first), vals.first)
        end
    end

    # Test transformations with parameter
    funcs_param = Dict(
        CN.tr_count_eq_param => [[1, 0, 1, 0, 1], [1, 0, 0, 0, 1]],
        CN.tr_count_l_param => [[2, 5, 3, 5, 4], [4, 5, 5, 5, 4]],
        CN.tr_count_g_param => [[2, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
        CN.tr_count_bounding_param => [[3, 1, 3, 2, 3], [5, 3, 1, 3, 5]],
        CN.tr_val_minus_param => [[0, 3, 0, 2, 1], [0, 0, 1, 0, 0]],
        CN.tr_param_minus_val => [[1, 0, 0, 0, 0], [1, 0, 0, 0, 1]],
    )

    for (f, results) in funcs_param
        @info f
        for (key, vals) in enumerate(data)
            @test f(vals.first; param = vals.second) == results[key]
            foreach(i -> f(i, vals.first; param = vals.second), vals.first)
        end
    end

end
