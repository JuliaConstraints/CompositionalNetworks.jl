const Transformation = LayerCore(
    :Transformation,
    false,
    (:(AbstractVector{<:Real}),) => AbstractVector{<:Real},
    (
        id = :((x) -> identity(x)),
        count_equal_right = :(
            (x) -> map(i -> count(t -> t == x[i], @view(x[(i + 1):end])), eachindex(x))
        ),
        count_less_right = :(
            (x) -> map(i -> count(t -> t < x[i], @view(x[(i + 1):end])), eachindex(x))
        ),
        count_great_right = :(
            (x) -> map(i -> count(t -> t > x[i], @view(x[(i + 1):end])), eachindex(x))
        ),
        count_equal_left = :(
            (x) -> map(i -> count(t -> t == x[i], @view(x[1:(i - 1)])), eachindex(x))
        ),
        count_less_left = :(
            (x) -> map(i -> count(t -> t < x[i], @view(x[1:(i - 1)])), eachindex(x))
        ),
        count_great_left = :(
            (x) -> map(i -> count(t -> t > x[i], @view(x[1:(i - 1)])), eachindex(x))
        ),
        count_equal_val = :((x; val) -> map(i -> count(t -> t == (i + val), x), x)),
        count_less_val = :((x; val) -> map(i -> count(t -> t < (i + val), x), x)),
        count_great_val = :((x; val) -> map(i -> count(t -> t > (i + val), x), x)),
        var_minus_val = :((x; val) -> map(i -> max(0, i - val), x)),
        val_minus_var = :((x; val) -> map(i -> max(0, val - i), x)),
        contiguous_vars_minus = :(
            (x) -> map(
            i -> i == length(x) ? 0 : max(0, x[i] - x[i + 1]),
            eachindex(x[1:end])
        )
        ),
        contiguous_vars_minus_rev = :(
            (x) -> map(
            i -> i == length(x) ? 0 : max(0, x[i + 1] - x[i]),
            eachindex(x[1:end])
        )
        ),
        count_equal = :((x) -> map(i -> count(t -> t == i, x), x)),
        count_less = :((x) -> map(i -> count(t -> t < i, x), x)),
        count_great = :((x) -> map(i -> count(t -> t > i, x), x)),
        count_bounding_val = :(
            (x; val) -> map(i -> count(t -> t >= i && t <= i + val, x), x)
        ),
        var_minus_vals = :((x; vals) -> map(i -> max(0, (i .- vals)...), x)),
        vals_minus_var = :((x; vals) -> map(i -> max(0, (vals .- i)...), x))
    )
)

# SECTION - Docstrings to put back/update

"""
    tr_identity(i, x)
    tr_identity(x)
    tr_identity(x, X::AbstractVector)

Identity function. Already defined in Julia as `identity`, specialized for vectors.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_eq(i, x)
    tr_count_eq(x)
    tr_count_eq(x, X::AbstractVector)

Count the number of elements equal to `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_eq_right(i, x)
    tr_count_eq_right(x)
    tr_count_eq_right(x, X::AbstractVector)

Count the number of elements to the right of and equal to `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_eq_left(i, x)
    tr_count_eq_left(x)
    tr_count_eq_left(x, X::AbstractVector)

Count the number of elements to the left of and equal to `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_greater(i, x)
    tr_count_greater(x)
    tr_count_greater(x, X::AbstractVector)

Count the number of elements greater than `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_lesser(i, x)
    tr_count_lesser(x)
    tr_count_lesser(x, X::AbstractVector)

Count the number of elements lesser than `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_g_left(i, x)
    tr_count_g_left(x)
    tr_count_g_left(x, X::AbstractVector)

Count the number of elements to the left of and greater than `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_l_left(i, x)
    tr_count_l_left(x)
    tr_count_l_left(x, X::AbstractVector)

Count the number of elements to the left of and lesser than `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_g_right(i, x)
    tr_count_g_right(x)
    tr_count_g_right(x, X::AbstractVector)

Count the number of elements to the right of and greater than `x[i]`. Extended method to vector with sig `(x)` are generated.
"""

"""
    tr_count_l_right(i, x)
    tr_count_l_right(x)
    tr_count_l_right(x, X::AbstractVector)

Count the number of elements to the right of and lesser than `x[i]`. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_eq_val(i, x; val)
    tr_count_eq_val(x; val)
    tr_count_eq_val(x, X::AbstractVector; val)

Count the number of elements equal to `x[i] + val`. Extended method to vector with sig `(x, val)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_l_val(i, x; val)
    tr_count_l_val(x; val)
    tr_count_l_val(x, X::AbstractVector; val)

Count the number of elements lesser than `x[i] + val`. Extended method to vector with sig `(x, val)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_g_val(i, x; val)
    tr_count_g_val(x; val)
    tr_count_g_val(x, X::AbstractVector; val)

Count the number of elements greater than `x[i] + val`. Extended method to vector with sig `(x, val)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_count_bounding_val(i, x; val)
    tr_count_bounding_val(x; val)
    tr_count_bounding_val(x, X::AbstractVector; val)

Count the number of elements bounded (not strictly) by `x[i]` and `x[i] + val`. An extended method to vector with sig `(x, val)` is generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_var_minus_val(i, x; val)
    tr_var_minus_val(x; val)
    tr_var_minus_val(x, X::AbstractVector; val)

Return the difference `x[i] - val` if positive, `0.0` otherwise.  Extended method to vector with sig `(x, val)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_val_minus_var(i, x; val)
    tr_val_minus_var(x; val)
    tr_val_minus_var(x, X::AbstractVector; val)

Return the difference `val - x[i]` if positive, `0.0` otherwise.  Extended method to vector with sig `(x, val)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_contiguous_vars_minus(i, x)
    tr_contiguous_vars_minus(x)
    tr_contiguous_vars_minus(x, X::AbstractVector)

Return the difference `x[i] - x[i + 1]` if positive, `0.0` otherwise. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    tr_contiguous_vars_minus_rev(i, x)
    tr_contiguous_vars_minus_rev(x)
    tr_contiguous_vars_minus_rev(x, X::AbstractVector)

Return the difference `x[i + 1] - x[i]` if positive, `0.0` otherwise. Extended method to vector with sig `(x)` are generated.
When `X` is provided, the result is computed without allocations.
"""

"""
    make_transformations(param::Symbol)

Generates a dictionary of transformation functions based on the specified parameterization.
This function facilitates the creation of parametric layers for constraint transformations,
allowing for flexible and dynamic constraint manipulation according to the needs of different
constraint programming models.

## Parameters
- `param::Symbol`: Specifies the type of transformations to generate. It can be `:none` for
  basic transformations that do not depend on external parameters, or `:val` for transformations that operate with respect to a specific value parameter.

## Returns
- `LittleDict{Symbol, Function}`: A dictionary mapping transformation names (`Symbol`) to
  their corresponding functions (`Function`). The functions encapsulate various types of
  transformations, such as counting, comparison, and contiguous value processing.

## Transformation Types
- When `param` is `:none`, the following transformations are available:
  - `:identity`: No transformation is applied.
  - `:count_eq`, `:count_eq_left`, `:count_eq_right`: Count equalities under different conditions.
  - `:count_greater`, `:count_lesser`: Count values greater or lesser than a threshold.
  - `:count_g_left`, `:count_l_left`, `:count_g_right`, `:count_l_right`: Count values with greater or lesser comparisons from different directions.
  - `:contiguous_vals_minus`, `:contiguous_vals_minus_rev`: Process contiguous values with subtraction in normal and reverse order.

- When `param` is `:val`, the transformations relate to operations involving a parameter value:
  - `:count_eq_param`, `:count_l_param`, `:count_g_param`: Count equalities or comparisons against a parameter value.
  - `:count_bounding_param`: Count values bounding a parameter value.
  - `:val_minus_param`, `:param_minus_val`: Subtract a parameter value from values or vice versa.

The function delegates to a version that uses `Val(param)` for dispatch, ensuring compile-time selection of the appropriate transformation set.

## Examples
```julia
# Get basic transformations
basic_transforms = make_transformations(:none)

# Apply an identity transformation
identity_result = basic_transforms[:identity](data)

# Get value-based transformations
val_transforms = make_transformations(:val)

# Apply a count equal to parameter transformation
count_eq_param_result = val_transforms[:count_eq_param](data, param)
```
"""

"""
    transformation_layer(param = Vector{Symbol}())
Generate the layer of transformations functions of the ICN. Iff `param` value is non empty, also includes all the related parametric transformations.
"""

## SECTION - Test Items
# @testitem "Transformation Layer" tags = [:transformation, :layer] begin
#     CN = CompositionalNetworks

#     data = [[1, 5, 2, 4, 3] => 2, [1, 2, 3, 2, 1] => 2]

#     # Test transformations without parameters
#     funcs = Dict(
#         CN.tr_identity => [data[1].first, data[2].first],
#         CN.tr_count_eq => [[0, 0, 0, 0, 0], [1, 1, 0, 1, 1]],
#         CN.tr_count_eq_right => [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0]],
#         CN.tr_count_eq_left => [[0, 0, 0, 0, 0], [0, 0, 0, 1, 1]],
#         CN.tr_count_greater => [[4, 0, 3, 1, 2], [3, 1, 0, 1, 3]],
#         CN.tr_count_lesser => [[0, 4, 1, 3, 2], [0, 2, 4, 2, 0]],
#         CN.tr_count_g_left => [[0, 0, 1, 1, 2], [0, 0, 0, 1, 3]],
#         CN.tr_count_l_left => [[0, 1, 1, 2, 2], [0, 1, 2, 1, 0]],
#         CN.tr_count_g_right => [[4, 0, 2, 0, 0], [3, 1, 0, 0, 0]],
#         CN.tr_count_l_right => [[0, 3, 0, 1, 0], [0, 1, 2, 1, 0]],
#         CN.tr_contiguous_vars_minus => [[0, 3, 0, 1, 0], [0, 0, 1, 1, 0]],
#         CN.tr_contiguous_vars_minus_rev => [[4, 0, 2, 0, 0], [1, 1, 0, 0, 0]],
#     )

#     for (f, results) in funcs
#         for (key, vals) in enumerate(data)
#             @test f(vals.first) == results[key]
#             foreach(i -> f(i, vals.first), vals.first)
#         end
#     end

#     # Test transformations with parameter
#     funcs_val = Dict(
#         CN.tr_count_eq_val => [[1, 0, 1, 0, 1], [1, 0, 0, 0, 1]],
#         CN.tr_count_l_val => [[2, 5, 3, 5, 4], [4, 5, 5, 5, 4]],
#         CN.tr_count_g_val => [[2, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
#         CN.tr_count_bounding_val => [[3, 1, 3, 2, 3], [5, 3, 1, 3, 5]],
#         CN.tr_var_minus_val => [[0, 3, 0, 2, 1], [0, 0, 1, 0, 0]],
#         CN.tr_val_minus_var => [[1, 0, 0, 0, 0], [1, 0, 0, 0, 1]],
#     )

#     for (f, results) in funcs_val
#         for (key, vals) in enumerate(data)
#             @test f(vals.first; val = vals.second) == results[key]
#             foreach(i -> f(i, vals.first; val = vals.second), vals.first)
#         end
#     end

# end
