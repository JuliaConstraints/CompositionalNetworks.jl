@inline _condition_operator(::typeof(==)) = Val(:eq)
@inline _condition_operator(::typeof(!=)) = Val(:ne)
@inline _condition_operator(::typeof(<=)) = Val(:le)
@inline _condition_operator(::typeof(>=)) = Val(:ge)
@inline _condition_operator(::typeof(<)) = Val(:lt)
@inline _condition_operator(::typeof(>)) = Val(:gt)
@inline _condition_operator(::typeof(in)) = Val(:in)
@inline _condition_operator(op) = Val(:generic)

@inline _condition_residual(lhs::Real, rhs::Real, ::Val{:eq}) = Float64(abs(lhs - rhs))
@inline _condition_residual(lhs, rhs, ::Val{:eq}) = lhs == rhs ? 0.0 : 1.0
@inline _condition_residual(lhs, rhs, ::Val{:ne}) = lhs == rhs ? 1.0 : 0.0
@inline _condition_residual(lhs::Real, rhs::Real, ::Val{:le}) =
    Float64(max(zero(lhs - rhs), lhs - rhs))
@inline _condition_residual(lhs::Real, rhs::Real, ::Val{:ge}) =
    Float64(max(zero(rhs - lhs), rhs - lhs))
@inline function _condition_residual(lhs::Real, rhs::Real, ::Val{:lt})
    lhs < rhs && return 0.0
    lhs != rhs && return Float64(lhs - rhs)
    return lhs isa Integer && rhs isa Integer ? 1.0 :
           eps(max(abs(Float64(lhs)), abs(Float64(rhs)), 1.0))
end
@inline function _condition_residual(lhs::Real, rhs::Real, ::Val{:gt})
    lhs > rhs && return 0.0
    lhs != rhs && return Float64(rhs - lhs)
    return lhs isa Integer && rhs isa Integer ? 1.0 :
           eps(max(abs(Float64(lhs)), abs(Float64(rhs)), 1.0))
end
@inline function _condition_residual(lhs::Real, rhs, ::Val{:in})
    lhs in rhs && return 0.0
    isempty(rhs) && return 1.0
    distance = Inf
    @inbounds for target in rhs
        target isa Real || return 1.0
        distance = min(distance, abs(Float64(lhs) - Float64(target)))
    end
    return distance
end
@inline _condition_residual(lhs, rhs, ::Val{:in}) = lhs in rhs ? 0.0 : 1.0
@inline _condition_residual(lhs, rhs, ::Val{:le}) = lhs <= rhs ? 0.0 : 1.0
@inline _condition_residual(lhs, rhs, ::Val{:ge}) = lhs >= rhs ? 0.0 : 1.0
@inline _condition_residual(lhs, rhs, ::Val{:lt}) = lhs < rhs ? 0.0 : 1.0
@inline _condition_residual(lhs, rhs, ::Val{:gt}) = lhs > rhs ? 0.0 : 1.0
@inline _condition_residual(lhs, rhs, ::Val{:generic}, op) =
    op(lhs, rhs) ? 0.0 : 1.0

@inline function _condition_residual(lhs, rhs, op::F) where {F}
    operator = _condition_operator(op)
    operator isa Val{:generic} && return _condition_residual(lhs, rhs, operator, op)
    return _condition_residual(lhs, rhs, operator)
end

function _block_layout(values, dim::Integer)
    Base.require_one_based_indexing(values)
    dim > 0 || throw(ArgumentError("the number of index blocks must be positive"))
    length(values) % dim == 0 || throw(DimensionMismatch(
        "the input length must be divisible by the number of index blocks"))
    return Int(dim), length(values) ÷ Int(dim)
end

"""Values gathered through indices stored in cyclic, equally-sized blocks."""
function _cyclic_indirect_values(values, dim::Integer; index_base=1)
    blocks, width = _block_layout(values, dim)
    output = fill(convert(eltype(values),index_base-1), length(values))
    @inbounds for block in 0:(blocks - 1)
        source_offset = block * width
        target_offset = ((block + 1) % blocks) * width
        for local_index in 1:width
            indirect_index = values[source_offset + local_index] - index_base + 1
            if indirect_index isa Integer && 1 ≤ indirect_index ≤ width
                output[source_offset + local_index] =
                    values[target_offset + Int(indirect_index)]
            end
        end
    end
    return output
end

"""One-based local coordinates repeated over equally-sized blocks."""
function _block_local_indices(values, dim::Integer; index_base=1)
    blocks, width = _block_layout(values, dim)
    output = Vector{Int}(undef, length(values))
    @inbounds for block in 0:(blocks - 1), local_index in 1:width
        origin = index_base isa Tuple ? first(index_base) : index_base
        output[block * width + local_index] = local_index + origin - 1
    end
    return output
end

"""Per-node predecessor counts in a one-based functional graph."""
function _predecessor_counts(values; index_base=1)
    Base.require_one_based_indexing(values)
    counts = zeros(Int, length(values))
    @inbounds for successor in values
        successor = successor - index_base + 1
        successor isa Integer && 1 <= successor <= length(values) || continue
        counts[Int(successor)] += 1
    end
    return counts
end

"""Indicators of non-fixed nodes excluded from the orbit of the first non-fixed node."""
function _orbit_exclusion_indicators(values)
    Base.require_one_based_indexing(values)
    output = zeros(Int, length(values))
    first_active = findfirst(index -> values[index] != index, eachindex(values))
    if isnothing(first_active)
        isempty(output) || (output[firstindex(output)] = 1)
        return output
    end

    visited = falses(length(values))
    current = first_active
    while current isa Integer && 1 <= current <= length(values) &&
          !visited[Int(current)] && values[Int(current)] != current
        visited[Int(current)] = true
        current = values[Int(current)]
    end
    @inbounds for index in eachindex(values)
        output[index] = values[index] != index && !visited[index]
    end
    return output
end

"""Indicator of the pointwise relation `value != one-based position`."""
function _nonfixed_indicators(values; index_base=1)
    Base.require_one_based_indexing(values)
    return map(index -> values[index] != index + index_base - 1, eachindex(values))
end

"""Signed differences on the disjoint adjacent pairs `(1,2), (3,4), ...`."""
function _disjoint_pair_differences(values)
    Base.require_one_based_indexing(values)
    difference_type = Base.promote_op(-, eltype(values), eltype(values))
    output = Vector{difference_type}(undef, length(values) ÷ 2)
    @inbounds for pair in eachindex(output)
        left = 2pair - 1
        output[pair] = values[left] - values[left + 1]
    end
    return output
end

const Transformation = LayerCore(
    :Transformation,
    false,
    (:(AbstractVector),) => AbstractVector,
    (
        id = :((x) -> identity(x)),
        absolute = :((x) -> abs.(x)),
        positive_part = :((x) -> map(value -> max(zero(value), value), x)),
        index_indicator = :((x; id) -> map(==(id), eachindex(x))),
        cyclic_indirect_values = :(
            (x; dim, index_base=1) -> CompositionalNetworks._cyclic_indirect_values(x, Int(dim);index_base)
        ),
        block_local_indices = :(
            (x; dim, index_base=1) -> CompositionalNetworks._block_local_indices(x, Int(dim);index_base)
        ),
        predecessor_counts = :(
            (x; index_base=1) -> CompositionalNetworks._predecessor_counts(x;index_base)
        ),
        orbit_exclusion_indicators = :(
            (x) -> CompositionalNetworks._orbit_exclusion_indicators(x)
        ),
        nonfixed_indicators = :(
            (x; index_base=1) -> CompositionalNetworks._nonfixed_indicators(x;index_base)
        ),
        disjoint_pair_differences = :(
            (x) -> CompositionalNetworks._disjoint_pair_differences(x)
        ),
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
        count_equal_val = :(
            (x; val) -> map(
                i -> count(j -> j != i && x[j] == x[i] + val, eachindex(x)),
                eachindex(x),
            )
        ),
        count_less_val = :(
            (x; val) -> map(
                i -> count(j -> j != i && x[j] < x[i] + val, eachindex(x)),
                eachindex(x),
            )
        ),
        count_great_val = :(
            (x; val) -> map(
                i -> count(j -> j != i && x[j] > x[i] + val, eachindex(x)),
                eachindex(x),
            )
        ),
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
        count_equal = :(
            (x) -> map(
                i -> count(j -> j != i && x[j] == x[i], eachindex(x)),
                eachindex(x),
            )
        ),
        count_less = :(
            (x) -> map(
                i -> count(j -> j != i && x[j] < x[i], eachindex(x)),
                eachindex(x),
            )
        ),
        count_great = :(
            (x) -> map(
                i -> count(j -> j != i && x[j] > x[i], eachindex(x)),
                eachindex(x),
            )
        ),
        count_bounding_val = :(
            (x; val) -> map(
                i -> count(
                    j -> j != i && x[i] <= x[j] <= x[i] + val,
                    eachindex(x),
                ),
                eachindex(x),
            )
        ),
        var_minus_vals = :((x; vals) -> map(i -> max(0, (i .- vals)...), x)),
        vals_minus_var = :((x; vals) -> map(i -> max(0, (vals .- i)...), x)),
        nonzero = :((x) -> filter(!iszero, x)),
        first_equal_position = :(
            (x; val) -> [something(findfirst(==(val), x), length(x) + 1)]
        ),
        # The existing scalar comparator, lifted pointwise. No constraint semantics.
        condition_residuals = :(
            (x; op, val) -> CompositionalNetworks._condition_residual.(x, Ref(val), Ref(op))
        ),
    )
)

@testitem "Paper transformation counts exclude the current variable" begin
    using Test

    values = [4, 2, 0]
    @test Transformation.fn[:count_equal](values) == [0, 0, 0]
    @test Transformation.fn[:count_equal_val]([1, 1, 2]; val = 0) == [1, 1, 0]
    @test Transformation.fn[:count_less_val](values; val = 2) == [2, 1, 0]
    @test Transformation.fn[:count_great_val]([0, 2, 4]; val = -1) == [2, 1, 0]
    @test Transformation.fn[:count_bounding_val]([0, 1, 4]; val = 2) == [1, 0, 0]
end


@testitem "Functional-graph transformations remain separate generic features" begin
    using Test

    predecessor = Transformation.fn[:predecessor_counts]
    exclusions = Transformation.fn[:orbit_exclusion_indicators]
    nonfixed = Transformation.fn[:nonfixed_indicators]

    @test predecessor([2, 3, 1, 4]) == [1, 1, 1, 1]
    @test predecessor([4, 3, 1, 3]) == [1, 0, 2, 1]
    @test predecessor([2, 3, 5, 1]) == [1, 1, 1, 0]
    @test exclusions([2, 3, 1, 4]) == [0, 0, 0, 0]
    @test exclusions([2, 1, 4, 3]) == [0, 0, 1, 1]
    @test exclusions([1, 2, 3, 4]) == [1, 0, 0, 0]
    @test nonfixed([2, 3, 1, 4]) == [true, true, true, false]
end

@testitem "Absolute transformation is unary and elementwise" begin
    using Test

    @test Transformation.fn[:absolute]([-3, 0, 2]) == [3, 0, 2]
    @test Transformation.fn[:positive_part]([-3, 0, 2]) == [0, 0, 2]
end

@testitem "Disjoint-pair differences are one local combinatorial stencil" begin
    using Test

    pair_difference = Transformation.fn[:disjoint_pair_differences]
    @test pair_difference([1, 4, -2, 3]) == [-3, -5]
    @test isempty(pair_difference(Int[]))
    @test pair_difference([1, 2, 3]) == [-1]
end

@testitem "Index transformations expose generic one-based relations" begin
    using Test

    @test Transformation.fn[:index_indicator]([9, 8, 7]; id = 2) ==
          [false, true, false]
    @test Transformation.fn[:cyclic_indirect_values]([2, 1, 4, 3]; dim = 1) ==
          [1, 2, 3, 4]
    @test Transformation.fn[:cyclic_indirect_values]([2, 1, 2, 1]; dim = 2) ==
          [1, 2, 1, 2]
    @test Transformation.fn[:cyclic_indirect_values]([0, 4, 2, 1]; dim = 2) ==
          [0, 0, 4, 0]
    @test Transformation.fn[:block_local_indices]([9, 8, 7, 6]; dim = 2) ==
          [1, 2, 1, 2]
    @test_throws DimensionMismatch Transformation.fn[:block_local_indices](
        [1, 2, 3]; dim = 2)
end

@testitem "Scalar condition residuals cover collection targets" begin
    using Test

    @test CompositionalNetworks._condition_residual(4, 1:3, in) == 1.0
    @test CompositionalNetworks._condition_residual(7, 1:3, in) == 4.0
    @test CompositionalNetworks._condition_residual(2, 1:3, in) == 0.0
    @test CompositionalNetworks._condition_residual(:blue, :blue, ==) == 0.0
    @test CompositionalNetworks._condition_residual(:blue, :red, ==) == 1.0
    @test CompositionalNetworks._condition_residual(:blue, (:red, :blue), in) == 0.0
end

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
