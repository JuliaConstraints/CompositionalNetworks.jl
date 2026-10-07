function _paired_parameters(pair_vars)
    pair_vars isa AbstractVector || throw(ArgumentError(
        "pairwise geometric operations require a vector of flattened parameters",
    ))
    return pair_vars
end

function _aligned_pair_map(operation, values, pair_vars)
    axes(values) == axes(pair_vars) || throw(DimensionMismatch(
        "aligned values and parameters must have the same axes",
    ))
    return operation.(values, pair_vars)
end

"""Indirect gather through a paired vector, with explicit coordinate origins.

An out-of-range index maps to the value-origin predecessor. It is a data sentinel,
not a constraint penalty; any required index guard remains part of the ICN.
"""
function _paired_gather(indices, values, index_base)
    value_origin, index_origin = index_base isa Tuple ? index_base : (index_base,index_base)
    return map(indices) do index
        j = index - index_origin + 1
        j isa Integer && checkbounds(Bool,values,j) ? values[j] : value_origin - 1
    end
end

function _pairwise_task_count(parameters, dimensions::Int)
    dimensions > 0 || throw(ArgumentError("pairwise dimension must be positive"))
    length(parameters) % dimensions == 0 || throw(DimensionMismatch(
        "pairwise parameters must be divisible by the dimension",
    ))
    Base.require_one_based_indexing(parameters)
    return length(parameters) ÷ dimensions
end

function _check_pairwise_geometry_arguments(values, parameters, dimensions::Int)
    tasks = _pairwise_task_count(parameters, dimensions)
    length(parameters) == length(values) || throw(DimensionMismatch(
        "pairwise values and parameters must have the same flattened length",
    ))
    Base.require_one_based_indexing(values, parameters)
    return tasks
end

_affine_margin_type(::Type{V}, ::Type{P}) where {V, P} =
    Base.promote_op(-, Base.promote_op(+, V, P), V)

"""Signed margins `x[i] + pair_vars[i] - x[i + 1]` on adjacent positions."""
function _adjacent_left_affine_margins(values, pair_vars)
    parameters = _paired_parameters(pair_vars)
    axes(values) == axes(parameters) || throw(DimensionMismatch(
        "adjacent affine values and parameters must have the same axes",
    ))
    Base.require_one_based_indexing(values, parameters)
    output = Vector{_affine_margin_type(eltype(values), eltype(parameters))}(
        undef, max(0, length(values) - 1),
    )
    @inbounds for index in 1:(length(values) - 1)
        output[index] = values[index] + parameters[index] - values[index + 1]
    end
    return output
end

@inline function _pair_is_disabled(
        parameters,
        dimensions::Int,
        first_task::Int,
        second_task::Int,
        zero_ignored::Bool,
)
    zero_ignored || return false
    first_offset = (first_task - 1) * dimensions
    second_offset = (second_task - 1) * dimensions
    @inbounds for dimension in 1:dimensions
        if iszero(parameters[first_offset + dimension]) ||
           iszero(parameters[second_offset + dimension])
            return true
        end
    end
    return false
end

"""Flatten both oriented affine margins for every pair and coordinate."""
function _pairwise_oriented_affine_margins(
        values,
        pair_vars,
        dimensions::Int,
)
    parameters = _paired_parameters(pair_vars)
    tasks = _check_pairwise_geometry_arguments(values, parameters, dimensions)
    output = Vector{_affine_margin_type(eltype(values), eltype(parameters))}(
        undef, dimensions * tasks * (tasks - 1),
    )
    output_index = 1
    @inbounds for first_task in 1:(tasks - 1)
        first_offset = (first_task - 1) * dimensions
        for second_task in (first_task + 1):tasks
            second_offset = (second_task - 1) * dimensions
            for dimension in 1:dimensions
                first_index = first_offset + dimension
                second_index = second_offset + dimension
                output[output_index] = values[first_index] + parameters[first_index] -
                                       values[second_index]
                output[output_index + 1] =
                    values[second_index] + parameters[second_index] - values[first_index]
                output_index += 2
            end
        end
    end
    return output
end

const PairedMap = LayerCore(
    :PairedMap,
    true,
    (:(AbstractVector),) => AbstractVector,
    (
        id = :((x) -> identity(x)),
        sub = :((x; pair_vars) -> abs.(x .- pair_vars)),
        sum = :((x; pair_vars) -> (x .+ pair_vars)),
        prod = :((x; pair_vars) -> (x .* pair_vars)),
        adjacent_left_affine_margins = :(
            (x; pair_vars) -> CompositionalNetworks._adjacent_left_affine_margins(
                x,
                pair_vars,
            )
        ),
        pairwise_oriented_affine_margins = :(
            (x; pair_vars, dim = 1) -> CompositionalNetworks._pairwise_oriented_affine_margins(
                x,
                pair_vars,
                Int(dim),
            )
        ),
        aligned_not_equal = :(
            (x; pair_vars) -> CompositionalNetworks._aligned_pair_map(!=, x, pair_vars)
        ),
        aligned_difference = :(
            (x; pair_vars) -> CompositionalNetworks._aligned_pair_map(-, x, pair_vars)
        ),
        gather = :(
            (x; pair_vars, index_base=1) -> CompositionalNetworks._paired_gather(x,pair_vars,index_base)
        ),
    )
)

@testitem "Pairwise oriented clearances expose generic affine margins" begin
    using Test

    @test PairedMap.fn[:pairwise_oriented_affine_margins](
        [0, 1, 4]; pair_vars = [2, 2, 1], dim = 1, bool = true,
    ) == [1.0, 3.0, -2.0, 5.0, -1.0, 4.0]
    @test PairedMap.fn[:pairwise_oriented_affine_margins](
        [0, 0, 1, 3]; pair_vars = [2, 2, 2, 2], dim = 2, bool = true,
    ) == [1.0, 3.0, -1.0, 5.0]
    @test PairedMap.fn[:pairwise_oriented_affine_margins](
        [0, 0, 1, 1]; pair_vars = [2, 2, 2, 2], dim = 2, bool = true,
    ) == [1.0, 3.0, 1.0, 3.0]
    @test PairedMap.fn[:pairwise_oriented_affine_margins](
        [1, 1]; pair_vars = [0, 2], dim = 1, bool = true,
    ) == [0, 2]
    @test PairedMap.fn[:pairwise_oriented_affine_margins](
        [1, 0]; pair_vars = [0, 2], dim = 1, bool = false,
    ) == [1.0, 1.0]
end

@testitem "Aligned inequality is an atomic typed-pair operation" begin
    using Test

    mismatch = PairedMap.fn[:aligned_not_equal]
    @test mismatch([1, 2, 3]; pair_vars = [1, 7, 3]) == [false, true, false]
    @test mismatch(["a", "b", "c"]; pair_vars = ["a", "c", "c"]) ==
          [false, true, false]
    @test_throws DimensionMismatch mismatch([1, 2]; pair_vars = [1])

    difference = PairedMap.fn[:aligned_difference]
    @test difference([1, -2, 7]; pair_vars = [4, -2, 3]) == [-3, 0, 4]
    @test_throws DimensionMismatch difference([1, 2]; pair_vars = [1])
end

@testitem "Numeric learning transfers through primitive capabilities" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @test, @test_throws

    struct AlgebraicToken
        value::Int
    end
    Base.:-(left::AlgebraicToken, right::AlgebraicToken) = left.value - right.value

    layers = [CN.PairedMap, CN.Pointwise, CN.Arithmetic,
        CN.Aggregation, CN.Comparison]
    network = CN.ICN(;
        parameters = [:pair_vars],
        parameter_values = (; pair_vars = [1, 2, 3]),
        layers,
        connection = UInt32.(eachindex(layers)),
    )
    fill!(network.weights.parent, false)
    selections = (
        (:aligned_difference,), (:absolute,), (:sum,), (:sum,), (:id,),
    )
    offset = Ref(0)
    for (layer, selected) in zip(network.layers, selections)
        names = collect(keys(layer.fn))
        for operation in selected
            network.weights.parent[offset[] + only(findall(==(operation), names))] = true
        end
        offset[] += length(layer.fn)
    end
    compiled = first(CN.compose(network; name = :algebraic_token_distance))
    values = AlgebraicToken.([1, 4, 3])
    targets = AlgebraicToken.([1, 2, 5])
    @test Base.invokelatest(compiled, values; pair_vars = targets) == 4.0
    @test_throws MethodError Base.invokelatest(
        compiled, [:a, :b]; pair_vars = [:a, :c])
end


@testitem "Adjacent affine margins expose a reusable directed stencil" begin
    using Test

    margins = PairedMap.fn[:adjacent_left_affine_margins]
    values = [1, 4, 5, 9]
    lengths = [2, 1, 3, 99]
    @test margins(values; pair_vars = lengths) == [-1.0, 0.0, -1.0]
    @test isempty(margins([1]; pair_vars = [99]))
    @test_throws DimensionMismatch margins(values; pair_vars = lengths[1:3])
end
