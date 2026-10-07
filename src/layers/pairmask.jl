function _zero_extent_group_mask(values, pair_vars, dimensions::Int, enabled::Bool)
    parameters = _paired_parameters(pair_vars)
    tasks = _pairwise_task_count(parameters, dimensions)
    group_width = 2dimensions
    expected = group_width * (tasks * (tasks - 1) ÷ 2)
    length(values) == expected || throw(DimensionMismatch(
        "pair mask input does not match the pairwise dimension layout",
    ))
    output = collect(values)
    enabled || return output
    group = 0
    @inbounds for first_task in 1:(tasks - 1), second_task in (first_task + 1):tasks
        group += 1
        _pair_is_disabled(
            parameters, dimensions, first_task, second_task, true,
        ) || continue
        first_index = (group - 1) * group_width + 1
        fill!(@view(output[first_index:(first_index + group_width - 1)]),
            zero(eltype(output)))
    end
    return output
end

"""Structural masks over flattened pairwise groups."""
const PairMask = LayerCore(
    :PairMask,
    true,
    (:(AbstractVector),) => AbstractVector,
    (
        id = :((x) -> identity(x)),
        zero_extent_groups = :(
            (x; pair_vars, dim = 1, bool = true) -> CompositionalNetworks._zero_extent_group_mask(
                x, pair_vars, Int(dim), Bool(bool),
            )
        ),
    ),
)

@testitem "Pair masking is independent from margin and reduction operations" begin
    using Test

    mask = PairMask.fn[:zero_extent_groups]
    @test mask([1.0, 2.0]; pair_vars = [2, 3], dim = 1, bool = true) == [1.0, 2.0]
    @test mask([1.0, 2.0]; pair_vars = [0, 3], dim = 1, bool = true) == [0.0, 0.0]
    @test mask([1.0, 2.0]; pair_vars = [0, 3], dim = 1, bool = false) == [1.0, 2.0]
end
