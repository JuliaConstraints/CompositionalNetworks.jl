function _group_minimum(values, width::Int)
    width > 0 || throw(ArgumentError("group width must be positive"))
    groups = cld(length(values), width)
    output = Vector{eltype(values)}(undef, groups)
    @inbounds for group in 1:groups
        first_index = (group - 1) * width + 1
        last_index = min(group * width, length(values))
        value = values[first_index]
        for index in (first_index + 1):last_index
            value = min(value, values[index])
        end
        output[group] = value
    end
    return output
end

"""Reductions over explicit consecutive groups, without pointwise post-processing."""
const GroupReduction = LayerCore(
    :GroupReduction,
    true,
    (:(AbstractVector),) => AbstractVector,
    (
        id = :((x) -> identity(x)),
        minimum = :(
            (x; dim = 1) -> CompositionalNetworks._group_minimum(x, 2 * Int(dim))
        ),
    ),
)

@testitem "Group minimum is a single axis reduction" begin
    using Test

    minimums = GroupReduction.fn[:minimum]
    @test GroupReduction.fn[:id]([1, 2]) == [1, 2]
    @test minimums([1, 3, -1, 5]; dim = 1) == [1, -1]
    @test minimums([1, 3, 1, 3]; dim = 2) == [1]
    @test minimums([1, 2, 3]; dim = 1) == [1, 3]
end
