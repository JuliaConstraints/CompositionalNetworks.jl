const Arithmetic = LayerCore(
    :Arithmetic,
    true,
    (:(AbstractVector{<:AbstractVector{<:Real}}),) => AbstractVector{<:Real},
    (sum = :((x) -> sum(x)), product = :((x) -> reduce((t...) -> broadcast(*, t...), x)))
)

# SECTION - Docstrings to put back/update
"""
    ar_sum(x)
Reduce `k = length(x)` vectors through sum to a single vector.
"""

"""
    ar_prod(x)
Reduce `k = length(x)` vectors through product to a single vector.
"""

"""
    arithmetic_layer()
Generate the layer of arithmetic operations of the ICN. The operations are mutually exclusive, that is only one will be selected.
"""

## SECTION - Test Items
# @testitem "Arithmetic Layer" tags = [:arithmetic, :layer] begin
#     CN = CompositionalNetworks

#     data = [[1, 5, 2, 4, 3] => 2, [1, 2, 3, 2, 1] => 2]

#     @test CN.ar_sum(map(p -> p.first, data)) == [2, 7, 5, 6, 4]
#     @test CN.ar_prod(map(p -> p.first, data)) == [1, 10, 6, 8, 3]

# end
