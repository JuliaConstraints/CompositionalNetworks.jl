const Aggregation = LayerCore(
    :Aggregation,
    true,
    (:(AbstractVector{<:Real}),) => T where {T<:Real},
    (
        sum=:((x) -> sum(x)),
        count_positive=:((x) -> count(i -> i > 0, x)),
        count_op_val=:((x; val, op) -> count(i -> op(i, val), x)),
        maximum=:((x) -> isempty(x) ? typemax(eltype(x)) : maximum(x)),
        minimum=:((x) -> isempty(x) ? typemax(eltype(x)) : minimum(x)),
    )
)

# SECTION - Docstrings to put back/update
"""
    ag_sum(x)
Aggregate through `+` a vector into a single scalar.
"""

"""
    ag_count_positive(x)
Count the number of strictly positive elements of `x`.
"""

"""
    aggregation_layer()
Generate the layer of aggregations of the ICN. The operations are mutually exclusive, that is only one will be selected.
"""

## SECTION - Test Items
# @testitem "Aggregation Layer" tags = [:aggregation, :layer] begin
#     CN = CompositionalNetworks

#     data = [[1, 5, 2, 4, 3] => 2, [1, 2, 3, 2, 1] => 2]

#     @test CN.ag_sum(data[1].first) == 15
#     @test CN.ag_sum(data[2].first) == 9

#     @test CN.ag_count_positive(data[1].first) == 5
#     @test CN.ag_count_positive(data[2].first) == 5
#     @test CN.ag_count_positive([1, 0, 1, 0, 1]) == 3
# end
