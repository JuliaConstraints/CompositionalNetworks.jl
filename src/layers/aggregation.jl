"""
    ag_sum(x)
Aggregate through `+` a vector into a single scalar.
"""
ag_sum(x) = sum(x)

"""
    ag_count_positive(x)
Count the number of strictly positive elements of `x`.
"""
ag_count_positive(x) = count(y -> y > 0.0, x)

"""
    aggregation_layer()
Generate the layer of aggregations of the ICN. The operations are mutually exclusive, that is only one will be selected.
"""
function aggregation_layer()
    aggregations =
        LittleDict{Symbol,Function}(:sum => ag_sum, :count_positive => ag_count_positive)

    return Layer(true, aggregations, Vector{Symbol}())
end

## SECTION - Test Items
@testitem "Aggregation Layer" tags = [:aggregation, :layer] begin
    CN = CompositionalNetworks

    data = [[1, 5, 2, 4, 3] => 2, [1, 2, 3, 2, 1] => 2]

    @test CN.ag_sum(data[1].first) == 15
    @test CN.ag_sum(data[2].first) == 9

    @test CN.ag_count_positive(data[1].first) == 5
    @test CN.ag_count_positive(data[2].first) == 5
    @test CN.ag_count_positive([1, 0, 1, 0, 1]) == 3
end
