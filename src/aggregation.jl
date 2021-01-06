"""
    _ag_sum(x)
Aggregate through `+` a vector into a single scalar.
"""
_ag_sum(x) = reduce(+, x)

"""
    _ag_count_positive(x)
Count the number of strictly positive elements of `x`.
"""
_ag_count_positive(x) = count(y -> y > 0.0, x)

"""
    aggregation_layer()
Generate the layer of aggregation functions of the ICN.
"""
function aggregation_layer()
    aggregations = LittleDict{Symbol, Function}(
        :sum => _ag_sum,
        :count_positive => _ag_count_positive,
    )

    return Layer(aggregations, true)
end
