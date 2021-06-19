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
    aggregations = LittleDict{Symbol, Function}(
        :sum => ag_sum,
        :count_positive => ag_count_positive,
    )

    return Layer(aggregations, true)
end
