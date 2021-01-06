"""
    _sum(x::V)
Aggregate through `+` a vector into a single scalar.
"""
_ag_sum(x) = reduce(+, x)

"""
    _count_positive(x::V)
Count the number of strictly positive elements of `x`.
"""
_count_positive(x::V) where {T <: Number, V <: AbstractVector{T}} = count(y -> y > 0.0, x)

"""
    aggregation_layer()
Generate the layer of aggregation functions of the ICN.
"""
function aggregation_layer()
    aggregations = LittleDict{Symbol, Function}(
        :sum => _ag_sum,
        :count_positive => _count_positive,
    )

    return Layer(aggregations, true)
end
