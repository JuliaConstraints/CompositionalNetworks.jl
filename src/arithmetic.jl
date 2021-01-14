"""
    _ar_sum(x)
Reduce `k = length(x)` vectors through sum to a single vector.
"""
_ar_sum(x) = reduce((y, z) -> y .+ z, x)

"""
    _ar_prod(x)
Reduce `k = length(x)` vectors through product to a single vector.
"""
_ar_prod(x) = reduce((y, z) -> y .* z, x)

"""
    arithmetic_layer()
Generate the layer of arithmetic operations of the ICN. The operations are mutually exclusive, that is only one will be selected.
"""
function arithmetic_layer()
    arithmetics = LittleDict{Symbol, Function}(
        :sum => _ar_sum,
        :prod => _ar_prod,
    )

    return Layer(arithmetics, true)
end
