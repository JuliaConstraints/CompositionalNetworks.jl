"""
    _ar_sum(x)
    _ar_prod(x)
Reduce `k = length(x)` vectors through sum/product to a single vector.
"""
_ar_sum(x) = reduce((y, z) -> y .+ z, x)
_ar_prod(x) = reduce((y, z) -> y .* z, x)

"""
    arithmetic_layer()
Generate the layer of arithmetic functions of the ICN.
"""
function arithmetic_layer()
    arithmetics = LittleDict{Symbol, Function}(
        :sum => _ar_sum,
        :prod => _ar_prod,
    )

    return Layer(arithmetics, true)
end
