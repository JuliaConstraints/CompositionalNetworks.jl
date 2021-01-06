"""
    _sum(x::W) where {T <: Number, V <: AbstractVector{T}, W <: AbstractVector{V}}
    _prod(x::W) where {T <: Number, V <: AbstractVector{T}, W <: AbstractVector{V}}
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
