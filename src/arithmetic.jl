"""
    _sum(x::V...)
    _prod(x::V...)
Reduce `k = length(x)` vectors through sum/product to a single vector.
"""
_sum(x::V...) where {T <: Number, V <: AbstractVector{T}} = reduce((y, z) -> y .+ z, x)
_prod(x::V...) where {T <: Number, V <: AbstractVector{T}} = reduce((y, z) -> y .* z, x)
