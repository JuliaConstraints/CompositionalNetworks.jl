"""
    _sum(x::W) where {T <: Number, V <: AbstractVector{T}, W <: AbstractVector{V}}
    _prod(x::W) where {T <: Number, V <: AbstractVector{T}, W <: AbstractVector{V}}
Reduce `k = length(x)` vectors through sum/product to a single vector.
"""
function _sum(x::W) where {T <: Number, V <: AbstractVector{T}, W <: AbstractVector{V}}
    return reduce((y, z) -> y .+ z, x)
end
function _prod(x::W) where {T <: Number, V <: AbstractVector{T}, W <: AbstractVector{V}}
    return reduce((y, z) -> y .* z, x)
end
