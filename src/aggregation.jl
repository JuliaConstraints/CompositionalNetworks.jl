"""
    _sum(x::V)
Aggregate through `+` a vector into a single scalar.
"""
_sum(x::V) where {T <: Number, V <: AbstractVector{T}} = reduce(+, x)

"""
    _count_positive(x::V)
Count the number of strictly positive elements of `x`.
"""
_count_positive(x::V) where {T <: Number, V <: AbstractVector{T}} = count(y -> y > 0.0, x)