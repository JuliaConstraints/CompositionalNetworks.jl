function _aligned_vector_difference(vectors)
    length(vectors) == 2 || throw(DimensionMismatch(
        "an aligned vector difference requires exactly two operands"))
    left, right = vectors
    axes(left) == axes(right) || throw(DimensionMismatch(
        "aligned vector operands must have identical axes"))
    return left .- right
end

# Only dense built-in scalars may use an owned intermediate. Custom arrays,
# arbitrary precision and longer reductions keep Base's original implementation.
const _ArithmeticScalar = Union{Int8,Int16,Int32,Int64,Int128,
    UInt8,UInt16,UInt32,UInt64,UInt128,Float16,Float32,Float64}

_vector_combine(::typeof(+), left, right) = left + right
_vector_combine(::typeof(*), left, right) = broadcast(*, left, right)

function _short_dense_arithmetic(op::F, vectors) where {F}
    # Base processes arrays shorter than 16 with the first pair followed by a
    # left fold. Avoid changing its pairwise/SIMD reduction tree for larger ones.
    vectors isa Vector && 3 <= length(vectors) < 16 || return nothing
    return _short_dense_arithmetic(op, vectors, first(vectors))
end

# Specialize once on the first operand, even when the layer's output container
# is Vector{AbstractVector}. Mixed types (including Bool promotions) fall back.
function _short_dense_arithmetic(op::F, vectors, first_vector::Vector{T}) where {F,T<:_ArithmeticScalar}
    isconcretetype(T) || return nothing
    size = length(first_vector)
    for vector in vectors
        vector isa Vector{T} && length(vector) == size || return nothing
    end
    # The initial pair always creates a fresh result. An input may occur several
    # times in vectors: no input, including an aliased one, may become scratch.
    result = _vector_combine(op, first_vector, vectors[2]::Vector{T})
    for position in 3:length(vectors)
        next = vectors[position]::Vector{T}
        @inbounds for index in eachindex(result, next)
            result[index] = op(result[index], next[index])
        end
    end
    return result
end
_short_dense_arithmetic(op, vectors, first_vector) = nothing

function _arithmetic_sum(vectors)
    result = _short_dense_arithmetic(+, vectors)
    return isnothing(result) ? sum(vectors) : result
end

function _arithmetic_product(vectors)
    result = _short_dense_arithmetic(*, vectors)
    return isnothing(result) ? reduce((t...) -> broadcast(*, t...), vectors) : result
end

const Arithmetic = LayerCore(
    :Arithmetic,
    true,
    (:(AbstractVector{<:AbstractVector}),) => AbstractVector,
    (
        sum = :((x) -> CompositionalNetworks._arithmetic_sum(x)),
        product = :((x) -> CompositionalNetworks._arithmetic_product(x)),
        difference = :((x) -> CompositionalNetworks._aligned_vector_difference(x)),
    )
)

@testitem "Aligned arithmetic difference is a binary atomic reducer" begin
    using Test

    @test Arithmetic.fn[:difference]([[3, 1], [1, 4]]) == [2, -3]
    @test_throws DimensionMismatch Arithmetic.fn[:difference]([[1], [2], [3]])
end

# SECTION - Docstrings to put back/update
"""
    ar_sum(x)
Reduce `k = length(x)` vectors through sum to a single vector.
"""

"""
    ar_prod(x)
Reduce `k = length(x)` vectors through product to a single vector.
"""

"""
    arithmetic_layer()
Generate the layer of arithmetic operations of the ICN. The operations are mutually exclusive, that is only one will be selected.
"""

## SECTION - Test Items
# @testitem "Arithmetic Layer" tags = [:arithmetic, :layer] begin
#     CN = CompositionalNetworks

#     data = [[1, 5, 2, 4, 3] => 2, [1, 2, 3, 2, 1] => 2]

#     @test CN.ar_sum(map(p -> p.first, data)) == [2, 7, 5, 6, 4]
#     @test CN.ar_prod(map(p -> p.first, data)) == [1, 10, 6, 8, 3]

# end
