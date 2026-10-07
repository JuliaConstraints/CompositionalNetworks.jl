"""Unary pointwise operations used after a vector arithmetic stage."""
const Pointwise = LayerCore(
    :Pointwise,
    true,
    (:(AbstractVector),) => AbstractVector,
    (
        id = :((x) -> identity(x)),
        absolute = :((x) -> abs.(x)),
        positive_part = :((x) -> map(value -> max(zero(value), value), x)),
    ),
)

@testitem "Post-arithmetic pointwise operations stay unary" begin
    using Test

    @test Pointwise.fn[:id]([-2, 0, 3]) == [-2, 0, 3]
    @test Pointwise.fn[:absolute]([-2, 0, 3]) == [2, 0, 3]
    @test Pointwise.fn[:positive_part]([-2, 0, 3]) == [0, 0, 3]
end
