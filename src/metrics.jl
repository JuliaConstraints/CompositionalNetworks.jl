"""
    hamming(x, X)
Compute the hamming distance of `x` over a collection of solutions `X`, i.e. the minimal number of variables to switch in `x`to reach a solution.
"""
hamming(x, X) = mapreduce(y -> Distances.hamming(x, y), min, X)

"""Count concept/penalty mismatches, without imposing a numerical distance target.

Solutions require a finite zero, non-solutions a finite strictly positive penalty.
Structural invalidity is not a candidate penalty function. This metric permits
constructive feasibility learning independently of Hamming or Manhattan accuracy.
"""
function zero_set_loss(icn::AbstractICN, configurations::Configurations, solution_vector = nothing;
        weights_validity::Bool = true, parameters...)
    weights_validity || return Inf
    loss = 0
    for config in configurations
        cost = evaluate(icn, config; parameters...)
        correct = isfinite(cost) && (config isa Solution ? iszero(cost) : cost > 0)
        loss += !correct
    end
    return Float64(loss)
end

function hamming(
        icn::AbstractICN,
        configurations::Configurations,
        solution_vector;
        weights_validity::Bool = true,
        parameters...
)
    weights_validity || return Inf
    sum(
        x -> abs(evaluate(icn, x; parameters...) - hamming(x.x, solution_vector)),
        configurations
    )
end

"""
    minkowski(x, X, p)
"""
minkowski(x, X, p) = mapreduce(y -> Distances.minkowski(x, y, p), min, X)

"""
    manhattan(x, X)
"""
manhattan(x, X) = mapreduce(y -> Distances.cityblock(x, y), min, X)

function manhattan(
        icn::AbstractICN,
        configurations::Configurations,
        solution_vector;
        weights_validity::Bool = true,
        parameters...
)
    weights_validity || return Inf
    sum(
        x -> abs(evaluate(icn, x; parameters...) - manhattan(x.x, solution_vector)),
        configurations
    ) / (get(icn.constants, :dom_size, 2) - 1)
end

"""
    weights_bias(x)
A metric that bias `x` towards operations with a lower bit. Do not affect the main metric.
"""
weights_bias(x) = sum(p -> p[1] * log2(1.0 + p[2]), enumerate(x)) / length(x)^4

@testitem "Learning metrics reject structurally invalid candidates before evaluation" begin
    using Test

    configurations = Set{Configuration}([
        Solution([1, 2]),
        NonSolution([1, 1]),
    ])
    network = ICN()
    solution_vector = [[1, 2]]
    @test isinf(hamming(
        network, configurations, solution_vector; weights_validity = false,
    ))
    @test isinf(manhattan(
        network, configurations, solution_vector; weights_validity = false,
    ))
end
