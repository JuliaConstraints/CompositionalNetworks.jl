"""
    hamming(x, X)
Compute the hamming distance of `x` over a collection of solutions `X`, i.e. the minimal number of variables to switch in `x`to reach a solution.
"""
hamming(x, X) = mapreduce(y -> Distances.hamming(x, y), min, X)

function hamming(
        icn::AbstractICN,
        configurations::Configurations,
        solution_vector;
        parameters...
)
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
        parameters...
)
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
