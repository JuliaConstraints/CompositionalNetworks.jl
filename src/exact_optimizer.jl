"""
    JuMPExactOptimizer(optimizer_factory; kwargs...)

Complete ICN structure optimizer backed by a JuMP MILP solver. The MILP master
enforces layer cardinalities and enumerates structurally valid weight vectors in
increasing regularization order. ICN outputs remain generic Julia functions and
are evaluated outside the master; a no-good cut then removes each candidate.

`telemetry[:exact]` is true only when the backend proves the search complete or
the non-negative training error reaches zero at the master's global lower bound.
"""
struct JuMPExactOptimizer <: AbstractOptimizer
    optimizer_factory::Any
    time_limit::Float64
    max_candidates::Int
    enumerate_training_exact::Bool
    silent::Bool
    attributes::Vector{Pair{Any, Any}}
    telemetry::Dict{Symbol, Any}
end

function JuMPExactOptimizer(
        optimizer_factory;
        time_limit = Inf,
        max_candidates = typemax(Int),
        enumerate_training_exact = false,
        silent = true,
        attributes = Pair[],
        telemetry = Dict{Symbol, Any}()
)
    limit = Float64(time_limit)
    (isinf(limit) || limit >= 0) ||
        throw(ArgumentError("time_limit must be non-negative or Inf"))
    max_candidates > 0 || throw(ArgumentError("max_candidates must be positive"))
    optimizer_attributes = Pair{Any, Any}[Pair{Any, Any}(first(attribute), last(attribute))
                                          for attribute in attributes]
    return JuMPExactOptimizer(
        optimizer_factory,
        limit,
        Int(max_candidates),
        Bool(enumerate_training_exact),
        Bool(silent),
        optimizer_attributes,
        telemetry
    )
end

@testitem "JuMP exact optimizer proves and reports ICN learning results" tags=[:extension] default_imports=false begin
    import CompositionalNetworks as CN
    import ConstraintDomains: domain
    import HiGHS
    import JuMP
    import Test: @test

    all_different(values) = allunique(values)
    domains=[domain(1:3) for _ in 1:3]
    network=CN.ICN(parameters = [:dom_size, :numvars])
    optimizer=CN.JuMPExactOptimizer(
        HiGHS.Optimizer;
        max_candidates = 5_000,
        silent = true
    )

    learned=CN.explore_learn(
        domains,
        all_different,
        optimizer;
        icn = network,
        metric_function = CN.hamming
    )

    @test last(learned)
    @test optimizer.telemetry[:exact]
    @test optimizer.telemetry[:best_training_error] == 0
    @test optimizer.telemetry[:candidates_evaluated] <= 5_000
    @test CN.check_weights_validity(first(learned), first(learned).weights)

    extension=Base.get_extension(CN, :JuMPExactExt)
    coefficients=extension._tie_break_coefficients(first(learned))
    candidate=BitVector(first(learned).weights)
    candidate[1:2].=true
    @test CN.apply!(first(learned), candidate)
    @test sum(coefficients .* candidate) ≈
          CN.weights_bias(candidate) + CN.regularization(first(learned))

    limited=CN.JuMPExactOptimizer(
        HiGHS.Optimizer;
        max_candidates = 1,
        silent = true
    )
    CN.explore_learn(
        domains,
        all_different,
        limited;
        icn = CN.ICN(parameters = [:dom_size, :numvars]),
        metric_function = CN.hamming
    )
    @test limited.telemetry[:candidates_evaluated] == 1
    @test !limited.telemetry[:exact]
    @test limited.telemetry[:termination_reason] == :candidate_limit
end
