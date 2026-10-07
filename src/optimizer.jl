abstract type AbstractOptimizer end

function optimize!(icn, configurations, metric_function, optimizer_config; parameters...)
    error("No backend loaded")
end

"""Exhaustively enumerate structurally valid ICNs without a solver master.

This backend is intended for small grammar slices. It enumerates the Cartesian
choices of mutex layers and non-empty subsets of non-mutex layers, stopping at
the first zero of a non-negative learning metric. Unlike `JuMPExactOptimizer`,
it has no solve/restart overhead between two Boolean weight vectors.
"""
struct StructuralEnumerationOptimizer <: AbstractOptimizer
    time_limit::Float64
    max_candidates::Int
    max_selected_per_nonmutex::Int
    enumerate_training_exact::Bool
    telemetry::Dict{Symbol, Any}
end

function StructuralEnumerationOptimizer(;
        time_limit = Inf,
        max_candidates = typemax(Int),
        max_selected_per_nonmutex = typemax(Int),
        enumerate_training_exact = false,
        telemetry = Dict{Symbol, Any}(),
)
    limit = Float64(time_limit)
    (isinf(limit) || limit >= 0) || throw(ArgumentError(
        "time_limit must be non-negative or Inf",
    ))
    max_candidates > 0 || throw(ArgumentError("max_candidates must be positive"))
    max_selected_per_nonmutex > 0 || throw(ArgumentError(
        "max_selected_per_nonmutex must be positive",
    ))
    return StructuralEnumerationOptimizer(
        limit,
        Int(max_candidates),
        Int(max_selected_per_nonmutex),
        Bool(enumerate_training_exact),
        telemetry,
    )
end

function optimize!(
        icn::AbstractICN,
        configurations::Configurations,
        metric_function::Union{Function, Vector{Function}},
        optimizer::StructuralEnumerationOptimizer;
        candidate_observer = nothing,
        parameters...,
)
    started_at = time()
    deadline = isfinite(optimizer.time_limit) ? started_at + optimizer.time_limit : Inf
    solution_vector = [configuration.x for configuration in solutions(configurations)]
    isempty(solution_vector) && throw(ArgumentError(
        "structural enumeration requires at least one solution",
    ))
    candidate = falses(length(icn.weights))
    offsets = Vector{Int}(undef, length(icn.layers))
    offset = 1
    for index in eachindex(icn.layers)
        offsets[index] = offset
        offset += icn.weightlen[index]
    end

    best_weights = nothing
    best_training_error = Inf
    best_objective = Inf
    candidates_evaluated = 0
    structurally_incompatible = 0
    stop = false
    termination_reason = :exhausted

    function evaluate_candidate!()
        if candidates_evaluated >= optimizer.max_candidates
            stop = true
            termination_reason = :candidate_limit
            return
        elseif time() >= deadline
            stop = true
            termination_reason = :time_limit
            return
        end
        structural_valid = apply!(icn, candidate)
        if !structural_valid
            candidates_evaluated += 1
            structurally_incompatible += 1
            isnothing(candidate_observer) || candidate_observer((;
                weights = copy(candidate),
                structural_valid = false,
                training_error = Inf,
                objective = Inf,
            ))
            return
        end
        training_error = Float64(training_metric_error(
            icn,
            configurations,
            solution_vector,
            metric_function;
            weights_validity = true,
            parameters...,
        ))
        isfinite(training_error) && training_error >= 0 || throw(ArgumentError(
            "structural enumeration requires a finite non-negative metric",
        ))
        objective = training_error + weights_bias(candidate) + regularization(icn)
        candidates_evaluated += 1
        isnothing(candidate_observer) || candidate_observer((;
            weights = copy(candidate),
            structural_valid = true,
            training_error,
            objective,
        ))
        if objective < best_objective
            best_weights = copy(candidate)
            best_training_error = training_error
            best_objective = objective
        end
        if iszero(training_error) && !optimizer.enumerate_training_exact
            stop = true
            termination_reason = :zero_error
        end
        return
    end

    function choose_subset!(layer_index, start, count, remaining, next_position)
        stop && return
        if iszero(remaining)
            visit_layer!(layer_index + 1)
            return
        end
        last_position = count - remaining + 1
        for position in next_position:last_position
            candidate[start + position - 1] = true
            choose_subset!(layer_index, start, count, remaining - 1, position + 1)
            candidate[start + position - 1] = false
            stop && return
        end
        return
    end

    function visit_layer!(layer_index)
        stop && return
        if layer_index > length(icn.layers)
            evaluate_candidate!()
            return
        end
        start = offsets[layer_index]
        count = icn.weightlen[layer_index]
        layer = icn.layers[layer_index]
        if layer.mutex
            for position in 1:count
                candidate[start + position - 1] = true
                visit_layer!(layer_index + 1)
                candidate[start + position - 1] = false
                stop && return
            end
        else
            maximum_selected = min(count, optimizer.max_selected_per_nonmutex)
            for selected in 1:maximum_selected
                choose_subset!(layer_index, start, count, selected, 1)
                stop && return
            end
        end
        return
    end

    visit_layer!(1)
    isnothing(best_weights) && error(
        "structural enumeration did not evaluate a candidate " *
        "(termination: $termination_reason)",
    )
    structural_valid = apply!(icn, best_weights)
    validity = structural_valid && iszero(best_training_error)
    empty!(optimizer.telemetry)
    optimizer.telemetry[:backend] = :structural_enumeration
    optimizer.telemetry[:candidates_evaluated] = candidates_evaluated
    optimizer.telemetry[:structurally_incompatible] = structurally_incompatible
    optimizer.telemetry[:time_limit] = optimizer.time_limit
    optimizer.telemetry[:time_limit_reached] = termination_reason === :time_limit
    optimizer.telemetry[:max_candidates] = optimizer.max_candidates
    optimizer.telemetry[:candidate_limit_reached] =
        termination_reason === :candidate_limit
    optimizer.telemetry[:max_selected_per_nonmutex] =
        optimizer.max_selected_per_nonmutex
    optimizer.telemetry[:enumerate_training_exact] = optimizer.enumerate_training_exact
    optimizer.telemetry[:best_training_error] = best_training_error
    optimizer.telemetry[:best_objective] = best_objective
    optimizer.telemetry[:exact] = validity || termination_reason === :exhausted
    optimizer.telemetry[:termination_reason] = termination_reason
    optimizer.telemetry[:elapsed_seconds] = time() - started_at
    return icn => validity
end

function training_metric_error(
        icn,
        configurations,
        solution_vector,
        metric_function;
        weights_validity::Bool,
        parameters...
)
    if metric_function isa Function
        return metric_function(
            icn,
            configurations,
            solution_vector;
            weights_validity,
            parameters...
        )
    end
    return minimum(metric_function) do metric
        metric(
            icn,
            configurations,
            solution_vector;
            weights_validity,
            parameters...
        )
    end
end

function training_metric_valid(args...; weights_validity::Bool, parameters...)
    weights_validity && iszero(training_metric_error(
        args...; weights_validity, parameters...))
end

# SECTION - GeneticOptimizer Extension
struct GeneticOptimizer <: AbstractOptimizer
    global_iter::Int
    local_iter::Int
    memoize::Bool
    pop_size::Int
    sampler::Union{Nothing, Function}
    time_limit::Float64
    telemetry::Dict{Symbol,Any}
end

GeneticOptimizer(global_iter, local_iter, memoize, pop_size, sampler) =
    GeneticOptimizer(global_iter, local_iter, memoize, pop_size, sampler, Inf, Dict{Symbol,Any}())

@testitem "Genetic learning exposes evaluated candidates without changing its return API" tags=[:extension] default_imports=false begin
    import CompositionalNetworks as CN
    import ConstraintDomains: domain
    import Evolutionary
    import Test: @test

    network=CN.ICN(
        parameters = [:dom_size, :numvars],
        layers = [CN.Transformation, CN.Arithmetic, CN.Aggregation, CN.Comparison],
        connection = UInt32[1, 2, 3, 4]
    )
    observed=NamedTuple[]
    result=CN.explore_learn(
        [domain(1:2), domain(1:2)],
        values->values[1]==values[2],
        CN.GeneticOptimizer(2, 1, false, 4, nothing);
        icn = network,
        metric_function = CN.hamming,
        candidate_observer = candidate->push!(observed, candidate)
    )
    @test result isa Pair
    @test length(observed) >= 2
    @test all(candidate -> candidate.weights isa BitVector, observed)
    @test all(candidate -> !isnan(candidate.objective), observed)
end

@testitem "GeneticOptimizer" tags=[:extension] default_imports=false begin
    import CompositionalNetworks:
                                  Transformation, Arithmetic, Aggregation, Comparison, ICN,
                                  SimpleFilter
    import CompositionalNetworks: GeneticOptimizer, explore_learn, generate_configurations
    import CompositionalNetworks: check_weights_validity, hamming, solutions
    import ConstraintDomains: domain
    import Evolutionary
    import Test: @test

    test_icn=ICN(;
        parameters = [:dom_size, :numvars, :val],
        layers = [Transformation, Arithmetic, Aggregation, Comparison],
        connection = [1, 2, 3, 4]
    )

    function allunique_val(x; val)
        for i in 1:(length(x) - 1)
            for j in (i + 1):length(x)
                if x[i]==x[j]
                    if x[i]!=val
                        return false
                    end
                end
            end
        end
        return true
    end

    function allunique_vals(x; vals)
        for i in 1:(length(x) - 1)
            for j in (i + 1):length(x)
                if x[i]==x[j]
                    if !(x[i] in vals)
                        return false
                    end
                end
            end
        end
        return true
    end

    domains=[domain([1, 2, 3, 4]) for _ in 1:4]
    optimizer=GeneticOptimizer(1, 2, false, 8, nothing)
    function reported_validity_is_exact(result, concept; parameters...)
        network, reported=result
        configurations=generate_configurations(concept, domains; parameters...)
        solution_vector=[configuration.x for configuration in solutions(configurations)]
        structural=check_weights_validity(network, network.weights)
        exact=structural&&iszero(hamming(
            network,
            configurations,
            solution_vector;
            weights_validity = structural,
            network.constants...,
            parameters...
        ))
        return reported==exact
    end

    learned_val=explore_learn(
        domains,
        allunique_val,
        optimizer,
        icn = test_icn,
        val = 3
    )
    @test reported_validity_is_exact(learned_val, allunique_val; val = 3)

    new_test_icn=ICN(;
        parameters = [:dom_size, :numvars, :vals],
        layers = [SimpleFilter, Transformation, Arithmetic, Aggregation, Comparison],
        connection = [1, 2, 3, 4, 5]
    )

    learned_vals=explore_learn(
        domains,
        allunique_vals,
        optimizer,
        icn = new_test_icn,
        vals = [3, 4]
    )
    @test reported_validity_is_exact(learned_vals, allunique_vals; vals = [3, 4])
end

# SECTION - CBLSOptimizer Extension
struct LocalSearchOptimizer <: AbstractOptimizer
    options::Any
    strategy_builder::Any
    telemetry::Dict{Symbol,Any}
end

LocalSearchOptimizer(options) = LocalSearchOptimizer(options, nothing, Dict{Symbol,Any}())
LocalSearchOptimizer(options, strategy_builder) =
    LocalSearchOptimizer(options, strategy_builder, Dict{Symbol,Any}())

@testitem "LocalSearchOptimizer" tags=[:extension] default_imports=false begin
    import CompositionalNetworks: Transformation, Arithmetic, Aggregation, SimpleFilter
    import CompositionalNetworks: LocalSearchOptimizer, explore_learn, Comparison, ICN
    import CompositionalNetworks: generate_configurations, check_weights_validity, hamming,
                                  solutions
    import ConstraintDomains: domain
    import LocalSearchSolvers
    import Test: @test

    test_icn=ICN(;
        parameters = [:dom_size, :numvars, :val],
        layers = [Transformation, Arithmetic, Aggregation, Comparison],
        connection = [1, 2, 3, 4]
    )

    function allunique_val(x; val)
        for i in 1:(length(x) - 1)
            for j in (i + 1):length(x)
                if x[i]==x[j]
                    if x[i]!=val
                        return false
                    end
                end
            end
        end
        return true
    end

    function allunique_vals(x; vals)
        for i in 1:(length(x) - 1)
            for j in (i + 1):length(x)
                if x[i]==x[j]
                    if !(x[i] in vals)
                        return false
                    end
                end
            end
        end
        return true
    end

    domains=[domain([1, 2, 3, 4]) for _ in 1:4]
    options=LocalSearchSolvers.Options(
        iteration = (false, 10),
        time_limit = Inf,
        print_level = :silent,
        log_to_file = false,
        use_progress_meter = false
    )
    optimizer=LocalSearchOptimizer(options)
    function reported_validity_is_exact(result, concept; parameters...)
        network, reported=result
        configurations=generate_configurations(concept, domains; parameters...)
        solution_vector=[configuration.x for configuration in solutions(configurations)]
        structural=check_weights_validity(network, network.weights)
        exact=structural&&iszero(hamming(
            network,
            configurations,
            solution_vector;
            weights_validity = structural,
            network.constants...,
            parameters...
        ))
        return reported==exact
    end

    learned_val=explore_learn(
        domains, allunique_val, optimizer; icn = test_icn, val = 3)
    @test reported_validity_is_exact(learned_val, allunique_val; val = 3)
    @test optimizer.telemetry[:requested_iterations] == 10
    @test optimizer.telemetry[:exact_fixed_work]
    @test !optimizer.telemetry[:time_limit_reached]

    new_test_icn=ICN(;
        parameters = [:dom_size, :numvars, :vals],
        layers = [SimpleFilter, Transformation, Arithmetic, Aggregation, Comparison],
        connection = [1, 2, 3, 4, 5]
    )

    learned_vals=explore_learn(
        domains, allunique_vals, optimizer; icn = new_test_icn, vals = [3, 4])
    @test reported_validity_is_exact(learned_vals, allunique_vals; vals = [3, 4])
end
