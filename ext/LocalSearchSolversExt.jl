module LocalSearchSolversExt

import CompositionalNetworks: CompositionalNetworks, AbstractICN, Configurations
import CompositionalNetworks: LocalSearchOptimizer, apply!, weights_bias, regularization
import CompositionalNetworks: evaluate, solutions
import CompositionalNetworks: training_metric_valid
import LocalSearchSolvers: model, domain, variable!, constraint!, objective!, solver, solve!
import LocalSearchSolvers: LocalSearchSolvers, MetaStrategy, has_solution, best_values

function _solver_iterations(search_solver)
    if isdefined(LocalSearchSolvers, :iterations)
        return Base.invokelatest(getfield(LocalSearchSolvers, :iterations), search_solver)
    end
    return hasproperty(search_solver, :iterations) ? getproperty(search_solver, :iterations) :
           missing
end

function CompositionalNetworks.LocalSearchOptimizer(;
        options::LocalSearchSolvers.Options = LocalSearchSolvers.Options(),
        strategy_builder = nothing,
        telemetry = Dict{Symbol,Any}(),
)
    return LocalSearchOptimizer(options, strategy_builder, telemetry)
end

mutually_exclusive(_, w) = abs(sum(w) - 1)

no_empty_layer(x; X = nothing) = max(0, 1 - sum(x))

parameter_specific_operations(x; X = nothing) = 0.0

function CompositionalNetworks.optimize!(
        icn::T,
        configurations::Configurations,
        metric_function::Union{Function, Vector{Function}},
        optimizer_config::LocalSearchOptimizer;
        candidate_observer = nothing,
        parameters...
) where {T <: AbstractICN}
    @debug "starting debug opt"
    m = model(; kind = :icn)
    n = length(icn.weights)
    solution_vector = [configuration.x for configuration in solutions(configurations)]
    # Each LocalSearchSolvers trajectory may evaluate the objective concurrently.
    # `apply!` mutates its ICN, so sharing one network would race on the weights.
    # Thread-owned copies keep the candidate loop allocation-free.
    fitness_networks = [deepcopy(icn) for _ in 1:Threads.maxthreadid()]
    time_limit = last(optimizer_config.options.time_limit)
    deadline = isfinite(time_limit) ? time() + time_limit : Inf
    deadline_reached = Threads.Atomic{Bool}(false)

    # All variables are boolean
    d = domain([false, true])

    # Add variables
    foreach(_ -> variable!(m, d), 1:n)

    # Add constraint
    start = 1
    for (i, layer) in enumerate(icn.layers)
        stop = start + icn.weightlen[i] - 1
        if layer.mutex
            f(x; X = nothing) = mutually_exclusive(icn.weightlen[i], x)
            constraint!(m, f, start:stop)
        else
            constraint!(m, no_empty_layer, start:stop)
        end
        start = stop + 1
    end

    function fitness(w)
        # LocalSearchSolvers normally checks its time limit between iterations.
        # An ICN iteration may itself scan a large neighbourhood, so make every
        # candidate evaluation participate in the same deadline. Once reached,
        # the remainder of the current scan becomes cheap and the outer loop
        # observes its normal time limit immediately afterwards.
        if time() >= deadline
            deadline_reached[] = true
            return Inf
        end
        network = fitness_networks[Threads.threadid()]
        weights_validity = apply!(network, w)

        score = if metric_function isa Function
            metric_function(
                network,
                configurations,
                solution_vector;
                weights_validity = weights_validity,
                parameters...
            )
        else
            minimum(
                met -> met(
                    network,
                    configurations,
                    solution_vector;
                    weights_validity = weights_validity,
                    parameters...
                ),
                metric_function
            )
        end

        objective = score + weights_bias(w) + regularization(network)
        if !isnothing(candidate_observer)
            candidate_observer((;
                weights = BitVector(w),
                structural_valid = weights_validity,
                training_error = Float64(score),
                objective = Float64(objective)
            ))
        end
        return objective
    end

    objective!(m, fitness)

    # Create solver and solve
    strategy = isnothing(optimizer_config.strategy_builder) ?
               MetaStrategy(m) : optimizer_config.strategy_builder(m)
    strategy isa MetaStrategy || throw(ArgumentError(
        "strategy_builder must return a LocalSearchSolvers.MetaStrategy"))
    search_solver = solver(m; options = optimizer_config.options, strategies = strategy)
    solve!(search_solver)
    empty!(optimizer_config.telemetry)
    optimizer_config.telemetry[:iterations] = _solver_iterations(search_solver)
    optimizer_config.telemetry[:julia_threads] = Threads.nthreads()
    optimizer_config.telemetry[:search_units] =
        hasproperty(search_solver, :subs) ? 1 + length(search_solver.subs) : 1
    # The main solver's counter is per trajectory, not the aggregate thread work.
    # Collect these after the join; no counter is added to the objective hot loop.
    local_iterations = Union{Missing,Int}[_solver_iterations(search_solver)]
    if hasproperty(search_solver, :subs)
        append!(local_iterations, _solver_iterations.(search_solver.subs))
    end
    optimizer_config.telemetry[:local_trajectory_iterations] = local_iterations
    optimizer_config.telemetry[:local_total_iterations] =
        any(ismissing, local_iterations) ? missing : sum(local_iterations)
    termination_status = deadline_reached[] ? :time_limit :
                         hasproperty(search_solver, :status) ?
                         getproperty(search_solver, :status) : :unknown
    optimizer_config.telemetry[:termination_status] = String(termination_status)
    optimizer_config.telemetry[:time_limit_reached] = termination_status === :time_limit
    requested_iterations = last(optimizer_config.options.iteration)
    optimizer_config.telemetry[:requested_iterations] = requested_iterations
    optimizer_config.telemetry[:exact_fixed_work] =
        termination_status === :iteration_limit &&
        (ismissing(optimizer_config.telemetry[:iterations]) ||
         optimizer_config.telemetry[:iterations] == requested_iterations)
    @debug "pool" search_solver.pool best_values(search_solver.pool) best_values(search_solver) search_solver.pool.configurations

    # Return best values

    structural_validity = if has_solution(search_solver)
        apply!(icn, BitVector(collect(best_values(search_solver))))
    else
        CompositionalNetworks.generate_new_valid_weights!(icn)
    end

    weights_validity = training_metric_valid(
        icn,
        configurations,
        solution_vector,
        metric_function;
        weights_validity = structural_validity,
        parameters...
    )

    return icn => weights_validity
end

end
