module GeneticExt

import CompositionalNetworks:
                              CompositionalNetworks, AbstractICN, Configurations, manhattan,
                              hamming
import CompositionalNetworks: GeneticOptimizer, apply!, weights_bias, regularization
import CompositionalNetworks: evaluate, solutions
import CompositionalNetworks: training_metric_valid
import Evolutionary: Evolutionary, tournament, SPX, flip, GA

function CompositionalNetworks.GeneticOptimizer(;
        global_iter = Threads.nthreads(),
        # local_iter=64,
        local_iter = 400,
        memoize = false,
        #pop_size=64,
        pop_size = 100,
        sampler = nothing,
        time_limit = Inf,
        telemetry = Dict{Symbol,Any}(),
)
    return GeneticOptimizer(
        global_iter, local_iter, memoize, pop_size, sampler,
        Float64(time_limit), telemetry,
    )
end

function generate_population(icn, pop_size; vect = [])
    population = Vector{BitVector}()
    if isempty(vect)
        foreach(_ -> push!(population, falses(length(icn.weights))), 1:pop_size)
    else
        foreach(_ -> push!(population, vect), 1:pop_size)
    end
    return population
end

function CompositionalNetworks.optimize!(
        icn::T,
        configurations::Configurations,
        # dom_size,
        metric_function::Union{Function, Vector{Function}},
        optimizer_config::GeneticOptimizer;
        samples = nothing,
        memoize = false,
        candidate_observer = nothing,
        parameters...
) where {T <: AbstractICN}

    # @info icn.weights

    # inplace = zeros(dom_size, 18)
    solution_iter = solutions(configurations)
    non_solutions = solutions(configurations; non_solutions = true)
    solution_vector = [i.x for i in solution_iter]

    function fitness(w)
        weights_validity = apply!(icn, w)

        a = if metric_function isa Function
            metric_function(
                icn,
                configurations,
                solution_vector;
                weights_validity = weights_validity,
                parameters...
            )
        else
            minimum(
                met -> met(
                    icn,
                    configurations,
                    solution_vector;
                    weights_validity = weights_validity,
                    parameters...
                ),
                metric_function
            )
        end

        b = weights_bias(w)
        c = regularization(icn)

        if !isnothing(candidate_observer)
            candidate_observer((;
                weights = BitVector(w),
                structural_valid = weights_validity,
                training_error = Float64(a),
                objective = Float64(a + b + c)
            ))
        end

        function new_regularization(icn::AbstractICN)
            start = 1
            count = 0
            total = 0
            for (i, layer) in enumerate(icn.layers)
                if !layer.mutex
                    ran = start:(start + icn.weightlen[i] - 1)
                    op = findall(icn.weights[ran])
                    max_op = ran .- (start - 1)
                    total += (sum(op) / sum(max_op))
                    count += 1
                end
                start += icn.weightlen[i]
            end
            return total / count
        end

        d = sum(findall(icn.weights)) /
            (length(icn.weights) * (length(icn.weights) + 1) / 2)

        e = new_regularization(icn)

        # @info "Lot of things" a b c d e
        #=
        println("""
         sum: $a
         weights bias: $b
         regularization: $c
         new reg: $e
         thread: $(Threads.threadid())
         """) =#

        return a + b + c
    end

    _icn_ga = GA(;
        populationSize = optimizer_config.pop_size,
        crossoverRate = 0.8,
        epsilon = 0.05,
        selection = tournament(4),
        crossover = SPX,
        mutation = flip,
        mutationRate = 1.0
    )

    best_weights = BitVector(icn.weights)
    best_objective = Inf
    started_at = time()
    completed_global_iterations = 0
    time_limit_reached = false
    for _ in 1:optimizer_config.global_iter
        remaining = optimizer_config.time_limit - (time() - started_at)
        if remaining <= 0
            time_limit_reached = true
            break
        end
        pop = generate_population(icn, optimizer_config.pop_size)
        result = Evolutionary.optimize(
            fitness,
            pop,
            _icn_ga,
            Evolutionary.Options(;
                iterations = optimizer_config.local_iter,
                time_limit = isfinite(remaining) ? remaining : NaN,
            )
        )
        completed_global_iterations += 1
        time_limit_reached = isfinite(remaining) &&
                             Evolutionary.time_run(result) >= remaining
        weights = BitVector(Evolutionary.minimizer(result))
        objective = fitness(weights)
        if objective < best_objective
            best_objective = objective
            copyto!(best_weights, weights)
        end
        time_limit_reached && break
    end
    empty!(optimizer_config.telemetry)
    optimizer_config.telemetry[:requested_global_iterations] = optimizer_config.global_iter
    optimizer_config.telemetry[:completed_global_iterations] = completed_global_iterations
    optimizer_config.telemetry[:time_limit] = optimizer_config.time_limit
    optimizer_config.telemetry[:time_limit_reached] = time_limit_reached
    optimizer_config.telemetry[:exact_fixed_work] =
        !time_limit_reached && completed_global_iterations == optimizer_config.global_iter
    structural_validity = apply!(icn, best_weights)
    validity = training_metric_valid(
        icn,
        configurations,
        solution_vector,
        metric_function;
        weights_validity = structural_validity,
        parameters...
    )
    return icn => validity
end

end
