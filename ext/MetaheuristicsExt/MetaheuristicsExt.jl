module MetaheuristicsExt

import CompositionalNetworks:
                              CompositionalNetworks, AbstractICN, Configurations, manhattan,
                              hamming
import CompositionalNetworks: MetaheuristicsOptimizer, apply!, weights_bias, regularization
import CompositionalNetworks: evaluate, solutions
import Metaheuristics: Metaheuristics, minimizer, GA, Algorithm, BitArraySpace

function generate_population(icn, pop_size; vect = [])
    population = Vector{BitVector}()
    if isempty(vect)
        foreach(_ -> push!(population, falses(length(icn.weights))), 1:pop_size)
    else
        foreach(_ -> push!(population, vect), 1:pop_size)
    end
    return population
end

function CompositionalNetworks.MetaheuristicsOptimizer(backend;
        maxiters = 400,
        maxtime = 500,
        extra_functions = Dict(),
        bounds = nothing
)

    if backend isa Metaheuristics.Algorithm{<:GA}
        extra_functions[:generate_population] = generate_population
    end

    return MetaheuristicsOptimizer(maxiters, maxtime, backend, bounds, extra_functions)
end


function CompositionalNetworks.optimize!(
        icn::T,
        configurations::Configurations,
        # dom_size,
        metric_function::Union{Function, Vector{Function}},
        optimizer_config::MetaheuristicsOptimizer;
        samples = nothing,
        memoize = false,
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


    #=
    _icn_ga = GA(;
        populationSize = optimizer_config.pop_size,
        crossoverRate = 0.8,
        epsilon = 0.05,
        selection = tournament(4),
        crossover = SPX,
        mutation = flip,
        mutationRate = 1.0
    )
    =#

    # pop = generate_population(icn, optimizer_config.pop_size)
    bounds = optimizer_config.bounds

    bounds = if isnothing(bounds)
        if optimizer_config.backend isa Metaheuristics.Algorithm{<:GA}
            BitSpacedArray(length(icn.weights))
        end
    else
        optimizer_config.bounds
    end

    r = Metaheuristics.optimize(
        fitness,
        optimizer_config.bounds,
        optimizer_config.backend
    )
    validity = apply!(icn, Metaheuristics.minimizer(r))
    return icn => validity
end

end
