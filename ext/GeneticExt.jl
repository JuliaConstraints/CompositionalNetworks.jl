module GeneticExt

import CompositionalNetworks: CompositionalNetworks, AbstractICN, Configurations, manhattan, hamming
import CompositionalNetworks: GeneticOptimizer, apply!, weights_bias, regularization
import CompositionalNetworks: evaluate, solutions
import Evolutionary: Evolutionary, tournament, SPX, flip, GA

function CompositionalNetworks.GeneticOptimizer(;
    global_iter=Threads.nthreads(),
    # local_iter=64,
    local_iter=400,
    memoize=false,
    #pop_size=64,
    pop_size=100,
    sampler=nothing,
)
    return GeneticOptimizer(global_iter, local_iter, memoize, pop_size, sampler)
end

function generate_population(icn, pop_size; vect=[])
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
    metric_function::Union{Function,Vector{Function}},
    optimizer_config::GeneticOptimizer; samples=nothing, memoize=false, parameters...) where {T<:AbstractICN}

    # @info icn.weights

    # inplace = zeros(dom_size, 18)
    solution_iter = solutions(configurations)
    non_solutions = solutions(configurations; non_solutions=true)
    solution_vector = [i.x for i in solution_iter]

    function fitness(w)
        weights_validity = apply!(icn, w)

        a = if metric_function isa Function
            metric_function(icn, configurations, solution_vector; weights_validity=weights_validity, parameters...)
        else
            minimum(met -> met(icn, configurations, solution_vector; weights_validity=weights_validity, parameters...), metric_function)
        end

        b = weights_bias(w)
        c = regularization(icn)

        function new_regularization(icn::AbstractICN)
            start = 1
            count = 0
            total = 0
            for (i, layer) in enumerate(icn.layers)
                if !layer.mutex
                    ran = start:(start+icn.weightlen[i]-1)
                    op = findall(icn.weights[ran])
                    max_op = ran .- (start - 1)
                    total += (sum(op) / sum(max_op))
                    count += 1
                end
                start += icn.weightlen[i]
            end
            return total / count
        end

        d = sum(findall(icn.weights)) / (length(icn.weights) * (length(icn.weights) + 1) / 2)

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
        populationSize=optimizer_config.pop_size,
        crossoverRate=0.8,
        epsilon=0.05,
        selection=tournament(4),
        crossover=SPX,
        mutation=flip,
        mutationRate=1.0
    )

    pop = generate_population(icn, optimizer_config.pop_size)
    r = Evolutionary.optimize(fitness, pop, _icn_ga, Evolutionary.Options(; iterations=optimizer_config.local_iter))
    validity = apply!(icn, Evolutionary.minimizer(r))
    return icn => validity
end

end
