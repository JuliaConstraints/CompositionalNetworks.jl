module GeneticExt

using CompositionalNetworks
using Evolutionary

function CompositionalNetworks.GeneticOptimizer(;
    global_iter=Threads.nthreads(),
    local_iter=64,
    memoize=false,
    pop_size=64,
    sampler=nothing,
)
    return GeneticOptimizer(global_iter, local_iter, memoize, pop_size, sampler)
end

function generate_population(icn, pop_size)
    population = Vector{BitVector}()
    foreach(_ -> push!(population, falses(length(icn.weights))), 1:pop_size)
    return population
end

function CompositionalNetworks.optimize!(
    icn::T,
    configurations::Configurations,
    # dom_size,
    metric_function::Function,
    optimizer_config::GeneticOptimizer; samples=nothing, memoize=false, parameters...) where {T<:AbstractICN}

    # @info icn.weights

    # inplace = zeros(dom_size, 18)
    solution_iter = solutions(configurations)
    non_solutions = solutions(configurations; non_solutions=true)
    solution_vector = [i.x for i in solution_iter]

    function fitness(w)
        weights_validity = CompositionalNetworks.apply!(icn, w)
        return sum(
            x -> abs(evaluate(icn, x; weights_validity=weights_validity, parameters...) - metric_function(x.x, solution_vector)), configurations
        ) + CompositionalNetworks.weights_bias(w)# + CompositionalNetworks.regularization(icn)
    end

    _icn_ga = GA(;
        populationSize=optimizer_config.pop_size,
        crossoverRate=0.8,
        epsilon=0.05,
        selection=tournament(2),
        crossover=SPX,
        mutation=flip,
        mutationRate=1.0
    )

    pop = generate_population(icn, optimizer_config.pop_size)
    r = Evolutionary.optimize(fitness, pop, _icn_ga, Evolutionary.Options(; iterations=optimizer_config.local_iter))
    validity = CompositionalNetworks.apply!(icn, Evolutionary.minimizer(r))
    return icn => validity
end

end
