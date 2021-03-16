"""
    generate_population(icn, pop_size
Generate a pôpulation of weigths (individuals) for the genetic algorithm weigthing `icn`.
"""
function generate_population(icn, pop_size)
    population = Vector{BitVector}()
    foreach(_ -> push!(population, falses(nbits(icn))), 1:pop_size)
    return population
end

"""
    loss(X, X_sols, icn, weigths, metric)
Compute the loss of `icn`.
"""
function loss(X, X_sols, icn, weigths, metric, dom_size, param)
    f = compose(icn, weigths)
    return (sum(x -> abs(f(x; param = param, dom_size = dom_size) - metric(x, X_sols)), X) + regularization(icn))
end

"""
    _optimize!(icn, X, X_sols; metric = hamming, pop_size = 200)
Optimize and set the weigths of an ICN with a given set of configuration `X` and solutions `X_sols`.
"""
function _optimize!(icn, X, X_sols, dom_size, param=nothing; metric=hamming, pop_size=200, iter=100)
    fitness = weigths -> loss(X, X_sols, icn, weigths, metric, dom_size, param)

    _icn_ga = GA(
        populationSize=pop_size,
        crossoverRate=0.8,
        epsilon=0.05,
        selection=tournament(2),
        crossover=singlepoint,
        mutation=flip,
        mutationRate=1.0
    )

    pop = generate_population(icn, pop_size)
    r = Evolutionary.optimize(fitness, pop, _icn_ga, Evolutionary.Options(iterations=iter))
    weights!(icn, Evolutionary.minimizer(r))
end

"""
    optimize!(icn, X, X_sols, global_iter, local_iter; metric=hamming, popSize=100)
Optimize and set the weigths of an ICN with a given set of configuration `X` and solutions `X_sols`. The best weigths among `global_iter` will be set.
"""
function optimize!(icn, X, X_sols, global_iter, local_iter, dom_size, param=nothing; metric=hamming, popSize=100)
    results = Dictionary{BitVector,Int}()
    aux_results = Vector{BitVector}(undef, global_iter)
    nt = Base.Threads.nthreads()
    @info """Starting optimization of weights$(nt > 1 ? " (multithreaded)" : "")"""
    Base.Threads.@threads for i in 1:global_iter
        @info "Iteration $i"
        aux_icn = deepcopy(icn)
        _optimize!(aux_icn, X, X_sols, dom_size, param;
            iter=local_iter, metric=metric, pop_size=popSize
        )
        aux_results[i] = weigths(aux_icn)
    end
    foreach(bv -> incsert!(results, bv), aux_results)
    best = rand(findall(x -> x == maximum(results), results))
    weights!(icn, best)
    @info show_composition(icn) best results
    return best, results
end
