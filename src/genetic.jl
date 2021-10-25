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
function loss(solutions, non_sltns, icn, weigths, metric, dom_size, param; samples=nothing)
    compo = compose(icn, weigths)
    f = composition(compo)
    X = if isnothing(samples)
        Iterators.flatten((solutions, non_sltns))
    else
        Iterators.flatten((solutions, rand(non_sltns, samples)))
    end
    σ = sum(x -> abs(f(x; param, dom_size) - metric(x, solutions)), X) + regularization(icn)
    return σ
end

"""
    _optimize!(icn, X, X_sols; metric = hamming, pop_size = 200)
Optimize and set the weigths of an ICN with a given set of configuration `X` and solutions `X_sols`.
"""
function _optimize!(
    icn,
    solutions,
    non_sltns,
    dom_size,
    param,
    metric,
    pop_size,
    iterations;
    samples=nothing,
    memoize=false,
)
    _metric = memoize ? (@memoize Dict memoize_metric(x, X) = metric(x, X)) : metric
    _bias = memoize ? (@memoize Dict memoize_bias(x) = weigths_bias(x)) : weigths_bias
    fitness =
        w ->
            loss(solutions, non_sltns, icn, w, _metric, dom_size, param; samples) + _bias(w)

    _icn_ga = GA(;
        populationSize=pop_size,
        crossoverRate=0.8,
        epsilon=0.05,
        selection=tournament(2),
        crossover=singlepoint,
        mutation=flip,
        mutationRate=1.0,
    )

    pop = generate_population(icn, pop_size)
    r = Evolutionary.optimize(fitness, pop, _icn_ga, Evolutionary.Options(; iterations))
    return weights!(icn, Evolutionary.minimizer(r))
end

"""
    optimize!(icn, X, X_sols, global_iter, local_iter; metric=hamming, popSize=100)
Optimize and set the weigths of an ICN with a given set of configuration `X` and solutions `X_sols`. The best weigths among `global_iter` will be set.
"""
function optimize!(
    icn,
    solutions,
    non_sltns,
    global_iter,
    iter,
    dom_size,
    param,
    metric,
    pop_size;
    sampler=nothing,
    memoize=false,
)
    results = Dictionary{BitVector,Int}()
    aux_results = Vector{BitVector}(undef, global_iter)
    nt = Base.Threads.nthreads()

    @info """Starting optimization of weights$(nt > 1 ? " (multithreaded)" : "")"""
    samples = isnothing(sampler) ? nothing : sampler(length(solutions) + length(non_sltns))
    @qthreads for i in 1:global_iter
        @info "Iteration $i"
        aux_icn = deepcopy(icn)
        _optimize!(
            aux_icn,
            solutions,
            non_sltns,
            dom_size,
            param,
            eval(metric),
            pop_size,
            iter;
            samples,
            memoize,
        )
        aux_results[i] = weigths(aux_icn)
    end
    foreach(bv -> incsert!(results, bv), aux_results)
    best = rand(findall(x -> x == maximum(results), results))
    weights!(icn, best)
    return best, results
end
