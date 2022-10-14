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
    inplace = zeros(dom_size, max_icn_length())
    _non_sltns = isnothing(samples) ? non_sltns : rand(non_sltns, samples)

    function fitness(w)
        compo = compose(icn, w)
        f = composition(compo)
        S = Iterators.flatten((solutions, _non_sltns))
        return sum(x -> abs(f(x; X=inplace, param, dom_size) - metric(x, solutions)), S) +
               regularization(icn) +
               weigths_bias(w)
    end
    _fitness = memoize ? (@memoize Dict memoize_fitness(w) = fitness(w)) : fitness

    _icn_ga = GA(;
        populationSize=pop_size,
        crossoverRate=0.8,
        epsilon=0.05,
        selection=tournament(2),
        crossover=SPX,
        mutation=flip,
        mutationRate=1.0,
    )

    pop = generate_population(icn, pop_size)
    r = Evolutionary.optimize(_fitness, pop, _icn_ga, Evolutionary.Options(; iterations))
    return weigths!(icn, Evolutionary.minimizer(r))
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

    @info """Starting optimization of weigths$(nt > 1 ? " (multithreaded)" : "")"""
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
    weigths!(icn, best)
    return best, results
end

struct GeneticOptimizer <: AbstractOptimizer
    global_iter::Int
    local_iter::Int
    memoize::Bool
    pop_size::Int
    sampler::Union{Nothing, Function}
end

function GeneticOptimizer(;
    global_iter=Threads.nthreads(),
    local_iter=64,
    memoize=false,
    pop_size=64,
    sampler=nothing,
)
    return GeneticOptimizer(global_iter, local_iter, memoize, pop_size, sampler)
end

function CN.optimize!(
    icn, solutions, non_sltns, dom_size, param, metric, optimizer::GeneticOptimizer
)
    return optimize!(
        icn,
        solutions,
        non_sltns,
        optimizer.global_iter,
        optimizer.local_iter,
        dom_size,
        param,
        metric,
        optimizer.pop_size;
        optimizer.sampler,
        optimizer.memoize,
    )
end
