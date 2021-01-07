"""
    _generate_population(icn, pop_size
Generate a pôpulation of weigths (individuals) for the genetic algorithm weigthing `icn`.
"""
function _generate_population(icn, pop_size)
    population = Vector{BitVector}()
    foreach(_ -> push!(population, falses(_nbits(icn))), 1:pop_size)
    return population
end
# function _generate_population(icn, pop_size)
#     population = Vector{BitVector}()
#     foreach(_ -> push!(population, _generate_weights(icn)), 1:pop_size)
#     return population
# end

"""
    _loss(X, X_sols, icn, weigths, metric)
Compute the loss of `icn`.
"""
function _loss(X, X_sols, icn, weigths, metric)
    f = compose(icn, weigths)
    return sum(x -> abs(f(x) - metric(x, X_sols)), X) + regularization(icn)
end

"""
    _single_point(v1::T, v2::T, icn) where {T <: AbstractVector}
    _flip(recombinant::T, icn) where {T <: BitVector}
Meta functions that add individuals a viability check in `Evolutionary.singlepoint` and `Evolutionary.flip`.
"""
function _single_point(v1::T, v2::T, icn) where {T <: AbstractVector}
    while true
        c1, c2 = singlepoint(v1, v2)
        _is_viable(icn, c1) && _is_viable(icn, c2) && (return c1, c2)
    end
end
function _flip(recombinant::T, icn) where {T <: BitVector}
    c = deepcopy(recombinant)
    while true
        flip(c)
        _is_viable(icn, c) && (recombinant = c: break)
    end
end

"""
    _optimize!(icn, X, X_sols; metric = hamming, pop_size = 200)
Optimize and set the weigths of an ICN with a given set of configuration `X` and solutions `X_sols`.
"""
function _optimize!(icn, X, X_sols; metric = hamming, pop_size = 200, iter = 100)
    fitness = weigths -> _loss(X, X_sols, icn, weigths, metric)
    _viable_single_point = (v1, v2) -> _single_point(v1, v2, icn)
    _viable_flip = v -> _flip(v, icn)
    # _icn_ga = GA(
    #     populationSize = pop_size,
    #     # crossoverRate = 0.8,
    #     epsilon = 10,
    #     # selection = rouletteinv,
    #     crossover = _viable_single_point,
    #     mutation = _viable_flip,
    #     # mutationRate = 0.8
    # )

    _icn_ga = GA(
        populationSize = pop_size,
        crossoverRate = 0.8,
        epsilon = 0.05,
        selection = tournament(2),
        crossover = singlepoint,
        mutation = flip,
        mutationRate = 1.0
    )

    pop = _generate_population(icn, pop_size)
    dpop = Dictionary{BitVector, Int}()
    for p in pop
        _incsert!(dpop, p)
    end
    @info maximum(dpop)


    res = optimize(fitness, pop, _icn_ga,
        # Options(iterations=iter, abstol = -0.1, reltol = -0.1, store_trace = true, show_trace = true)
        Options(iterations=iter, store_trace = true, show_trace = true)
        )
        
    @info res
    @info summary(res)
    @info trace(res)
    _weigths!(icn, minimizer(res))
end

function optimize(icn, X, X_sols, global_iterations, local_iterations; metric = hamming, popSize = 100)
    results = Dictionary{BitVector, Int}()
    @info "Starting optimization of weights"
    for i in 1:global_iterations
        @info "Iteration $i"
        _optimize!(icn, X, X_sols)
        _incsert!(results, _weigths(icn))
    end
    best = findall(x -> x == maximum(results), results)
    @info best results
    return best, results
end

# function optimize(icn, X, X_sols, global_iterations, local_iterations; metric = hamming, pop_size = 100)
#     results = Dictionary{BitVector, Int}()
#     @info "Starting optimization of weights"
#     for i in 1:global_iterations
#         @info "Iteration $i"
#         _optimize!(icn, X, X_sols; iter = local_iterations, pop_size = pop_size)
#         _incsert!(results, _weigths(icn))
#     end
#     best = findall(x -> x == maximum(results), results)
#     @info best results
#     return best, results
# end