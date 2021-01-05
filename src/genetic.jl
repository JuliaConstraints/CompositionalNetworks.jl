function _generate_population(icn, pop_size)
    population = Vector{BitVector}()
    foreach(_ -> push!(population, _generate_weights(icn)), 1:pop_size)
    return population
end

function _loss(X, icn, weigths, metric)
    f = compose(icn, weigths)
    return sum(x -> abs(f(x) - metric(x, X)), X) + regularization(icn)
end

function _optimize(icn, X; ga = GA(), metric = hamming, pop_size = 100)
    f = weigths -> _loss(X, icn, weigths, metric)

    _icn_ga = GA(
        populationSize = pop_size,
        crossoverRate = 0.8,
        epsilon = 0.05,
        selection = rouletteinv,
        crossover = singlepoint,
        mutation = flip,
    )

    pop = generate_population(icn, pop_size)
    @info typeof(pop)
    @info pop

    optimize(f, pop, _icn_ga)
end
