function _generate_population(icn, pop_size)
    population = Vector{BitVector}()
    foreach(_ -> push!(population, _generate_weights(icn)), 1:pop_size)
    return population
end

function _loss(X, X_sols, icn, weigths, metric)
    f = compose(icn, weigths)
    return sum(x -> abs(f(x) - metric(x, X_sols)), X) + regularization(icn)
end

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

function _optimize!(icn, X, X_sols; ga = GA(), metric = hamming, pop_size = 200)
    fitness = weigths -> _loss(X, X_sols, icn, weigths, metric)

    _viable_single_point = (v1, v2) -> _single_point(v1, v2, icn)
    _viable_flip = v -> _flip(v, icn)
    _icn_ga = GA(
        populationSize = pop_size,
        crossoverRate = 0.4,
        epsilon = 0.03,
        selection = rouletteinv,
        crossover = _viable_single_point,
        mutation = _viable_flip,
        mutationRate = 0.8
    )

    pop = _generate_population(icn, pop_size)
    res = optimize(fitness, pop, _icn_ga)
    _weigths!(icn, minimizer(res))
end
