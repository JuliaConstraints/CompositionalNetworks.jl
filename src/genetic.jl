function _generate_inclusive_operations(predicate, bits)
    ind = bitrand(bits)
    while true
        predicate(ind) && break
        ind = bitrand(bits)
    end
    return ind
end

function _generate_exclusive_operation(max_op_number)
    op = rand(1:max_op_number)
    return _as_bitvector(op, max_op_number)
end


function generate_population(icn, pop_size)
    population = falses(_nbits(icn), pop_size)

    for i in 1:pop_size
        bitvecs = map(l -> _exclu(l) ?
            _generate_exclusive_operation(_length(l)) :
            _generate_inclusive_operations(any, _length(l)),
            _layers(icn)
        )
        population[:, i] = vcat(bitvecs...)
    end
    return population
end

function _loss(X, icn, weigths, metric)
    f = compose(icn, weigths)
    return sum(x -> abs(f(x) - metric(x, X)), X) + regularization(icn)
end

function _optimize(icn, X; ga = _icn_ga, metric = hamming, pop_size = 100)
    f = weigths -> _loss(X, icn, weigths, metric)

    _icn_ga = GA(populationSize = pop_size, crossoverRate = 0.8, ɛ = .05, selection = tournament, crossover = singlepoint, mutation = flip)

    pop = generate_population(icn, pop_size)
    pop = Vector(map(Vector,eachcol(pop)))

    optimize(f, pop, _icn_ga)
end
