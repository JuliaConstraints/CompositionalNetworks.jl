function _generate_inclusive_operations(predicate, bits)
    while true
        ind = bitrand(bits)
        predicate(ind) && break
    end
    return ind
end

function _generate_exclusive_operation(max_op_number)
    op = rand(1:nax_op_number)
    return _as_bitvector(op, max_op_number)
end


function generate_population(icn, pop_size)
    population = falses(_length(icn), pop_size)

    for i in 1:pop_size
        bitvecs = map(l -> _exclu(l) ?
            _generate_exclusive_operation(_length(l)) :
            _generate_inclusive_operations(any, _length(l)),
            layers(icn)
        )
        population[:, i] = vcat(bitvecs...)
    end
    return population
end

function _loss(X, icn, weigths, metric)
    f = compose(icn, weigths)
    return sum(x -> abs(f(x) - metric(x)), X) + regularization(icn)
end
