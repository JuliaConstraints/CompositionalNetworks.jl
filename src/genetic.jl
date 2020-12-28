function _generate_individual_layer(predicate, bits)
    while true
        ind = bitrand(bits)
        predicate(ind) && break
    end
    return ind
end
    

function generate_population(pop_size, nvars, dom_size;
    tr_layer = transformation_layer(),
    ar_layer = arithmetic_layer(),
    ag_layer = aggregation_layer(),
    co_layer = comparison_layer(nvars, dom_size),
)
    tr_length = length(tr_layer)
    ar_length = ceil(log2(length(ar_layer)))
    ag_length = ceil(log2(length(ag_layer)))
    co_length = length(co_layer) 

    tr_bitvec = _generate_individual_layer(any, tr_length)
    ar_bitvec = _generate_individual_layer()

    co_bitvec = _generate_individual_layer(any, co_length)


    


end
