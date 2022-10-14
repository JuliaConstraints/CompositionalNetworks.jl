"""
    Layer
A structure to store a `LittleDict` of operations that can be selected during the learning phase of an ICN. If the layer is exclusive, only one operation can be selected at a time.
"""
struct Layer
    functions::LittleDict{Symbol, Function}
    exclusive::Bool
end

"""
    functions(layer)
Access the operations of a layer. The container is ordered.
"""
functions(layer) = layer.functions

"""
    length(layer)
Return the number of operations in a layer.
"""
Base.length(layer::Layer) = length(functions(layer))

"""
    exclu(layer)
Return `true` if the layer has mutually exclusive operations.
"""
exclu(layer) = layer.exclusive

"""
    symbol(layer, i)
Return the i-th symbols of the operations in a given layer.
"""
symbol(layer, i) = collect(keys(functions(layer)))[i]

"""
    nbits_exclu(layer)
Convert the length of an exclusive layer into a number of bits.
"""
nbits_exclu(layer) = ceil(Int, log2(length(layer)))

"""
    show_layer(layer)
Return a string that contains the elements in a layer.
"""
show_layer(layer) = layer |> functions |> keys |> string

"""
    selected_size(layer, layer_weigths)
Return the number of operations selected by `layer_weigths` in `layer`.
"""
selected_size(layer, layer_weigths) = exclu(layer) ? 1 : sum(layer_weigths)

"""
    is_viable(layer, w)
    is_viable(icn)
    is_viable(icn, w)
Assert if a pair of layer/icn and weigths compose a viable pattern. If no weigths are given with an icn, it will check the current internal value.
"""
is_viable(layer::Layer, w) = exclu(layer) ? as_int(w) < length(layer) : any(w)

"""
    generate_inclusive_operations(predicate, bits)
    generate_exclusive_operation(max_op_number)
Generates the operations (weigths) of a layer with inclusive/exclusive operations.
"""
function generate_inclusive_operations(predicate, bits)
    ind = falses(bits)
    while true
        ind = bitrand(bits)
        predicate(ind) && break
    end
    return ind
end

function generate_exclusive_operation(max_op_number)
    op = rand(1:max_op_number)
    return as_bitvector(op, max_op_number)
end

"""
    generate_weigths(layers)
    generate_weigths(icn)
Generate the weigths of a collection of layers or of an ICN.
"""
function generate_weigths(layers)
    bitvecs = map(l -> exclu(l) ?
            generate_exclusive_operation(length(l)) :
            generate_inclusive_operations(any, length(l)),
            layers
    )
    return vcat(bitvecs...)
end
