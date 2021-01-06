struct Layer
    functions::LittleDict{Symbol, Function}
    exclusive::Bool
end

"""
    _functions(layer)
Access the operations of a layer. The container is ordered.
"""
_functions(layer) = layer.functions

"""
    _length(layer)
Return the number of operations in a layer.
"""
_length(layer::Layer) = length(_functions(layer))

"""
    _exclu(layer)
Return `true` if the layer has mutually exclusive operations.
"""
_exclu(layer) = layer.exclusive

"""
    _symbol(layer, i)
Return the i-th symbols of the operations in a given layer. 
"""
_symbol(layer, i) = collect(keys(_functions(layer)))[i]

"""
    _nbits_exclu(layer)
Convert the length of an exclusive layer into a number of bits.
"""
_nbits_exclu(layer) = ceil(Int, log2(_length(layer)))

"""
    _show_layer(layer)
Return a string that contains the elements in a layer.
"""
_show_layer(layer) = layer |> _functions |> keys |> string

"""
    _selected_size(layer, layer_weights)
Return the number of operations selected by `layer_weights` in `layer`.
"""
_selected_size(layer, layer_weights) = _exclu(layer) ? 1 : sum(layer_weights)

"""
    _is_viable(layer, w)
    _is_viable(icn)
    _is_viable(icn, w)
Assert if a pair of layer/icn and weigths compose a viable pattern. If no weigths are given with an icn, it will check the current internal value.
"""
_is_viable(layer::Layer, w) = _exclu(layer) ? _as_int(w) < _length(layer) : any(w)

"""
    _generate_inclusive_operations(predicate, bits)
    _generate_exclusive_operation(max_op_number)
Generates the operations (weigths) of a layer with inclusive/exclusive operations.
"""
function _generate_inclusive_operations(predicate, bits)
    ind = falses(bits)
    while true
        ind = bitrand(bits)
        predicate(ind) && break
    end
    return ind
end
function _generate_exclusive_operation(max_op_number)
    op = rand(1:max_op_number)
    return _as_bitvector(op, max_op_number)
end

"""
    _generate_weights(layers)
    _generate_weights(icn)
Generate the weigths of a collection of layers or of an ICN.
"""
function _generate_weights(layers)
    bitvecs = map(l -> _exclu(l) ?
            _generate_exclusive_operation(_length(l)) :
            _generate_inclusive_operations(any, _length(l)),
            layers
    )
    return vcat(bitvecs...)
end