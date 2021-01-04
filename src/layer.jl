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
