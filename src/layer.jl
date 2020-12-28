struct Layer
    functions::LittleDict{Symbol, Function}
    exclusive::Bool
end

_functions(layer) = layer.functions
_length(layer) = length(_functions(layer))
_exclu(layer) = layer.exclusive
_symbol(layer, i) = collect(keys(_functions(layer)))[i]
