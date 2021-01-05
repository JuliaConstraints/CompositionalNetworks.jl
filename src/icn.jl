"""
    ICN(; nvars, dom_size, param, transformation, arithmetic, aggregation, comparison)
Construct an Interpretable Compositional Network, with the following arguments:
- `nvars`: number of variable in the constraint
- `dom_size: maximum domain size of any variable in the constraint`
- `param`: optional parameter (default to `nothing`)
- `transformation`: a transformation layer (optional)
- `arithmetic`: a arithmetic layer (optional)
- `aggregation`: a aggregation layer (optional)
- `comparison`: a comparison layer (optional)
"""
mutable struct ICN
    transformation::Layer
    arithmetic::Layer
    aggregation::Layer
    comparison::Layer
    weigths::BitVector

    function ICN(; nvars, dom_size,
        param=nothing,
        tr_layer=transformation_layer(param),
        ar_layer=arithmetic_layer(),
        ag_layer=aggregation_layer(),
        co_layer=comparison_layer(nvars, dom_size, param),
    )
        w = _generate_weights([tr_layer, ar_layer, ag_layer, co_layer])
        new(tr_layer, ar_layer, ag_layer, co_layer, w)
    end
end

"""
    _layers(icn)
Return the ordered layers of an ICN.
"""
_layers(icn) = [icn.transformation, icn.arithmetic, icn.aggregation, icn.comparison]

"""
    _length(icn)
Return the total number of operations of an ICN.
"""
_length(icn::ICN) = sum(_length, _layers(icn))

"""
    _nbits(icn)
Return the expected number of bits of a viable weigth of an ICN.
"""
_nbits(icn) = mapreduce(l -> _exclu(l) ? _nbits_exclu(l) : _length(l), +, _layers(icn))

"""
    _weigths(icn)
Access the current set of weigths of an ICN.
"""
_weigths(icn) = icn.weigths

function _is_viable(icn::ICN, weigths)
    _start = 0
    _end = 0

    for layer in _layers(icn)
        _start = _end + 1
        _end += _exclu(layer) ? _nbits_exclu(layer) : _length(layer)

        w = @view weigths[_start:_end]

        !_is_viable(layer, w) && return false
    end
    return true
end

_is_viable(icn::ICN) = _is_viable(icn, _weigths(icn))

"""
    _weights!(icn, weights)
Set the weights of an ICN with a `BitVector`.
"""
function _weigths!(icn, weigths)
    @assert length(weigths) == _nbits(icn)
    # @assert _is_viable(icn, weigths)
    icn.weigths = weigths
end

"""
    show_layers(icn)
Return a formated string with each layers in the icn.
"""
show_layers(icn) = map(_show_layer, _layers(icn))

_generate_weights(icn::ICN) = _generate_weights(_layers(icn))

"""
    _compose(icn)
Internal function called by `compose` and `show_composition`.
"""
function _compose(icn::ICN)
    # @info "mark 1"
    funcs = Vector{Vector{Function}}()
    symbols = Vector{Vector{Symbol}}()

    _start = 0
    _end = 0

    for layer in _layers(icn)
        # @info "mark 2"
        _start = _end + 1
        _end += _exclu(layer) ? _nbits_exclu(layer) : _length(layer)

        if _exclu(layer)
            f_id = _as_int(@view _weigths(icn)[_start:_end])
            s = _symbol(layer, f_id + 1)
            push!(funcs, [_functions(layer)[s]])
            push!(symbols, [s])
        else
            layer_funcs = Vector{Function}()
            layer_symbs = Vector{Symbol}()
            for (f_id, b) in enumerate(@view _weigths(icn)[_start:_end])
                if b
                    s = _symbol(layer, f_id)
                    push!(layer_funcs, _functions(layer)[s])
                    push!(layer_symbs, s)
                end
            end
            push!(funcs, layer_funcs)
            push!(symbols, layer_symbs)
        end
    end

    l = length(funcs[1])
    composition = x -> fill(x, l) .|> funcs[1] |> funcs[2][1] |> funcs[3][1] |> funcs[4][1]
    return composition, symbols
end

"""
    show_composition(icn)
Return the composition (weights) of an ICN.
"""
function show_composition(icn)    
    !_is_viable(icn) && (@warn "not viable"; _weigths!(icn, _generate_weights(icn)))
    symbs = _compose(icn)[2]
    aux = map(s -> _reduce_symbols(s, "+", length(s) > 1), symbs)
    return _reduce_symbols(aux, "∘", false)
end

"""
    compose(icn)
    compose(icn, weights)
Return a function composed by some of the operations of a given ICN. Can be applied to any vector of variables. If `weights` are given, will assign to `icn`.
"""
function compose(icn::ICN)
    !_is_viable(icn) && (@warn "not viable"; @info icn; _weigths!(icn, _generate_weights(icn)))
    _compose(icn)[1]
end
function compose(icn, weigths)
    _weigths!(icn, weigths)
    compose(icn)
end

"""
    regularization(icn)
Return the regularization value of an ICN weights, which is proportional to the normalized number of operations selected in the icn layers.
"""
function regularization(icn)
    Σop = 0
    _start = 0
    _end = 0
    for layer in _layers(icn)
        l = _length(layer)
        _start = _end + 1
        _end += _exclu(layer) ? _nbits_exclu(layer) : l
        Σop += _selected_size(layer, @view _weigths(icn)[_start:_end])
    end
    return Σop / (_length(icn) + 1)
end
