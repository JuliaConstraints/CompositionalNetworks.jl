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

    function ICN(;
        param=false,
        tr_layer=transformation_layer(param),
        ar_layer=arithmetic_layer(),
        ag_layer=aggregation_layer(),
        co_layer=comparison_layer(param),
    )
        w = generate_weights([tr_layer, ar_layer, ag_layer, co_layer])
        new(tr_layer, ar_layer, ag_layer, co_layer, w)
    end
end

"""
    layers(icn)
Return the ordered layers of an ICN.
"""
layers(icn) = [icn.transformation, icn.arithmetic, icn.aggregation, icn.comparison]

"""
    Base.length(icn)
Return the total number of operations of an ICN.
"""
Base.length(icn::ICN) = sum(length, layers(icn))

"""
    nbits(icn)
Return the expected number of bits of a viable weigth of an ICN.
"""
nbits(icn) = mapreduce(l -> exclu(l) ? nbits_exclu(l) : length(l), +, layers(icn))

"""
    weigths(icn)
Access the current set of weigths of an ICN.
"""
weigths(icn) = icn.weigths

function is_viable(icn::ICN, weigths)
    _start = 0
    _end = 0

    for layer in layers(icn)
        _start = _end + 1
        _end += exclu(layer) ? nbits_exclu(layer) : length(layer)

        w = @view weigths[_start:_end]

        !is_viable(layer, w) && return false
    end
    return true
end
is_viable(icn::ICN) = is_viable(icn, weigths(icn))

"""
    weights!(icn, weights)
Set the weights of an ICN with a `BitVector`.
"""
function weights!(icn, weigths)
    @assert length(weigths) == nbits(icn)
    icn.weigths = weigths
end

"""
    show_layers(icn)
Return a formated string with each layers in the icn.
"""
show_layers(icn) = map(show_layer, layers(icn))

generate_weights(icn::ICN) = generate_weights(layers(icn))

"""
    _compose(icn)
Internal function called by `compose` and `show_composition`.
"""
function _compose(icn::ICN)
    !is_viable(icn) && (return ((x; param=nothing, dom_size=0) -> typemax(Float64)), [])

    funcs = Vector{Vector{Function}}()
    symbols = Vector{Vector{Symbol}}()

    _start = 0
    _end = 0

    for layer in layers(icn)
        _start = _end + 1
        _end += exclu(layer) ? nbits_exclu(layer) : length(layer)

        if exclu(layer)
            f_id = as_int(@view weigths(icn)[_start:_end])
            s = symbol(layer, f_id + 1)
            push!(funcs, [functions(layer)[s]])
            push!(symbols, [s])
        else
            layer_funcs = Vector{Function}()
            layer_symbs = Vector{Symbol}()
            for (f_id, b) in enumerate(@view weigths(icn)[_start:_end])
                if b
                    s = symbol(layer, f_id)
                    push!(layer_funcs, functions(layer)[s])
                    push!(layer_symbs, s)
                end
            end
            push!(funcs, layer_funcs)
            push!(symbols, layer_symbs)
        end
    end

    function composition(x; X=zeros(length(x), length(funcs[1])), param=nothing, dom_size)
        tr_in(Tuple(funcs[1]), X, x, param)
        for i in 1:length(x)
            X[i,1] = funcs[2][1](@view X[i,:])
        end
        funcs[3][1](@view X[:, 1]) |>
        (y -> funcs[4][1](y; param, dom_size, nvars=length(x)))
    end

    return composition, symbols
end

"""
    compose(icn)
    compose(icn, weights)
Return a function composed by some of the operations of a given ICN. Can be applied to any vector of variables. If `weights` are given, will assign to `icn`.
"""
function compose(icn::ICN; action=:composition)
    return action == :symbols ? _compose(icn)[2] : _compose(icn)[1]
end
function compose(icn, weigths; action=:composition)
    weights!(icn, weigths)
    compose(icn; action=action)
end

"""
    show_composition(icn)
Return the composition (weights) of an ICN.
"""
function show_composition(icn)
    symbs = compose(icn, action=:symbols)
    aux = map(s -> reduce_symbols(s, ", ", length(s) > 1), symbs)
    return reduce_symbols(aux, " ∘ ", false)
end

"""
    regularization(icn)
Return the regularization value of an ICN weights, which is proportional to the normalized number of operations selected in the icn layers.
"""
function regularization(icn)
    Σmax = 0
    Σop = 0
    _start = 0
    _end = 0
    for layer in layers(icn)
        l = length(layer)
        _start = _end + 1
        _end += exclu(layer) ? nbits_exclu(layer) : l
        if !exclu(layer)
            Σop += selected_size(layer, @view weigths(icn)[_start:_end])
            Σmax += length(layer)
        end
    end
    return Σop / (Σmax + 1)
end

max_icn_length(icn = ICN(param = true)) = length(icn.transformation)
