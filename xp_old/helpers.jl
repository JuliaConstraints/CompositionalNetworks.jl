"""
Contains all icn functions that somehow aren't recognized by the compiler
TODO:: Fix it later
"""


function as_int(v::AbstractVector)
    n = 0
    for (i, b) in enumerate(v)
        n += b ? 2^(i - 1) : 0
    end
    return n
end

nbits(icn) = mapreduce(l -> exclu(l) ? nbits_exclu(l) : length(l), +, layers(icn))

functions(layer) = layer.functions

symbol(layer, i) = collect(keys(functions(layer)))[i]

nbits_exclu(layer) = ceil(Int, log2(length(layer)))

exclu(layer) = layer.exclusive

layers(icn) = [icn.transformation, icn.arithmetic, icn.aggregation, icn.comparison]

weigths(icn) = icn.weigths

is_viable(layer, w) = exclu(layer) ? as_int(w) < length(layer) : any(w)

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

function weights!(icn, weigths)
    @assert length(weigths) == nbits(icn)
    icn.weigths = weigths
end

@unroll function tr_in(tr, X, x, param)
    @unroll for i in 1:length(tr)
        X[:,i] = tr[i](x; param)
    end
end

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
function compose(icn::ICN; action=:composition)
    return action == :symbols ? _compose(icn)[2] : _compose(icn)[1]
end
function compose(icn, weigths; action=:composition)
    weights!(icn, weigths)
    compose(icn; action=action)
end

fitness = w -> loss(X, X_sols, icn, w, metric, maximum(length, domains), param)
fitness = (w, X, X_sols, icn, metric, param) -> loss(X, X_sols, icn, w, metric, maximum(length, domains), param)

function loss(X, X_sols, icn, weigths, metric, dom_size, param) 
    f = compose(icn, weigths)
    return (sum(x -> abs.(f(x; param = param, dom_size = dom_size) - metric(x, X_sols)), X))
end

function complete_search_space(domains, concept, param=nothing)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    message = "Space size for complete search"
    space_size = prod(length, domains)

    space_size < 10^6 ? @info(message, space_size) : @warn(message, space_size)

    f = isnothing(param) ? ((x; param = p) -> concept(x)) : concept

    configurations = Base.Iterators.product(map(d -> get_domain(d), domains)...)
    foreach(c -> (cv = collect(c); push!(f(cv; param=param) ? solutions : non_sltns, cv)), configurations)

    return solutions, non_sltns
end