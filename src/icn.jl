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
    weights::BitVector

    function ICN(;
        param = Vector{Symbol}(),
        tr_layer = transformation_layer(param),
        ar_layer = arithmetic_layer(),
        ag_layer = aggregation_layer(),
        co_layer = comparison_layer(param),
    )
        w = generate_weights([tr_layer, ar_layer, ag_layer, co_layer])
        return new(tr_layer, ar_layer, ag_layer, co_layer, w)
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
Return the expected number of bits of a viable weight of an ICN.
"""
nbits(icn) = mapreduce(l -> exclu(l) ? nbits_exclu(l) : length(l), +, layers(icn))

"""
    weights(icn)
Access the current set of weights of an ICN.
"""
weights(icn) = icn.weights

function is_viable(icn::ICN, weights)
    _start = 0
    _end = 0

    for layer in layers(icn)
        _start = _end + 1
        _end += exclu(layer) ? nbits_exclu(layer) : length(layer)

        w = @view weights[_start:_end]

        !is_viable(layer, w) && return false
    end
    return true
end
is_viable(icn::ICN) = is_viable(icn, weights(icn))

"""
    weights!(icn, weights)
Set the weights of an ICN with a `BitVector`.
"""
function weights!(icn, weights)
    length(weights) == nbits(icn) || @warn icn weights nbits(icn)
    @assert length(weights) == nbits(icn)
    return icn.weights = weights
end

"""
    show_layers(icn)
Return a formatted string with each layers in the icn.
"""
show_layers(icn) = map(show_layer, layers(icn))

generate_weights(icn::ICN) = generate_weights(layers(icn))

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
            Σop += selected_size(layer, @view weights(icn)[_start:_end])
            Σmax += length(layer)
        end
    end
    return Σop / (Σmax + 1)
end

max_icn_length(icn = ICN(; param = [:val])) = length(icn.transformation)

"""
    _compose(icn)
Internal function called by `compose` and `show_composition`.
"""
function _compose(icn::ICN)
    !is_viable(icn) && (
        return (
            (x; X = zeros(length(x), max_icn_length()), param = nothing, dom_size = 0) -> typemax(Float64)
        ),
        []
    )

    funcs = Vector{Vector{Function}}()
    symbols = Vector{Vector{Symbol}}()

    _start = 0
    _end = 0

    for layer in layers(icn)
        _start = _end + 1
        _end += exclu(layer) ? nbits_exclu(layer) : length(layer)

        if exclu(layer)
            f_id = as_int(@view weights(icn)[_start:_end])
            # @warn "debug" f_id _end _start weights(icn) (exclu(layer) ? "nbits_exclu(layer)" : "length(layer)") (@view weights(icn)[_start:_end])
            s = symbol(layer, f_id + 1)
            push!(funcs, [functions(layer)[s]])
            push!(symbols, [s])
        else
            layer_funcs = Vector{Function}()
            layer_symbs = Vector{Symbol}()
            for (f_id, b) in enumerate(@view weights(icn)[_start:_end])
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

    function composition(x; X = zeros(length(x), length(funcs[1])), dom_size, params...)
        tr_in(Tuple(funcs[1]), X, x; params...)
        X[1:length(x), 1] .=
            1:length(x) .|> (i -> funcs[2][1](@view X[i, 1:length(funcs[1])]))
        return (y -> funcs[4][1](y; dom_size, nvars = length(x), params...))(
            funcs[3][1](@view X[:, 1]),
        )
    end

    return composition, symbols
end

abstract type AbstractICN end

function extract_params(fnexprs::AbstractVector{JLFunction}, parameters::NamedTuple)
	v = falses(length(fnexprs))
	keynames = keys(parameters)
	for i in 1:length(fnexprs)
		exprs = fnexprs[i].kwargs
		v[i] = if exprs == [:(params...)]
			true
		else
			flag = falses(length(exprs))
			for j in 1:length(exprs)-1
				for k in 1:length(keynames)
					has_symbol(exprs[j], keynames[k]) && (flag[j] = true)
				end
			end
			!(false in flag)
		end
	end
	return findall(v)
end

struct ICNNew{S,T} <: AbstractICN where {T <: Union{AbstractICN, Nothing}, S <: Union{AbstractVector{<:AbstractLayer}, Nothing}}
	weights::AbstractVector{Bool}
	parameters::NamedTuple
	layers::S
	connection::Tuple{Vararg{Tuple{Vararg{Int}}}}
	icn::T
	function ICN(; weights = BitVector[], parameters = (), layers = nothing, connection = ((),), icn = nothing)
		len = [length(layer.fn) for layer in layers]
		if isempty(weights)
			ind_weight = Array{BitVector}(undef, length(len))
			for (i,layer) in enumerate(layers)
				l = length(layer.fn)
				ind_weight[i] = if !layer.mutex
					BitVector(rand(Bool, l))
				else
					b = falses(l)
					b[rand(1:l)] = true
					b
				end
			end
			weights = vcat(ind_weight...)
		end
		@assert length(weights) === sum(len)
		consider = Int[]
		index = 0
		for (i, layer) in enumerate(layers)
			parindex = extract_params(layer.fnexprs, parameters)
			append!(consider, parindex .+ index)
			index += len[i]
		end
		new{typeof(layers), typeof(icn)}(@view(weights[consider]), parameters, layers, connection, icn)
	end
end
