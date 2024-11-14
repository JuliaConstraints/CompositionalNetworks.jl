abstract type AbstractICN end

function extract_params(fnexprs, parameters)
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

function check_weights_validity(icn::AbstractICN, weights::AbstractVector{Bool})
    @assert length(weights) === sum(icn.weightlen)
    index = 0
    for (i, layer) in enumerate(icn.layers)
        if layer.mutex
            sum = 0
            for j in weights[(1:icn.weightlen[i]).+index]
                sum += j
                sum > 1 && return false
            end
        end
        index += icn.weightlen[i]
    end
    return true
end

function apply!(icn::AbstractICN, weights::BitVector)::Union{<:AbstractICN,Nothing}
    icn.weights .= weights
    return if check_weights_validity(icn, weights)
        icn
    else
        nothing
    end
end

function evaluate(icn::AbstractICN, config::Configuration)
    input = config.x
    index = 0
    ind = icn.weights.indices
    icn.parameters.num_variables.x = length(config)
    connected_layers = [icn.layers[i] for i in icn.connection]
    for (i, layer) in enumerate(connected_layers)
        off = (1:icn.weightlen[i]) .+ index
        considerfns = values(layer.fn)[ind[off]]
        output = if layer.mutex
            considerfns[findfirst(icn.weights[off])](input)
        else
            o = []
            num_size = 10
            for j in considerfns[findall(icn.weights[off])]
                push!(o, j(input))
            end
            o
        end

        index += icn.weightlen[i]
        input = output
    end
    return output
end

function evaluate(icn::Nothing, config::Configuration)
    return Inf
end

(icn::AbstractICN)(weights::BitVector) = apply!(icn, weights)
(icn::Union{Nothing,AbstractICN})(config::Configuration) = evaluate(icn, config)

struct ICN{S,T} <: AbstractICN where {T <: Union{AbstractICN, Nothing}, S <: Union{AbstractVector{<:AbstractLayer}, Nothing}}
	weights::AbstractVector{Bool}
	parameters::NamedTuple
	layers::S
	connection::Union{Tuple{Vararg{Tuple{Vararg{Int}}}}, Tuple{Vararg{Int}}}
	weightlen::AbstractVector{Int}
	icn::T
	function ICN(; weights = BitVector[], parameters = (), layers = [Transformation, Arithmetic, Aggregation, Comparison], connection = (1,2,3,4), icn = nothing)
		
		len = [length(layer.fn) for layer in layers]
		parindexes = [extract_params(layer.fnexprs, parameters) for layer in layers]
		weightlen = length.(parindexes)
		
		weights = if isempty(weights)
			ind_weight = Array{BitVector}(undef, length(len))
			# initialization of weights
			for (i,layer) in enumerate(layers)
				l = length(layer.fn)
				ind_weight[i] = if !(layer.mutex)
					BitVector(rand(Bool, l))
				else
					b = falses(l)
					b[rand(parindexes[i])] = true
					@info b
					b
				end
			end
			vcat(ind_weight...)
		else
			# Checking the provided weights for if they match mutex or not
			# TODO: Ask Jefu if this is required or not
			####################
			index = 0
			for (i, layer) in enumerate(layers)
				if layer.mutex && !(sum(weights[parindexes[i] .+ index]) == 1 && sum(weights[1:len[i]] .+ index) == 1)
					error("Invalid weights provided")
				end
				index += length(layer.fn)
			end
			####################
			weights
		end
		@warn weights
		@assert length(weights) === sum(len)
		index, jindex = 0, 0
		consider = Array{Int}(undef, sum(length.(parindexes)))
		for (i, layer) in enumerate(layers)
			consider[(1:length(parindexes[i])) .+ jindex] .= parindexes[i] .+ index
			index += len[i]
			jindex += length(parindexes[i])
		end
		@error consider
		new{typeof(layers), typeof(icn)}(@view(weights[consider]), parameters, layers, connection, weightlen, icn)
	end
end

icn(parameters) = ICN(
	parameters = parameters,
	layers = [Transformation, Arithmetic, Aggregation, Comparison],
	connection = (1,2,3,4),
)
