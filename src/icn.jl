abstract type AbstractICN end

#=
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
=#

function check_weights_validity(icn::AbstractICN, weights::AbstractVector{Bool})
    @assert length(weights) === sum(icn.weightlen)
    offset = 1
    for (i, layer) in enumerate(icn.layers)
        index = offset:(offset + icn.weightlen[i] - 1)

        flag = if layer.mutex
            sum(icn.weights[index]) == 1
        else
            sum(icn.weights[index]) >= 1
        end
        if !flag
            return false
        end
        offset += icn.weightlen[i]
    end
    return true
end

function generate_new_valid_weights(
        layers::T,
        weightlen::Vector{Int}
) where {T <: AbstractVector{<:AbstractLayer}}
    weights = Array{Bool}(undef, sum(weightlen))
    offset = 1
    for (i, layer) in enumerate(layers)
        index = offset:(offset + weightlen[i] - 1)
        # @info index weightlen[i] weights[offset]
        weights[index] .= if layer.mutex
            temp = falses(weightlen[i])
            temp[rand(1:length(temp))] = true
            temp
        else
            rand(Bool, weightlen[i])
        end
        offset += weightlen[i]
    end
    return weights
end

function generate_new_valid_weights!(icn::T) where {T <: AbstractICN}
    icn.weights .= generate_new_valid_weights(icn.layers, icn.weightlen)
    nothing
end

function apply!(icn::AbstractICN, weights::AbstractVector{Bool})::Bool
    icn.weights .= weights
    return check_weights_validity(icn, weights)
end

function evaluate(
        icn::AbstractICN,
        config::Configuration;
        weights_validity = true,
        parameters...
)
    if weights_validity
        input = config.x
        # @warn icn.weights icn.weightlen
        weightoffset = 1
        lengthoff = 0
        for (i, layer) in enumerate(icn.layers)
            weightrange = weightoffset:(weightoffset + icn.weightlen[i] - 1)
            considerweights = icn.weights.indices[1][weightrange] .- lengthoff

            # @error considerweights findall(icn.weights[weightrange]) weightrange
            considerweights = considerweights[findall(icn.weights[weightrange])]

            considerfns = [layer.fn[i] for i in considerweights]
            output = nothing
            # @info layer.name output layer.argtype[1] layer.argtype[2] input considerweights layer.mutex considerfns
            input = layer.mutex ? considerfns[1](input; parameters...) :
                    [j(input; parameters...) for j in considerfns]
            # @warn "What?" input
            #input = output
            weightoffset += icn.weightlen[i]
            lengthoff += length(layer.fn)
        end
        return Float64(input)
    else
        return Inf
    end
end

evaluate(
        icn::AbstractICN,
        config::AbstractVector;
        weights_validity = true,
        parameters...
) = evaluate(icn, UnknownSolution(config); weights_validity = weights_validity, parameters...)

function evaluate(
        icns::Vector{<:AbstractICN},
        config::Configuration;
        weights_validity = trues(length(icns)),
        parameters...
)
    evaluation_output = Array{Float64}(undef, length(icns))
    for (i, icn) in enumerate(icns)
        # @info weights_validity[i], parameters, icn.parameters
        evaluation_output[i] = evaluate(
            icn, config; weights_validity = weights_validity[i], parameters...)
    end
    return sum(evaluation_output) / length(evaluation_output)
end

function evaluate(
        icn_validity::Pair{<:AbstractICN, Bool},
        config::Configuration;
        parameters...
)
    evaluate(
        icn_validity[1],
        config;
        weights_validity = icn_validity[2],
        icn_validity[1].constants...,
        parameters...
    )
end

function evaluate(
        icns::Vector{Pair{<:AbstractICN, Bool}},
        config::Configuration;
        parameters...
)
    evaluation_output = Array{Float64}(undef, length(icns))
    vals = if haskey(parameters, :vals)
        parameters[:vals]
    else
        nothing
    end
    param = Base.structdiff((; parameters...,), NamedTuple{(:vals,)})
    params = [(val = i, param...) for i in vals]

    for (i, icn_validity) in enumerate(icns)
        # @info weights_validity[i], parameters, icn.parameters
        evaluation_output[i] = evaluate(
            icn_validity[1],
            config;
            weights_validity = icn_validity[2],
            icn_validity[1].constants...,
            params[i]...
        )
    end
    return sum(evaluation_output) / length(evaluation_output)
end

#=
function evaluate(icn::Nothing, config::Configuration)
    return Inf
end
=#

(icn::AbstractICN)(weights::AbstractVector{Bool}) = apply!(icn, weights)
(icn::AbstractICN)(config::Configuration) = evaluate(icn, config)

struct ICN{S} <: AbstractICN where {S <: Union{AbstractVector{<:AbstractLayer}, Nothing}}
    weights::AbstractVector{Bool}
    parameters::Set{Symbol}
    layers::S
    connection::Vector{UInt32}
    weightlen::AbstractVector{Int}
    constants::Dict
    function ICN(;
        weights = AbstractVector{Bool}[],
        parameters = Symbol[],
        layers = [Transformation, Arithmetic, Aggregation, Comparison],
        connection = UInt32[1, 2, 3, 4],
        constants = Dict()
    )
        len = [length(layer.fn) for layer in layers]

        parindexes = Vector{Int}[]
        for layer in layers
            lfn = Int[]
            for (j, fn) in enumerate(layer.fn)
                par = extract_parameters(
                    fn,
                    parameters = append!(
                        copy(USUAL_CONSTRAINT_PARAMETERS),
                        [:numvars, :dom_size, :op_filter, :filter_val]
                    )
                )
                if !isempty(par)
                    if intersect(par[1], parameters) == par[1]
                        push!(lfn, j)
                    end
                else
                    push!(lfn, j)
                end
            end
            push!(parindexes, lfn)
        end

        # parindexes = [extract_params(layer.fnexprs, parameters) for layer in layers]
        weightlen = length.(parindexes)

        index, jindex = 0, 0
        consider = Array{Int}(undef, sum(length.(parindexes)))
        for (i, layer) in enumerate(layers)
            consider[(1:length(parindexes[i])) .+ jindex] .= parindexes[i] .+ index
            index += len[i]
            jindex += length(parindexes[i])
        end

        weights = if isempty(weights)
            w = falses(sum(len))
            #@info consider w generate_valid_weights(layers, weightlen)
            w[consider] .= generate_new_valid_weights(layers, weightlen)
            w
        else
            # Checking the provided weights for if they match mutex or not
            # TODO: Ask Jefu if this is required or not
            ####################
            index = 0
            for (i, layer) in enumerate(layers)
                if layer.mutex && !(
                    sum(weights[parindexes[i] .+ index]) == 1 &&
                    sum(weights[1:len[i]] .+ index) == 1
                )
                    error("Invalid weights provided")
                end
                index += length(layer.fn)
            end
            ####################
            weights
        end
        # @warn weights weights[consider]
        @assert length(weights) === sum(len)

        # @error consider
        # @info parameters
        new{typeof(layers)}(
            @view(weights[consider]),
            Set(parameters),
            layers,
            connection,
            weightlen,
            constants
        )
    end
end

function regularization(icn::AbstractICN)
    max_op = 0
    op = 0
    start = 1
    for (i, layer) in enumerate(icn.layers)
        if !layer.mutex
            op += length(findall(icn.weights[start:(start + icn.weightlen[i] - 1)]))
            max_op += icn.weightlen[i]
        end
        start += icn.weightlen[i]
    end
    return op / (max_op + 1)
end

function create_icn(icn::ICN, parameters::Vector{Symbol})
    ICN(
        weights = icn.weights,
        parameters = parameters,
        layers = icn.layers,
        connection = icn.connection
    )
end

#FIXME - This is a temporary fix
max_icn_length(args...; kargs...) = 42
