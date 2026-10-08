abstract type AbstractICN end

_operation_keys(functions::NamedTuple) = keys(functions)
_operation_keys(functions) = collect(keys(functions))

function _selected_weight_count(weights, positions)
    selected = 0
    for position in positions
        selected += weights[position]
    end
    return selected
end

function _selected_layer_names(functions, weights, indices, positions, function_offset)
    operation_keys = _operation_keys(functions)
    operations = Symbol[]
    for position in positions
        weights[position] || continue
        push!(operations, operation_keys[indices[position] - function_offset])
    end
    return operations
end

function _selected_operation_names(
        icn::AbstractICN, compact_weights::AbstractVector{Bool})
    length(compact_weights) == length(icn.weights) || throw(DimensionMismatch(
        "received $(length(compact_weights)) weights for an ICN with " *
        "$(length(icn.weights)) weights",
    ))
    parent_indices = parentindices(icn.weights)[1]
    selected = Vector{Vector{Symbol}}(undef, length(icn.layers))
    compact_offset = 1
    function_offset = 0
    for (index, layer) in enumerate(icn.layers)
        compact_range = compact_offset:(compact_offset + icn.weightlen[index] - 1)
        selected[index] = _selected_layer_names(layer.fn, compact_weights, parent_indices,
            compact_range, function_offset)
        compact_offset += icn.weightlen[index]
        function_offset += length(layer.fn)
    end
    return selected
end

function _pipeline_shape_valid(icn::AbstractICN, weights::AbstractVector{Bool})
    selected = _selected_operation_names(icn, weights)
    for (index, layer) in pairs(icn.layers)
        operations = selected[index]
        if layer.name === :Transformation &&
           any(op -> op in (:disjoint_pair_differences, :nonzero, :first_equal_position), operations) &&
           length(operations) != 1
            # Data-dependent or reducing shapes cannot share a transformation
            # block with another vector shape. Repetition belongs to the graph.
            return false
        end
        if layer.name === :PairMask && :zero_extent_groups in operations
            index > 1 || return false
            icn.layers[index - 1].name === :PairedMap || return false
            selected[index - 1] == [:pairwise_oriented_affine_margins] || return false
        end
        if layer.name === :Arithmetic && :difference in operations
            index > 1 || return false
            icn.layers[index - 1].mutex && return false
            length(selected[index - 1]) == 2 || return false
        end
    end
    return true
end

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

        selected_count = _selected_weight_count(weights, index)
        flag = layer.mutex ? selected_count == 1 : selected_count >= 1
        if !flag
            return false
        end
        offset += icn.weightlen[i]
    end
    return _pipeline_shape_valid(icn, weights)
end

function generate_new_valid_weights(
        layers::AbstractVector,
        weightlen::Vector{Int}
)
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
    while true
        candidate = generate_new_valid_weights(icn.layers, icn.weightlen)
        apply!(icn, candidate) && return true
    end
end

function apply!(icn::AbstractICN, weights::AbstractVector{Bool})::Bool
    length(weights) == length(icn.weights) || throw(DimensionMismatch(
        "received $(length(weights)) weights for an ICN with $(length(icn.weights)) weights"
    ))
    @inbounds for index in eachindex(icn.weights)
        icn.weights[index] = weights[index]
    end
    return check_weights_validity(icn, icn.weights)
end

function apply!(
        icn::AbstractICN,
        weights::AbstractDictionary{<:Integer,Bool}
)::Bool
    length(weights) == length(icn.weights) || throw(DimensionMismatch(
        "received $(length(weights)) weights for an ICN with $(length(icn.weights)) weights"
    ))
    @inbounds for index in eachindex(icn.weights)
        icn.weights[index] = weights[index]
    end
    return check_weights_validity(icn, icn.weights)
end

@testitem "ICN weights accept LocalSearchSolvers dictionary assignments" begin
    using Dictionaries: Dictionary
    using Test

    network = ICN()
    weights = collect(network.weights)
    assignment = Dictionary(collect(eachindex(weights)), weights)

    @test apply!(network, assignment) == check_weights_validity(network, weights)
    @test collect(network.weights) == weights
    @test_throws DimensionMismatch apply!(
        network,
        Dictionary([1], Bool[true]),
    )
end

@testitem "Multiple transformations preserve their declared numeric container type" begin
    using Test

    network = ICN()
    fill!(network.weights, false)
    let offset = 1
        for (layer_index, layer) in enumerate(network.layers)
            layer_weights = offset:(offset + network.weightlen[layer_index] - 1)
            network.weights[first(layer_weights)] = true
            if layer === Transformation && length(layer_weights) > 1
                network.weights[first(layer_weights) + 1] = true
            end
            offset = last(layer_weights) + 1
        end
    end

    @test check_weights_validity(network, network.weights)
    @test evaluate(network, Solution([1, 2, 1])) isa Real
end

const _PAIRWISE_AFFINE_MARGINS_INDEX =
    findfirst(==(:pairwise_oriented_affine_margins), keys(PairedMap.fn))
const _ZERO_EXTENT_GROUPS_INDEX =
    findfirst(==(:zero_extent_groups), keys(PairMask.fn))
const _GROUP_MINIMUM_INDEX = findfirst(==(:minimum), keys(GroupReduction.fn))
const _TRANSFORMATION_ID_INDEX = findfirst(==(:id), keys(Transformation.fn))
const _POSITIVE_PART_INDEX = findfirst(==(:positive_part), keys(Transformation.fn))
const _ARITHMETIC_SUM_INDEX = findfirst(==(:sum), keys(Arithmetic.fn))
const _AGGREGATION_SUM_INDEX = findfirst(==(:sum), keys(Aggregation.fn))
const _AGGREGATION_COUNT_POSITIVE_INDEX =
    findfirst(==(:count_positive), keys(Aggregation.fn))
const _AGGREGATION_MAXIMUM_INDEX = findfirst(==(:maximum), keys(Aggregation.fn))
const _AGGREGATION_MINIMUM_INDEX = findfirst(==(:minimum), keys(Aggregation.fn))
const _COMPARISON_ID_INDEX = findfirst(==(:id), keys(Comparison.fn))

@inline function _only_selected_operation_index(
        weights,
        weight_indices,
        weight_lengths,
        layer_index::Int,
        weight_offset::Int,
        function_offset::Int,
)
    selected = 0
    @inbounds for index in 1:weight_lengths[layer_index]
        weight_index = weight_offset + index - 1
        parent_index = weight_indices[weight_index]
        weights[parent_index] || continue
        iszero(selected) || return nothing
        selected = parent_index - function_offset
    end
    iszero(selected) && return nothing
    return selected
end

"Evaluate a recognized structural composition without materializing layer outputs."
@inline function _evaluate_structural_specialization(icn::AbstractICN, x; parameters...)
    return _evaluate_structural_specialization(
        icn.layers, icn.weights, icn.weightlen, x; parameters...,
    )
end

@inline function _evaluate_structural_specialization(
        layers,
        weight_view,
        weight_lengths,
        x;
        parameters...,
)
    length(layers) == 7 || return nothing
    layers[1] === PairedMap || return nothing
    layers[2] === PairMask || return nothing
    layers[3] === GroupReduction || return nothing
    layers[4] === Transformation || return nothing
    layers[5] === Arithmetic || return nothing
    layers[6] === Aggregation || return nothing
    layers[7] === Comparison || return nothing

    weight_offset = 1
    function_offset = 0
    weights = parent(weight_view)
    weight_indices = parentindices(weight_view)[1]
    front = _only_selected_operation_index(
        weights, weight_indices, weight_lengths, 1, weight_offset, function_offset)
    weight_offset += weight_lengths[1]
    function_offset += length(PairedMap.fn)
    mask = _only_selected_operation_index(
        weights, weight_indices, weight_lengths, 2, weight_offset, function_offset)
    weight_offset += weight_lengths[2]
    function_offset += length(PairMask.fn)
    group_reduction = _only_selected_operation_index(
        weights, weight_indices, weight_lengths, 3, weight_offset, function_offset)
    weight_offset += weight_lengths[3]
    function_offset += length(GroupReduction.fn)
    transformation = _only_selected_operation_index(
        weights, weight_indices, weight_lengths, 4, weight_offset, function_offset)
    weight_offset += weight_lengths[4]
    function_offset += length(Transformation.fn)
    arithmetic = _only_selected_operation_index(
        weights, weight_indices, weight_lengths, 5, weight_offset, function_offset)
    weight_offset += weight_lengths[5]
    function_offset += length(Arithmetic.fn)
    aggregation = _only_selected_operation_index(
        weights, weight_indices, weight_lengths, 6, weight_offset, function_offset)
    weight_offset += weight_lengths[6]
    function_offset += length(Aggregation.fn)
    comparison = _only_selected_operation_index(
        weights, weight_indices, weight_lengths, 7, weight_offset, function_offset)

    front == _PAIRWISE_AFFINE_MARGINS_INDEX || return nothing
    mask == _ZERO_EXTENT_GROUPS_INDEX || return nothing
    group_reduction == _GROUP_MINIMUM_INDEX || return nothing
    arithmetic == _ARITHMETIC_SUM_INDEX || return nothing
    comparison == _COMPARISON_ID_INDEX || return nothing
    positive_part = transformation == _POSITIVE_PART_INDEX
    identity_count = transformation == _TRANSFORMATION_ID_INDEX &&
                     aggregation == _AGGREGATION_COUNT_POSITIVE_INDEX
    (positive_part || identity_count) || return nothing
    haskey(parameters, :pair_vars) || return nothing
    pair_vars = parameters[:pair_vars]
    dimensions = get(parameters, :dim, 1)
    zero_ignored = get(parameters, :bool, true)
    if aggregation == _AGGREGATION_SUM_INDEX
        return _aggregate_pairwise_disjunction(
            Val(:sum), x; pair_vars, dim = dimensions, bool = zero_ignored,
        )
    elseif aggregation == _AGGREGATION_COUNT_POSITIVE_INDEX
        return _aggregate_pairwise_disjunction(
            Val(:count_positive), x; pair_vars, dim = dimensions, bool = zero_ignored,
        )
    elseif aggregation == _AGGREGATION_MAXIMUM_INDEX
        return _aggregate_pairwise_disjunction(
            Val(:maximum), x; pair_vars, dim = dimensions, bool = zero_ignored,
        )
    elseif aggregation == _AGGREGATION_MINIMUM_INDEX
        return _aggregate_pairwise_disjunction(
            Val(:minimum), x; pair_vars, dim = dimensions, bool = zero_ignored,
        )
    end
    return nothing
end

# Keep the changing input and concrete layer behind a function barrier. Selection
# uses compact weight positions mapped into the full operation catalog; it must
# be read on every call because apply! mutates the weights during local search.
function _evaluate_selected_layer(layer, weights, indices, positions, function_offset,
        input; parameters...)
    if layer.mutex
        for position in positions
            if weights[position]
                return layer.fn[indices[position] - function_offset](input; parameters...)
            end
        end
        # As in the materialized selection, an empty mutex layer is invalid.
        throw(BoundsError(Function[], 1))
    end
    selected_count = 0
    for position in positions
        selected_count += weights[position]
    end
    output_type = last(layer.argtype)
    outputs = Vector{output_type}(undef, selected_count)
    output_index = 1
    for position in positions
        if weights[position]
            outputs[output_index] = layer.fn[indices[position] - function_offset](input; parameters...)
            output_index += 1
        end
    end
    return outputs
end

function evaluate(
        icn::AbstractICN,
        config::Configuration;
        weights_validity = true,
        parameters...
)
    if weights_validity
        input = config.x
        specialized = _evaluate_structural_specialization(icn, input; parameters...)
        isnothing(specialized) || return Float64(specialized)
        weightoffset = 1
        lengthoff = 0
        indices = parentindices(icn.weights)[1]
        for (i, layer) in enumerate(icn.layers)
            weightrange = weightoffset:(weightoffset + icn.weightlen[i] - 1)
            input = _evaluate_selected_layer(layer, icn.weights, indices, weightrange,
                lengthoff, input; parameters...)
            weightoffset += icn.weightlen[i]
            lengthoff += length(layer.fn)
        end
        return Float64(input)
    else
        return Inf
    end
end

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
        reduction::Symbol = :mean,
        reduction_op::Function = (==),
        reduction_val::Integer = 0,
        reduction_weights = nothing,
        reduction_workspace = nothing,
        parameters...
)
    evaluation_output = if isnothing(reduction_workspace)
        Vector{Float64}(undef, length(icns))
    else
        length(reduction_workspace) >= length(icns) || throw(DimensionMismatch(
            "the reduction workspace must have one element per component ICN",
        ))
        @view reduction_workspace[1:length(icns)]
    end
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
    return reduce_icn_outputs(
        evaluation_output,
        reduction;
        op = reduction_op,
        val = reduction_val,
        weights = reduction_weights,
    )
end

"""
    reduce_icn_outputs(errors, reduction; op = ==, val = 0, weights = nothing)

Combine the non-negative error values produced by one ICN per parameter value.
The reduction defines the quantifier independently from each component ICN:

- `:exists_min` and `:exists_product` are zero when at least one component is zero;
- `:forall_max`, `:forall_sum`, and the backward-compatible `:mean` are zero
  when every component is zero;
- `:count` converts component zeroes to Boolean indicators and returns an
  integer violation degree for `op(count, val)`.

Optional non-negative `weights` scale the component errors. They do not change
the zero set as long as every weight is strictly positive.
"""
function reduce_icn_outputs(
        errors::AbstractVector{<:Real},
        reduction::Symbol;
        op::Function = (==),
        val::Integer = 0,
        weights = nothing,
)
    isempty(errors) && return _empty_reduction(reduction, op, val)
    if !isnothing(weights)
        length(weights) == length(errors) ||
            throw(DimensionMismatch("one reduction weight is required per ICN output"))
        all(>(0), weights) || throw(ArgumentError("ICN reduction weights must be positive"))
    end
    reduction === :exists_min && return _weighted_minimum(errors, weights)
    reduction === :exists_product && return _weighted_product(errors, weights)
    reduction === :forall_max && return _weighted_maximum(errors, weights)
    reduction === :forall_sum && return _weighted_sum(errors, weights)
    reduction === :mean && return _weighted_sum(errors, weights) / length(errors)
    reduction === :count && return _count_violation(count(iszero, errors), op, val)
    throw(ArgumentError("unknown ICN output reduction: $reduction"))
end

"""
    reduce_icn_outputs(error_function, values, reduction; kwargs...)

Evaluate and reduce parameterized ICN outputs lazily. Unlike the vector method,
this form does not materialize every output and can short-circuit existential
reductions as soon as their value is zero.
"""
function reduce_icn_outputs(
        error_function::F,
        values,
        reduction::Symbol;
        op::Function = (==),
        val::Integer = 0,
        weights = nothing,
) where {F}
    isempty(values) && return _empty_reduction(reduction, op, val)
    if !isnothing(weights)
        length(weights) == length(values) || throw(DimensionMismatch(
            "one reduction weight is required per parameter value",
        ))
        all(>(0), weights) || throw(ArgumentError("ICN reduction weights must be positive"))
    end

    if reduction === :exists_min
        result = Inf
        @inbounds for (index, item) in pairs(values)
            result = min(result, _weighted_output(error_function(item), weights, index))
            iszero(result) && return result
        end
        return result
    elseif reduction === :exists_product
        result = 1.0
        @inbounds for (index, item) in pairs(values)
            result *= _weighted_output(error_function(item), weights, index)
            iszero(result) && return result
        end
        return result
    elseif reduction === :forall_max
        result = -Inf
        @inbounds for (index, item) in pairs(values)
            result = max(result, _weighted_output(error_function(item), weights, index))
        end
        return result
    elseif reduction === :forall_sum || reduction === :mean
        result = 0.0
        @inbounds for (index, item) in pairs(values)
            result += _weighted_output(error_function(item), weights, index)
        end
        return reduction === :mean ? result / length(values) : result
    elseif reduction === :count
        count_value = 0
        @inbounds for item in values
            count_value += iszero(error_function(item))
        end
        return _count_violation(count_value, op, val)
    end
    throw(ArgumentError("unknown ICN output reduction: $reduction"))
end

Base.@propagate_inbounds _weighted_output(value, ::Nothing, index) = value
Base.@propagate_inbounds _weighted_output(value, weights, index) = weights[index] * value

"""
    icn_zero_set(error_function, values, reduction; op = ==, val = 0, weights = nothing)

Decide whether a non-negative ICN reduction is zero without computing its full
penalty. Existential and universal quantifiers short-circuit; standard cardinality
comparisons also stop when the remaining values cannot change the decision.
"""
function icn_zero_set(
        error_function::F,
        values,
        reduction::Symbol;
        op::Function = (==),
        val::Integer = 0,
        weights = nothing,
) where {F}
    if !isnothing(weights)
        length(weights) == length(values) || throw(DimensionMismatch(
            "one reduction weight is required per parameter value",
        ))
        all(>(0), weights) || throw(ArgumentError("ICN reduction weights must be positive"))
    end
    if reduction === :exists_min || reduction === :exists_product
        @inbounds for item in values
            iszero(error_function(item)) && return true
        end
        return false
    elseif reduction === :forall_max || reduction === :forall_sum || reduction === :mean
        @inbounds for item in values
            iszero(error_function(item)) || return false
        end
        return true
    elseif reduction === :count
        return _count_zero_set(error_function, values, op, val)
    end
    throw(ArgumentError("unknown ICN output reduction: $reduction"))
end

icn_zero_set(errors::AbstractVector{<:Real}, reduction::Symbol; kwargs...) =
    icn_zero_set(identity, errors, reduction; kwargs...)

@inline function _count_zero_set(
        error_function::F, values, op::Function, target::Integer) where {F}
    successes = 0
    total = length(values)
    decision = _bounded_count_decision(successes, total, op, target)
    isnothing(decision) || return decision
    @inbounds for (processed, item) in enumerate(values)
        successes += iszero(error_function(item))
        remaining = total - processed
        decision = _bounded_count_decision(successes, remaining, op, target)
        isnothing(decision) || return decision
    end
    return op(successes, target)
end

@inline function _bounded_count_decision(
        successes::Integer, remaining::Integer, op::Function, target::Integer)
    if op === (>=)
        successes >= target && return true
        successes + remaining < target && return false
    elseif op === (>)
        successes > target && return true
        successes + remaining <= target && return false
    elseif op === (<=)
        successes > target && return false
        successes + remaining <= target && return true
    elseif op === (<)
        successes >= target && return false
        successes + remaining < target && return true
    elseif op === (==)
        (successes > target || successes + remaining < target) && return false
        iszero(remaining) && return successes == target
    elseif op === (!=)
        (successes > target || successes + remaining < target) && return true
        iszero(remaining) && return successes != target
    end
    return nothing
end

@inline _weighted_value(errors, ::Nothing, index) = @inbounds errors[index]
@inline _weighted_value(errors, weights, index) =
    @inbounds weights[index] * errors[index]

function _weighted_sum(errors, weights)
    total = zero(promote_type(eltype(errors), isnothing(weights) ? Bool : eltype(weights)))
    @inbounds for index in eachindex(errors)
        total += _weighted_value(errors, weights, index)
    end
    return total
end

function _weighted_product(errors, weights)
    total = one(promote_type(eltype(errors), isnothing(weights) ? Bool : eltype(weights)))
    @inbounds for index in eachindex(errors)
        total *= _weighted_value(errors, weights, index)
    end
    return total
end

function _weighted_minimum(errors, weights)
    value = _weighted_value(errors, weights, firstindex(errors))
    @inbounds for index in (firstindex(errors) + 1):lastindex(errors)
        value = min(value, _weighted_value(errors, weights, index))
    end
    return value
end

function _weighted_maximum(errors, weights)
    value = _weighted_value(errors, weights, firstindex(errors))
    @inbounds for index in (firstindex(errors) + 1):lastindex(errors)
        value = max(value, _weighted_value(errors, weights, index))
    end
    return value
end

function _empty_reduction(reduction::Symbol, op::Function, val::Integer)
    reduction === :exists_min && return Inf
    reduction === :exists_product && return 1.0
    reduction === :forall_max && return 0.0
    reduction === :forall_sum && return 0.0
    reduction === :mean && return 0.0
    reduction === :count && return _count_violation(0, op, val)
    throw(ArgumentError("unknown ICN output reduction: $reduction"))
end

@inline function _count_violation(count_value::Integer, op::Function, target::Integer)
    op(count_value, target) && return 0
    op === (==) && return abs(count_value - target)
    op === (<=) && return max(0, count_value - target)
    op === (<) && return max(0, count_value - target + 1)
    op === (>=) && return max(0, target - count_value)
    op === (>) && return max(0, target - count_value + 1)
    return 1
end

@testitem "Per-value ICNs have explicit quantifier reductions" begin
    using Test

    errors = [0.0, 2.0, 3.0]
    @test reduce_icn_outputs(errors, :exists_min) == 0.0
    @test reduce_icn_outputs(errors, :exists_product) == 0.0
    @test reduce_icn_outputs(errors, :forall_max) == 3.0
    @test reduce_icn_outputs(errors, :forall_sum) == 5.0
    @test reduce_icn_outputs(errors, :mean) == 5 / 3
    @test reduce_icn_outputs(errors, :count; op = (==), val = 1) == 0
    @test reduce_icn_outputs(errors, :count; op = (>=), val = 2) == 1
    @test reduce_icn_outputs([1.0, 2.0], :forall_sum; weights = [2.0, 3.0]) == 8.0
    @test_throws DimensionMismatch reduce_icn_outputs(errors, :forall_sum; weights = [1.0])
    @test_throws ArgumentError reduce_icn_outputs(errors, :forall_sum; weights = [1, 0, 1])
    @test reduce_icn_outputs(Float64[], :forall_sum) == 0.0
    @test isinf(reduce_icn_outputs(Float64[], :exists_min))
    @test reduce_icn_outputs(identity, errors, :exists_min) == 0.0
    @test reduce_icn_outputs(identity, errors, :forall_sum) == 5.0
    @test reduce_icn_outputs(identity, errors, :count; op = (>=), val = 2) == 1

    calls = Ref(0)
    counted(value) = (calls[] += 1; value)
    @test !icn_zero_set(counted, [1.0, 0.0, 0.0], :forall_sum)
    @test calls[] == 1
    calls[] = 0
    @test icn_zero_set(counted, [1.0, 0.0, 2.0], :exists_min)
    @test calls[] == 2
    calls[] = 0
    @test icn_zero_set(counted, [0.0, 1.0, 1.0], :count; op = (>=), val = 1)
    @test calls[] == 1
    calls[] = 0
    @test !icn_zero_set(counted, [1.0, 0.0, 0.0], :count; op = (==), val = 3)
    @test calls[] == 1
end

#=
function evaluate(icn::Nothing, config::Configuration)
    return Inf
end
=#

(icn::AbstractICN)(weights::AbstractVector{Bool}) = apply!(icn, weights)
(icn::AbstractICN)(config::Configuration) = evaluate(icn, config)

const _BASE_LAYERS = AbstractLayer[Transformation, Arithmetic, Aggregation, Comparison]

_parameter_layers(::Val, _) = AbstractLayer[]
_parameter_layers(::Val{:pair_vars}, value::AbstractVector) = AbstractLayer[PairedMap]
_parameter_layers(::Val{:pair_vars}, value::AbstractMatrix) =
    AbstractLayer[EventMap, SegmentMap]
_parameter_layers(::Val{:id}, _) = AbstractLayer[SimpleFilter]
_parameter_layers(::Val{:filter_val}, _) = AbstractLayer[SimpleFilter]
_parameter_layers(::Val{:vals}, value::AbstractVector) = AbstractLayer[SimpleFilter]
_parameter_layers(::Val{:vals}, value::AbstractMatrix) = AbstractLayer[SimpleFilter]
_parameter_layers(::Val{:language}, ::AbstractLanguage) = AbstractLayer[Language]

"""One executable ICN branch induced by a collection-valued concept parameter."""
struct ICNBranchPlan
    source::Symbol
    selector::Union{Nothing, Int}
    layers::Vector{AbstractLayer}
    parameters::Vector{Symbol}
end

"""
Structural alternatives derived exclusively from keyword names and runtime types.

`direct_layers` describes a single ICN consuming the supplied parameters. `branches`
describes the independent ICNs induced by vector `vals`, by the value/condition rows of
matrix `vals`, or by the rows of matrix-valued `pair_vars`. Matrix combinators and row
branches coexist; their selection and reduction remain learnable.
"""
struct ICNStructurePlan
    direct_layers::Union{Nothing, Vector{AbstractLayer}}
    branches::Vector{ICNBranchPlan}
    reductions::Vector{Symbol}
    dim::Union{Nothing, Int}
    boolean_guard::Bool
    groups::Vector{ICNStructurePlan}
end

ICNStructurePlan(direct_layers, branches, reductions, dim, boolean_guard) =
    ICNStructurePlan(direct_layers, branches, reductions, dim, boolean_guard,
        ICNStructurePlan[])

function _needs_post_arithmetic(parameters::NamedTuple)
    indexed = haskey(parameters, :id) && parameters.id isa Integer
    blocked = haskey(parameters, :dim) && parameters.dim isa Integer &&
              !haskey(parameters, :pair_vars)
    return indexed || blocked
end

function _ordered_layers(enabled, parameters::NamedTuple)
    layers = AbstractLayer[]
    for layer in (
            PairedMap, PairMask, GroupReduction, EventMap, SegmentMap,
            Language, SimpleFilter,
        )
        layer.name in enabled && push!(layers, layer)
    end
    # A matrix-valued `pair_vars` first becomes typed interval segments and is
    # then projected by SegmentMap. Ordinary value transformations do not accept
    # that typed structural intermediate, so this signature directly feeds the
    # arithmetic stage. This choice depends only on the keyword type/shape.
    suffix = if SegmentMap.name in enabled
        AbstractLayer[Arithmetic, Aggregation, Comparison]
    elseif _needs_post_arithmetic(parameters)
        AbstractLayer[
            Transformation, Arithmetic, Pointwise, Aggregation, Comparison,
        ]
    else
        _BASE_LAYERS
    end
    append!(layers, suffix)
    return layers
end

function _enabled_parameter_layers(parameters::NamedTuple; skip = ())
    enabled = Set{Symbol}()
    for (name, value) in pairs(parameters)
        name in skip && continue
        isnothing(value) && continue
        foreach(layer -> push!(enabled, layer.name), _parameter_layers(Val(name), value))
    end
    if !(:pair_vars in skip) && haskey(parameters, :pair_vars) &&
       parameters.pair_vars isa AbstractVector && haskey(parameters, :dim) &&
       parameters.dim isa Integer && haskey(parameters, :bool) &&
       parameters.bool isa Bool
        push!(enabled, PairMask.name)
        push!(enabled, GroupReduction.name)
    end
    return enabled
end

function _branch_parameters(parameters, source, replacement)
    names = Symbol[name for name in keys(parameters) if name != source]
    append!(names, replacement)
    return sort!(unique!(names); by = string)
end

"""
    structure_for_parameters(parameters::NamedTuple)

Derive the maximal current ICN structure from parameter names and types. Scalar `val`,
`op`, `dim` and `bool` never create a filter layer: they only enable compatible
operations, axes, reductions or Boolean validity guards. A vector `vals` offers one base
ICN per value. A matrix `vals` offers one row ICN whose first column is bound to
`filter_val` and whose remaining columns describe an occurrence condition; this
normalization depends only on the keyword name, its matrix type, and its shape. Matrix
`pair_vars` creates one paired branch per row.
"""
function structure_for_parameters(parameters::NamedTuple)
    dimension = haskey(parameters, :dim) && parameters.dim isa Integer ?
                Int(parameters.dim) : nothing
    boolean_guard = haskey(parameters, :bool) && parameters.bool isa Bool
    if haskey(parameters, :pair_vars) &&
       _grouped_parameter_rows(parameters.pair_vars)
        retained = (;
            (name => value for (name, value) in pairs(parameters) if
             name != :pair_vars)...,
        )
        groups = ICNStructurePlan[
            structure_for_parameters((; retained..., pair_vars = group))
            for group in parameters.pair_vars
        ]
        return ICNStructurePlan(
            nothing,
            ICNBranchPlan[],
            Symbol[:exists_min, :exists_product, :forall_sum, :forall_max, :count],
            dimension,
            boolean_guard,
            groups,
        )
    end
    enabled = _enabled_parameter_layers(parameters)
    matrix_vals = haskey(parameters, :vals) && parameters.vals isa AbstractMatrix
    nested_pair_rows = haskey(parameters, :pair_vars) &&
                       _nested_parameter_rows(parameters.pair_vars)
    direct_layers = matrix_vals || nested_pair_rows ? nothing :
                    _ordered_layers(enabled, parameters)
    branches = ICNBranchPlan[]
    reductions = Symbol[:direct]

    if haskey(parameters, :vals) && parameters.vals isa AbstractVector
        branch_enabled = _enabled_parameter_layers(parameters; skip = (:vals,))
        branch_layers = _ordered_layers(branch_enabled, parameters)
        branch_parameters = _branch_parameters(parameters, :vals, [:val])
        for index in eachindex(parameters.vals)
            push!(branches, ICNBranchPlan(:vals, Int(index), copy(branch_layers),
                copy(branch_parameters)))
        end
        append!(reductions,
            (:exists_min, :exists_product, :forall_max, :forall_sum, :mean, :count))
    elseif matrix_vals
        size(parameters.vals, 2) >= 1 || throw(ArgumentError(
            "matrix-valued vals requires at least one column",
        ))
        branch_enabled = _enabled_parameter_layers(parameters; skip = (:vals,))
        push!(branch_enabled, SimpleFilter.name)
        branch_layers = _ordered_layers(branch_enabled, parameters)
        branch_parameters = _branch_parameters(
            parameters, :vals, [:filter_val, :op, :val])
        for row in axes(parameters.vals, 1)
            push!(branches, ICNBranchPlan(:vals_row, Int(row), copy(branch_layers),
                copy(branch_parameters)))
        end
        if boolean_guard
            membership_parameters = _branch_parameters(parameters, :vals, [:vals])
            push!(branches, ICNBranchPlan(:vals_domain, nothing,
                copy(branch_layers), copy(membership_parameters)))
        end
        reductions = boolean_guard ? Symbol[:forall_sum, :guarded_sum] : Symbol[:forall_sum]
    end

    if haskey(parameters, :pair_vars) &&
       (parameters.pair_vars isa AbstractMatrix || nested_pair_rows)
        # Preserve row-wise ICNs while also offering matrix combinators. The latter are
        # enabled by name and type only; learning still decides whether to select them.
        branch_enabled = _enabled_parameter_layers(parameters; skip = (:pair_vars,))
        push!(branch_enabled, PairedMap.name)
        branch_layers = _ordered_layers(branch_enabled, parameters)
        branch_parameters = _branch_parameters(parameters, :pair_vars, [:pair_vars])
        empty!(branches)
        for row in axes(parameters.pair_vars, 1)
            push!(branches, ICNBranchPlan(:pair_vars_row, Int(row), copy(branch_layers),
                copy(branch_parameters)))
        end
        reductions = Symbol[
            :exists_min, :exists_product, :forall_sum, :forall_max, :count,
        ]
    elseif isempty(branches)
        push!(branches, ICNBranchPlan(:direct, nothing, copy(direct_layers),
            sort!(collect(keys(parameters)); by = string)))
    end

    unique!(reductions)
    return ICNStructurePlan(direct_layers, branches, reductions, dimension, boolean_guard)
end

function _nested_parameter_rows(value)
    value isa AbstractVector || return false
    isempty(value) && return eltype(value) <: AbstractVector
    return all(value) do row
        (row isa AbstractVector || row isa Tuple) &&
            all(item -> !_collection_parameter(item), row)
    end
end

function _grouped_parameter_rows(value)
    value isa Tuple || return false
    isempty(value) && return false
    return all(group -> _nested_parameter_rows(group), value)
end

@inline function _vals_row_condition(values::AbstractMatrix, row::Int)
    columns = size(values, 2)
    columns >= 1 || throw(ArgumentError(
        "matrix-valued vals requires at least one column",
    ))
    filter_val = @inbounds values[row, 1]
    columns == 1 && return (filter_val, (==), 1)
    target = @inbounds values[row, 2]
    # Explicit operator/target pairs and numeric interval columns have different
    # types. The row interpretation never depends on a constraint identifier.
    target isa Tuple{Function,Any} && return (filter_val, target[1], target[2])
    columns == 2 && return (filter_val, (==), target)
    upper = @inbounds values[row, 3]
    step = target <= upper ? one(target) : -one(target)
    return (filter_val, in, target:step:upper)
end

"""Materialize the keyword values consumed by one structural branch."""
function bind_branch_parameters(parameters::NamedTuple, branch::ICNBranchPlan)
    branch.source === :direct && return parameters
    source_name = if branch.source in (:vals_row, :vals_domain)
        :vals
    elseif branch.source === :pair_vars_row
        :pair_vars
    else
        branch.source
    end
    source_value = getproperty(parameters, source_name)
    retained = (;
        (name => value for (name, value) in pairs(parameters) if
         name != source_name &&
         !(branch.source === :vals_row && name in (:filter_val, :op, :val)))...,
    )
    if branch.source === :vals
        return (; retained..., val = source_value[something(branch.selector)])
    elseif branch.source === :vals_row
        row = something(branch.selector)
        filter_val, op, val = _vals_row_condition(source_value, row)
        return (; retained..., filter_val, op, val)
    elseif branch.source === :vals_domain
        return (; retained..., vals = @view(source_value[:, 1]))
    elseif branch.source === :pair_vars_row
        row = something(branch.selector)
        pair_vars = source_value isa AbstractMatrix ?
                    collect(@view source_value[row, :]) : collect(source_value[row])
        return (; retained..., pair_vars)
    elseif branch.source === :pair_vars
        return (;
            retained...,
            pair_vars = collect(@view source_value[something(branch.selector), :]),
        )
    end
    throw(ArgumentError("unsupported ICN branch source: $(branch.source)"))
end

"""
    layers_for_parameters(parameters::NamedTuple)

Select the existing ICN layers that are compatible with the names and runtime shapes of
the parameters supplied to a Boolean concept. This only selects layers from the current
grammar; it does not add operations or silently reinterpret matrix-valued parameters.
"""
function layers_for_parameters(parameters::NamedTuple)
    plan = structure_for_parameters(parameters)
    isnothing(plan.direct_layers) && throw(ArgumentError(
        "this matrix-valued parameter signature induces multiple ICN branches; use " *
        "structure_for_parameters to obtain its branch plan",
    ))
    return copy(plan.direct_layers)
end

layers_for_parameters() = copy(_BASE_LAYERS)

_collection_parameter(value) =
    value isa AbstractArray || value isa AbstractRange || value isa AbstractSet ||
    value isa Tuple

function _operation_parameter_compatible(layer::Symbol, operation::Symbol, parameters)
    isempty(parameters) && return true
    if haskey(parameters, :val) && _collection_parameter(parameters.val)
        if layer === :SimpleFilter && operation in (
                :filter_equal_val, :filter_ge_val, :filter_great_val,
                :filter_less_val, :filter_le_val, :filter_ne_val)
            return false
        elseif layer === :Transformation && operation in (
                :count_equal_val, :count_less_val, :count_great_val,
                :var_minus_val, :val_minus_var, :count_bounding_val)
            return false
        elseif layer === :Comparison && operation in (
                :abs_val, :val_minus_var, :var_minus_val,
                :euclidean_val, :euclidean_val_op)
            return false
        end
    end
    return true
end

# The original intersection comparison also rejects duplicate required names.
# Keep that contract while avoiding a temporary set and vector for short native
# symbol lists. Other collections and unusually wide signatures retain intersect.
function _parameter_names_match(names, parameters)
    names isa Vector{Symbol} &&
        (parameters isa Vector{Symbol} ||
         (parameters isa Vector{Union{}} && isempty(parameters))) &&
        length(names) <= 16 && length(parameters) <= 64 ||
        return intersect(names, parameters) == names
    isempty(parameters) && return isempty(names)
    for (index, name) in enumerate(names)
        name in parameters || return false
        for prior in 1:(index - 1)
            names[prior] === name && return false
        end
    end
    return true
end

struct ICN{S} <: AbstractICN where {S <: Union{AbstractVector{<:AbstractLayer}, Nothing}}
    weights::AbstractVector{Bool}
    parameters::Set{Symbol}
    layers::S
    connection::Vector{UInt32}
    weightlen::AbstractVector{Int}
    constants::Dict
    function ICN(;
            weights = BitVector[],
            parameters = Symbol[],
            parameter_values = NamedTuple(),
            layers = [Transformation, Arithmetic, Aggregation, Comparison],
            connection = UInt32[1, 2, 3, 4],
            constants = Dict()
    )
        len = [length(layer.fn) for layer in layers]
        parameter_names = collect(parameters)

        parindexes = Vector{Int}[]
        for layer in layers
            lfn = Int[]
            for (j, (operation, fn)) in enumerate(pairs(layer.fn))
                par = extract_parameters(
                    fn,
                    parameters = append!(
                        copy(USUAL_CONSTRAINT_PARAMETERS),
                        (:numvars, :dom_size, :op_filter, :filter_val)
                    )
                )
                names_match = isempty(par) ||
                              _parameter_names_match(par[1], parameter_names)
                if names_match && _operation_parameter_compatible(
                        layer.name, operation, parameter_values)
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
            Set(parameter_names),
            layers,
            connection,
            weightlen,
            constants
        )
    end
end

@testitem "Parameter signatures select compatible ICN layers" begin
    using Test

    scalar = layers_for_parameters((; val = 2, op = ==))
    paired = layers_for_parameters((; pair_vars = [1, 2, 3]))
    paired_filtered = layers_for_parameters((;
        op = ==, pair_vars = [1, 2, 3], val = 1,
    ))
    reordered = layers_for_parameters((;
        pair_vars = [1, 2, 3], val = 1, op = ==,
    ))

    @test scalar == CompositionalNetworks._BASE_LAYERS
    @test first(paired) === PairedMap
    matrix_layers = layers_for_parameters((; pair_vars = [1 2 3; 2 3 4]))
    @test first(matrix_layers) === EventMap
    matrix_plan = structure_for_parameters((;
        pair_vars = [1 2 3; 2 3 4], dim = 2, bool = true))
    @test first(matrix_plan.direct_layers) === EventMap
    @test length(matrix_plan.branches) == 2
    @test all(first(branch.layers) === PairedMap for branch in matrix_plan.branches)
    @test getfield.(matrix_plan.branches, :selector) == [1, 2]
    @test matrix_plan.dim == 2
    @test matrix_plan.boolean_guard
    @test paired_filtered[1:1] == [PairedMap]
    @test reordered == paired_filtered

    vals_plan = structure_for_parameters((; vals = [1, 3], op = ==, bool = true))
    @test first(vals_plan.direct_layers) === SimpleFilter
    @test length(vals_plan.branches) == 2
    @test all(branch.layers == CompositionalNetworks._BASE_LAYERS
        for branch in vals_plan.branches)
    @test :exists_min in vals_plan.reductions
    @test :forall_sum in vals_plan.reductions
    @test vals_plan.boolean_guard
    @test bind_branch_parameters((; vals = [1, 3], op = ==), vals_plan.branches[2]) ==
          (; op = ==, val = 3)
    cardinality_values = [2 0 1; 5 1 3; 10 2 3]
    cardinality_plan = structure_for_parameters((;
        vals = cardinality_values, bool = false,
    ))
    @test isnothing(cardinality_plan.direct_layers)
    @test cardinality_plan.boolean_guard
    @test count(branch -> branch.source === :vals_row, cardinality_plan.branches) == 3
    @test count(branch -> branch.source === :vals_domain, cardinality_plan.branches) == 1
    @test cardinality_plan.reductions == [:forall_sum, :guarded_sum]
    @test bind_branch_parameters(
        (; vals = cardinality_values, bool = false), cardinality_plan.branches[1],
    ) == (; bool = false, filter_val = 2, op = in, val = 0:1)
    one_column_values = reshape([2, 5], 2, 1)
    @test bind_branch_parameters(
        (; vals = one_column_values,),
        structure_for_parameters((; vals = one_column_values,)).branches[2],
    ) == (; filter_val = 5, op = (==), val = 1)
    @test bind_branch_parameters(
        (; vals = [2 1; 5 2],),
        structure_for_parameters((; vals = [2 1; 5 2],)).branches[2],
    ) == (; filter_val = 5, op = (==), val = 2)
    domain_parameters = bind_branch_parameters(
        (; vals = cardinality_values, bool = true), cardinality_plan.branches[end],
    )
    @test domain_parameters.bool
    @test domain_parameters.vals == [2, 5, 10]
    @test_throws ArgumentError layers_for_parameters((; vals = cardinality_values))
    @test bind_branch_parameters(
        (; pair_vars = [1 2 3; 4 5 6], dim = 2), matrix_plan.branches[2],
    ) == (; dim = 2, pair_vars = [4, 5, 6])
    table_plan = structure_for_parameters((;
        pair_vars = [[1, 2, 3], [3, 2, 1]],
    ))
    @test isnothing(table_plan.direct_layers)
    @test getfield.(table_plan.branches, :source) ==
          [:pair_vars_row, :pair_vars_row]
    @test all(first(branch.layers) === PairedMap for branch in table_plan.branches)
    @test bind_branch_parameters(
        (; pair_vars = [[1, 2, 3], [3, 2, 1]]), table_plan.branches[2],
    ) == (; pair_vars = [3, 2, 1])
    grouped_tables = structure_for_parameters((;
        pair_vars = ([[1, 2]], [[2, 1]]),
    ))
    @test isnothing(grouped_tables.direct_layers)
    @test isempty(grouped_tables.branches)
    @test length(grouped_tables.groups) == 2
    @test all(length(group.branches) == 1 for group in grouped_tables.groups)
    @test all(group.branches[1].source === :pair_vars_row
        for group in grouped_tables.groups)

    network = ICN(;
        parameters = [:dom_size, :numvars, :pair_vars],
        layers = paired,
        connection = UInt32.(eachindex(paired)),
    )
    @test :pair_vars in network.parameters
    @test network.layers == paired
end

function regularization(icn::AbstractICN)
    max_op = 0
    op = 0
    start = 1
    for (i, layer) in enumerate(icn.layers)
        if !layer.mutex
            op += _selected_weight_count(icn.weights, start:(start + icn.weightlen[i] - 1))
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
