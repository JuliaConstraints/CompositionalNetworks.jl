"""Reference an earlier component's reduced value or unreduced branch outputs.

The result is a vector unless indexed. Forward edges and cycles are rejected when
constructing the network. This is wiring metadata, never a constraint callback.
"""
struct ComponentReference{I <: Tuple}
    component::Int
    branches::Bool
    indices::I
end
ComponentReference(component::Integer, indices...; branches = false) =
    ComponentReference(Int(component), branches, indices)

"""Search space of one component: an ICN and finite choices for its wiring.

Every choice (including repetition and reduction) receives Boolean weights in
`ComposedICN`. Parameters are binding metadata, not closures deciding a constraint.
The supplied child's weights remain trainable, not frozen witness weights.
"""
struct ICNComponent{C <: AbstractICN, I <: Tuple, P <: Tuple, R <: Tuple, Q <: Tuple}
    network::C
    inputs::I
    bindings::P
    repetitions::R
    reductions::Q
end
const _COMPOSED_REDUCTIONS = (:forall_sum, :forall_max, :exists_min, :exists_product)
const _COMPOSED_REPETITIONS = (:once, :vals, :vals_rows, :vals_domain,
    :when_bool_true, :when_bool_false, :pair_vars_groups, :pair_vars_rows,
    :singletons, :prefixes, :strict_prefixes, :adjacent_pairs, :successor_walks, :windows,
    :present_pairs, :indices, :where_index_op, :before_index, :after_index,
    :rows, :columns, :row_pairs, :column_pairs, :adjacent_rows, :adjacent_columns,
    :aligned_singletons, :input_values, :at_coordinate,
    :paired_vals,
    :take_first_id, :take_last_val, :valid_index)
function ICNComponent(network::AbstractICN; inputs = (InputReference(),),
        bindings = ((; (p => ParameterReference(p) for p in sort!(collect(network.parameters); by = string))...),),
        repetitions = (:once,), reductions = (:forall_sum,))
    all(!isempty, (inputs, bindings, repetitions, reductions)) ||
        throw(ArgumentError("component choices must be nonempty"))
    all(p -> p isa NamedTuple, bindings) || throw(ArgumentError("bindings must be named tuples"))
    all(q -> q in _COMPOSED_REDUCTIONS, reductions) || throw(ArgumentError("unsupported reduction"))
    all(repetitions) do modes
        all(mode -> mode in _COMPOSED_REPETITIONS, modes isa Tuple ? modes : (modes,))
    end || throw(ArgumentError("unsupported repetition"))
    return ICNComponent(network, Tuple(inputs), Tuple(bindings), Tuple(repetitions), Tuple(reductions))
end

struct CompositionChoiceLayer
    name::Symbol
    mutex::Bool
end

"""An acyclic network of ICNs with one optimizer-visible Boolean genotype.

The genotype contains child operation weights, input/parameter wiring choices,
repetition choices, branch quantifiers, output activation and the final quantifier.
Existing AbstractICN optimizers see the same mutex/nonempty block contract as before.
No family dispatcher or Boolean concept participates in its evaluation.
"""
struct ComposedICN <: AbstractICN
    components::Vector{ICNComponent}
    weights::BitVector
    layers::Vector{CompositionChoiceLayer}
    weightlen::Vector{Int}
    blocks::Vector{NamedTuple{(:component, :role, :range), Tuple{Int, Symbol, UnitRange{Int}}}}
    constants::Dict{Symbol, Any}
    parameters::Set{Symbol}
    reductions::Tuple
end

function _check_component_reference(ref::ComponentReference, index)
    1 <= ref.component < index || throw(ArgumentError("component references must point to earlier nodes"))
end
_check_component_reference(ref::Union{InputReference, ParameterReference}, index) = nothing
_check_component_reference(ref::ReshapedReference,index) = _check_component_reference(ref.reference,index)
function _check_component_reference(values::Union{Tuple, NamedTuple}, index)
    foreach(v -> _check_component_reference(v, index), values)
end
_check_component_reference(value, index) = nothing

function ComposedICN(components; reductions = _COMPOSED_REDUCTIONS, constants = Dict{Symbol, Any}())
    isempty(components) && throw(ArgumentError("a composed ICN needs at least one component"))
    !isempty(reductions) && all(q -> q in _COMPOSED_REDUCTIONS, reductions) ||
        throw(ArgumentError("unsupported output reductions"))
    parts = ICNComponent[deepcopy(c) for c in components]
    weights, layers, lengths = BitVector(), CompositionChoiceLayer[], Int[]
    blocks = NamedTuple{(:component, :role, :range), Tuple{Int, Symbol, UnitRange{Int}}}[]
    function block!(component, role, bits, mutex)
        range = (length(weights) + 1):(length(weights) + length(bits))
        append!(weights, bits)
        push!(layers, CompositionChoiceLayer(role, mutex))
        push!(lengths, length(bits))
        push!(blocks, (; component, role, range))
    end
    choice(n) = BitVector(i == 1 for i in 1:n)
    for (i, part) in enumerate(parts)
        all(ref -> ref isa Union{InputReference, ParameterReference, ComponentReference,ReshapedReference}, part.inputs) ||
            throw(ArgumentError("component inputs must be references, not callbacks"))
        _check_component_reference(part.inputs, i)
        _check_component_reference(part.bindings, i)
        offset = 1
        for (j, layer) in enumerate(part.network.layers)
            n = part.network.weightlen[j]
            block!(i, :operation, part.network.weights[offset:(offset + n - 1)], layer.mutex)
            offset += n
        end
        block!(i, :input, choice(length(part.inputs)), true)
        block!(i, :binding, choice(length(part.bindings)), true)
        block!(i, :repetition, choice(length(part.repetitions)), true)
        block!(i, :reduction, choice(length(part.reductions)), true)
    end
    block!(0, :outputs, trues(length(parts)), false)
    block!(0, :reduction, choice(length(reductions)), true)
    params = union((p.network.parameters for p in parts)...)
    return ComposedICN(parts, weights, layers, lengths, blocks, Dict{Symbol, Any}(constants), params, Tuple(reductions))
end

"""Build an ICN with explicit model-variable bindings inside its genotype.

`sample_input` determines keyword types/shapes only: its values are never captured.
All numerical binding references are resolved anew against the complete assignment.
No concept identifier, oracle or penalty callback is accepted. The ordinary builder
still chooses the child grammar from resolved keyword names/types/shapes.
"""
function learnable_binding(bindings::NamedTuple, sample_input;
        input=InputReference(), runtime=(;), max_depth=2, nodes=0)
    sample=resolve_parameters(bindings,sample_input,runtime)
    child=nodes>0 ? learnable_graph(sample;nodes) : learnable_composition(sample;max_depth)
    network=ComposedICN([ICNComponent(child;inputs=(input,),bindings=(bindings,))])
    empty!(network.parameters)
    function visit(ref)
        if ref isa ParameterReference
            push!(network.parameters,ref.name)
        elseif ref isa ReshapedReference
            visit(ref.reference)
        elseif ref isa Union{Tuple,NamedTuple}
            foreach(visit,ref)
        end
    end
    visit(input); visit(bindings)
    return network
end

"""Public genotype layout, for solvers, encoders and reproducible tests."""
composition_weight_blocks(network::ComposedICN) = copy(network.blocks)
function check_weights_validity(network::ComposedICN, weights::AbstractVector{Bool})
    length(weights) == length(network.weights) || throw(DimensionMismatch("composition weights"))
    for (block, layer) in zip(network.blocks, network.layers)
        active = count(identity, view(weights, block.range))
        (layer.mutex ? active == 1 : active >= 1) || return false
    end
    for (i, part) in enumerate(network.components)
        blocks = filter(b -> b.component == i && b.role == :operation, network.blocks)
        range = first(first(blocks).range):last(last(blocks).range)
        check_weights_validity(part.network, view(weights, range)) || return false
    end
    return true
end

function _component_weights!(network::ComposedICN)
    for (i, part) in enumerate(network.components)
        blocks = filter(b -> b.component == i && b.role == :operation, network.blocks)
        apply!(part.network, view(network.weights, first(first(blocks).range):last(last(blocks).range)))
    end
    return network
end
function _composition_choice(network, i, role)
    block = only(b for b in network.blocks if b.component == i && b.role == role)
    return something(findfirst(identity, view(network.weights, block.range)))
end
function _graph_binding(ref::ComponentReference, x, params, outputs, branches)
    values = ref.branches ? branches[ref.component] : [outputs[ref.component]]
    return _route_selection(values, ref.indices)
end
_graph_binding(ref, x, params, outputs, branches) = _resolve_parameter(ref, x, params)
_graph_binding(ref::ReshapedReference,x,params,outputs,branches) =
    _reshape_binding(_graph_binding(ref.reference,x,params,outputs,branches),ref.shape)
_graph_binding(ref::Union{Tuple, NamedTuple}, x, params, outputs, branches) =
    map(value -> _graph_binding(value, x, params, outputs, branches), ref)

function _repeat_contexts(mode::Symbol, x, p)
    mode === :once && return ((x, p),)
    if mode === :take_first_id
        isempty(x) && return ()
        return ((view(x,2:length(x)),(;p...,id=first(x))),)
    elseif mode === :take_last_val
        isempty(x) && return ()
        return ((view(x,1:length(x)-1),(;p...,val=last(x))),)
    elseif mode === :valid_index
        id=get(p,:id,nothing)
        return id isa Integer && 1<=id<=length(x) ? ((x,p),) : ()
    end
    mode === :paired_vals && return ((x,(;p...,pair_vars=value)) for value in p.vals)
    mode === :input_values && return ((x,(;p...,filter_val=value)) for value in unique(x))
    if mode === :at_coordinate
        rows,columns = p.dim
        rows*columns == length(x) || throw(DimensionMismatch("rectangular coordinate layout"))
        bases=get(p,:index_base,(1,1))
        row,column = p.id .- bases .+ 1
        1<=row<=rows && 1<=column<=columns || return ()
        retained=Base.structdiff(p,NamedTuple{(:dim,)})
        return ((x,(;retained...,id=row+(column-1)*rows)),)
    end
    if mode === :aligned_singletons
        axes(x) == axes(p.pair_vars) || throw(DimensionMismatch("aligned singleton scopes"))
        return ((view(x,i:i),(;p...,pair_vars=view(p.pair_vars,i:i))) for i in eachindex(x))
    end
    mode === :indices && return ((x, (; p..., id=i)) for i in eachindex(x))
    mode === :where_index_op && return p.op(p.id + get(p, :index_base, 1) - 1, p.val) ? ((x,p),) : ()
    mode === :before_index && return ((view(x, 1:(p.id-1)), p),)
    mode === :after_index && return ((view(x, (p.id+1):length(x)), p),)
    if mode in (:rows, :columns, :row_pairs, :column_pairs, :adjacent_rows, :adjacent_columns)
        rows, columns = p.dim
        rows > 0 && columns > 0 && rows*columns == length(x) ||
            throw(DimensionMismatch("positive rectangular layout must match the flattened input"))
        matrix = reshape(x, rows, columns)
        slices = mode in (:rows,:row_pairs,:adjacent_rows) ? eachrow(matrix) : eachcol(matrix)
        retained = Base.structdiff(p, NamedTuple{(:dim,)})
        if mode in (:rows,:columns)
            return ((values,retained) for values in slices)
        end
        adjacent = mode in (:adjacent_rows,:adjacent_columns)
        return ((slices[i],(; retained...,pair_vars=slices[j]))
            for i in 1:length(slices)-1 for j in (adjacent ? (i+1:i+1) : (i+1:length(slices))))
    end
    if mode === :present_pairs
        axes(x) == axes(p.pair_vars) || throw(DimensionMismatch("aligned missing-data mask"))
        present(v) = !ismissing(v) && !isnothing(v)
        selected = findall(present, p.pair_vars)
        # In particular, an all-Missing row must not leave an empty Missing vector:
        # its reduction identity would itself be missing instead of numerical zero.
        paired = isempty(selected) ? similar(x,0) : filter(present,p.pair_vars)
        return ((view(x, selected), (; p..., pair_vars = paired)),)
    end
    if mode === :successor_walks
        # Pure repeated indirect indexing, no cycle acceptance or cost function.
        # Invalid indices map to a zero sentinel; a separate genotype must check
        # index validity if its target concept requires it.
        Base.require_one_based_indexing(x)
        return (begin
            walk = zeros(Int, length(x))
            current = root
            for step in eachindex(walk)
                if current isa Integer && 1 <= current <= length(x)
                    successor = x[current] - get(p,:index_base,1) + 1
                    current = successor isa Integer && 1 <= successor <= length(x) ? Int(successor) : 0
                else
                    current = 0
                end
                walk[step] = current
            end
            (walk, p)
        end for root in eachindex(x))
    elseif mode === :windows
        if first(p.window) isa Tuple
            strides,widths,lengths,circular = p.window
            length(strides)==length(widths)==length(lengths)>0 || throw(DimensionMismatch("window layout"))
            all(>(0),strides) && all(>(0),widths) && all(>(0),lengths) ||
                throw(ArgumentError("positive multi-window layout required"))
            sum(lengths)==length(x) || throw(DimensionMismatch("flattened window input"))
            offsets=cumsum([0;collect(lengths)])
            retained=Base.structdiff(p,NamedTuple{(:window,)})
            steps=circular ? cld(first(lengths),first(strides)) :
                max(0,fld(first(lengths)-first(widths),first(strides))+1)
            # The first list controls termination; other lists wrap independently,
            # as in XCSP3-Java-Tools XSlide.buildScopes (also in noncircular mode).
            return ((view(x,[offsets[j]+mod1(step*strides[j]+k,lengths[j])
                for j in eachindex(lengths) for k in 1:widths[j]]),retained)
                for step in 0:steps-1)
        end
        stride, width = p.window[1:2]
        circular = length(p.window) == 3 && p.window[3]
        stride > 0 && width > 0 || throw(ArgumentError("positive window stride and width required"))
        retained = Base.structdiff(p, NamedTuple{(:window,)})
        if circular
            return ((view(x, [mod1(start+j, length(x)) for j in 0:(width-1)]), retained)
                for start in 1:stride:length(x))
        end
        return ((view(x, start:(start+width-1)), retained)
            for start in 1:stride:(length(x)-width+1))
    end
    mode === :when_bool_true && return p.bool ? ((x, p),) : ()
    mode === :when_bool_false && return p.bool ? () : ((x, p),)
    if mode === :vals_rows
        rows = p.vals
        rows isa AbstractMatrix || throw(ArgumentError("vals_rows needs a matrix"))
        retained = Base.structdiff(p, NamedTuple{(:vals, :filter_val, :op, :val)})
        return (begin
            filter_val, op, val = _vals_row_condition(rows, i)
            (x, (; retained..., filter_val, op, val))
        end for i in axes(rows, 1))
    elseif mode === :vals_domain
        return ((x, (; p..., vals = view(p.vals, :, 1))),)
    elseif mode === :pair_vars_groups
        return ((x, (; p..., pair_vars = group)) for group in p.pair_vars)
    end
    if mode === :vals
        p.vals isa AbstractVector || throw(ArgumentError("vals repetition needs a vector"))
        retained = Base.structdiff(p, NamedTuple{(:vals,)})
        return ((x, (; retained..., val)) for val in p.vals)
    elseif mode === :pair_vars_rows
        rows = p.pair_vars
        n = rows isa AbstractMatrix ? size(rows, 1) : length(rows)
        return (begin
            row = rows isa AbstractMatrix ? view(rows, i, :) : rows[i]
            aligned_targets = haskey(p,:val) && (p.val isa AbstractVector ||
                (p.val isa Tuple && get(p,:op,nothing) isa Tuple))
            val = aligned_targets ? p.val[i] : get(p, :val, nothing)
            op = haskey(p, :op) && p.op isa Tuple ? p.op[i] : get(p, :op, nothing)
            retained = Base.structdiff(p, NamedTuple{(:pair_vars, :op, :val)})
            mapped = (; retained..., pair_vars = row)
            mapped = haskey(p, :op) ? (; mapped..., op) : mapped
            mapped = haskey(p, :val) ? (; mapped..., val) : mapped
            (x, mapped)
        end for i in 1:n)
    end
    Base.require_one_based_indexing(x)
    n = mode === :adjacent_pairs ? max(0, length(x) - 1) : length(x)
    return (begin
        scope = mode === :singletons ? (i:i) : mode === :prefixes ? (1:i) :
                mode === :strict_prefixes ? (1:(i - 1)) : (i:(i + 1))
        (view(x, scope), p)
    end for i in 1:n)
end

function _component_outputs!(result, child, x, p, modes::Tuple)
    if isempty(modes)
        push!(result, evaluate(child, Solution(x); (; child.constants..., p...)...))
        return result
    end
    for (values, parameters) in _repeat_contexts(first(modes), x, p)
        _component_outputs!(result, child, values, parameters, Base.tail(modes))
    end
    return result
end
_composition_reduce(values, q) = isempty(values) && q === :exists_min ? 1.0 : reduce_icn_outputs(values, q)
function _active_components(network)
    block = only(b for b in network.blocks if b.component == 0 && b.role == :outputs)
    active = BitVector(view(network.weights, block.range))
    function visit(ref)
        if ref isa ComponentReference
            active[ref.component] = true
        elseif ref isa ReshapedReference
            visit(ref.reference)
        elseif ref isa Union{Tuple, NamedTuple}
            foreach(visit, ref)
        end
    end
    for i in reverse(eachindex(network.components))
        active[i] || continue
        part = network.components[i]
        visit(part.inputs[_composition_choice(network, i, :input)])
        visit(part.bindings[_composition_choice(network, i, :binding)])
    end
    return active
end
function evaluate(network::ComposedICN, config::Configuration;
        weights_validity = check_weights_validity(network, network.weights), parameters...)
    weights_validity || return Inf
    _component_weights!(network)
    outputs = Float64[]
    branches = Vector{Float64}[]
    runtime = (; network.constants..., parameters...)
    active = _active_components(network)
    output_block = only(b for b in network.blocks if b.component == 0 && b.role == :outputs)
    terminal = view(network.weights,output_block.range)
    for (i, part) in enumerate(network.components)
        if !active[i]
            push!(branches, Float64[])
            push!(outputs, 0.0)
            continue
        end
        input = part.inputs[_composition_choice(network, i, :input)]
        bindings = part.bindings[_composition_choice(network, i, :binding)]
        mode = part.repetitions[_composition_choice(network, i, :repetition)]
        q = part.reductions[_composition_choice(network, i, :reduction)]
        x = _graph_binding(input, config.x, runtime, outputs, branches)
        p = _graph_binding(bindings, config.x, runtime, outputs, branches)
        values = _component_outputs!(Float64[], part.network, x, p, mode isa Tuple ? mode : (mode,))
        # Internal arithmetic features may be signed. Only selected cost outputs
        # must be nonnegative; this prevents cancellation between terminal costs.
        all(v -> isfinite(v) && (!terminal[i] || v >= 0), values) || return Inf
        push!(branches, values)
        push!(outputs, _composition_reduce(values, q))
    end
    block = only(b for b in network.blocks if b.component == 0 && b.role == :outputs)
    selected = outputs[view(network.weights, block.range)]
    return _composition_reduce(selected, network.reductions[_composition_choice(network, 0, :reduction)])
end

"""Construct parameter-induced repeated ICNs without any constraint identifier.

The default factory retains the normal layer/operation choices. A custom factory
can bound those choices for small tests, but must return an ordinary trainable ICN.
"""
function _signature_network(p)
    layers = layers_for_parameters(p)
    return ICN(; parameters = collect(keys(p)), parameter_values = p,
        layers, connection = UInt32.(eachindex(layers)))
end
function learnable_composition(parameters::NamedTuple; factory = _signature_network,
        include_direct = true, max_depth::Integer = 2)
    max_depth >= 1 || throw(ArgumentError("positive composition depth required"))
    bindings = ((; (name => ParameterReference(name) for name in keys(parameters))...),)
    function parameterized_network(parts; reductions = _COMPOSED_REDUCTIONS)
        network = ComposedICN(parts; reductions)
        # Public parameters are the root signature, not internally bound child roles.
        empty!(network.parameters)
        union!(network.parameters, keys(parameters))
        return network
    end
    child_for(sample) = learnable_composition(sample; factory, include_direct = false, max_depth)
    function assembled(component)
        parts = ICNComponent[]
        plan = structure_for_parameters(parameters)
        aligned_conditions = (haskey(parameters, :op) && parameters.op isa Tuple) ||
            (haskey(parameters, :pair_vars) && parameters.pair_vars isa AbstractMatrix &&
             haskey(parameters, :val) && parameters.val isa AbstractVector &&
             get(parameters, :op, nothing) !== in)
        if include_direct && !isnothing(plan.direct_layers) && !aligned_conditions
            push!(parts, ICNComponent(factory(parameters); bindings))
        end
        push!(parts, component)
        if max_depth > 1 && haskey(parameters, :vals) && parameters.vals isa AbstractVector
            # Independent quantifiers at each level: Q_scope Q_vals child(x,val).
            # Available because vals is a collection, not because of a concept name.
            inner = learnable_composition(parameters; factory, include_direct = false,
                max_depth = max_depth - 1)
            push!(parts, ICNComponent(inner; bindings,
                repetitions = (:singletons, :prefixes, :strict_prefixes, :adjacent_pairs),
                reductions = _COMPOSED_REDUCTIONS))
        end
        if max_depth > 1
            # Collection-induced outputs can themselves feed an ordinary ICN.
            # Both the source edge and every postprocessor operation are weights.
            # The postprocessor consumes outputs, not the original aligned data.
            scalar_parameters = Base.structdiff(parameters,
                NamedTuple{(:pair_vars, :vals, :id, :dim, :filter_val, :language)})
            postprocessor = factory(scalar_parameters)
            inputs = (InputReference(),
                (ComponentReference(i; branches = true) for i in eachindex(parts))...)
            push!(parts, ICNComponent(postprocessor; inputs, bindings))
            if haskey(parameters, :bool) && parameters.bool isa Bool
                # Independently weighted optional postcondition over the same outputs.
                push!(parts, ICNComponent(postprocessor; inputs, bindings,
                    repetitions = (:when_bool_true, :when_bool_false)))
            end
        end
        return parameterized_network(parts; reductions = (:forall_sum,))
    end
    id_input = haskey(parameters,:id) && isnothing(parameters.id)
    val_input = haskey(parameters,:val) && isnothing(parameters.val)
    if id_input || val_input
        # Nothing denotes an unbound role. Extraction is explicit in the
        # genotype; it is not selected by a constraint identifier.
        sample = id_input ? (;parameters...,id=1) : parameters
        sample = val_input ? (;sample...,val=0) : sample
        child = learnable_composition(sample;factory,include_direct,max_depth)
        modes = id_input && val_input ?
            (:once,:take_first_id,:take_last_val,(:take_first_id,:take_last_val),
             (:take_first_id,:take_last_val,:valid_index)) :
            id_input ? (:once,:take_first_id,(:take_first_id,:valid_index)) :
            (:once,:take_last_val,(:take_last_val,:valid_index))
        return parameterized_network([ICNComponent(child;bindings,repetitions=modes,
            reductions=_COMPOSED_REDUCTIONS)])
    elseif haskey(parameters,:pair_vars) && parameters.pair_vars isa NamedTuple
        parts = ICNComponent[]
        for (name,value) in pairs(parameters.pair_vars)
            # A heterogeneous field keeps every operation induced by its type,
            # including the direct event/segment path of matrix parameters.
            child = learnable_composition((;parameters...,pair_vars=value);
                factory, include_direct = true, max_depth)
            mapping = (;first(bindings)...,pair_vars=ParameterReference(:pair_vars,name))
            inputs = (InputReference(),
                (ComponentReference(i;branches=true) for i in eachindex(parts))...)
            repetitions = value isa AbstractVector ? (:once,:aligned_singletons) : (:once,)
            push!(parts,ICNComponent(child;inputs,bindings=(mapping,),repetitions,
                reductions=_COMPOSED_REDUCTIONS))
        end
        return parameterized_network(parts)
    elseif haskey(parameters, :dim) && parameters.dim isa Tuple{Integer,Integer}
        # Rectangular scope metadata introduces unary and paired slice choices.
        # All scopes and their leaf networks remain selectable by weights.
        tuple_values = get(parameters, :vals, nothing) isa AbstractVector &&
            eltype(parameters.vals) <: Union{AbstractVector,Tuple}
        sample = tuple_values ? Base.structdiff(parameters, NamedTuple{(:dim,:vals)}) :
            Base.structdiff(parameters, NamedTuple{(:dim,)})
        scalar = learnable_composition(sample; factory, include_direct=true, max_depth)
        paired_sample = tuple_values ? Base.structdiff(parameters,NamedTuple{(:dim,)}) : sample
        paired = child_for((;paired_sample...,pair_vars=zeros(Int,last(parameters.dim))))
        unary_modes = (:rows,:columns)
        paired_modes = (:row_pairs,:column_pairs,:adjacent_rows,:adjacent_columns)
        parts = ICNComponent[
            ICNComponent(scalar;bindings,repetitions=unary_modes,reductions=_COMPOSED_REDUCTIONS),
            ICNComponent(scalar;bindings,repetitions=unary_modes,reductions=_COMPOSED_REDUCTIONS),
            ICNComponent(paired;bindings,repetitions=paired_modes,reductions=_COMPOSED_REDUCTIONS),
            ICNComponent(paired;bindings,repetitions=paired_modes,reductions=_COMPOSED_REDUCTIONS)]
        if get(parameters,:id,nothing) isa Tuple{Integer,Integer}
            indexed=child_for((;sample...,id=1))
            push!(parts,ICNComponent(indexed;bindings,repetitions=(:at_coordinate,),reductions=_COMPOSED_REDUCTIONS))
        end
        return parameterized_network(parts)
    elseif haskey(parameters, :window) && parameters.window isa Union{Tuple{Integer,Integer},Tuple{Integer,Integer,Bool},Tuple{Tuple,Tuple,Tuple,Bool}}
        child = child_for(Base.structdiff(parameters, NamedTuple{(:window,)}))
        parts=ICNComponent[ICNComponent(child; bindings,
            repetitions = (:once, :windows), reductions = _COMPOSED_REDUCTIONS)]
        if max_depth > 1
            scalar=Base.structdiff(parameters,NamedTuple{(:window,:pair_vars,:vals,:dim,:language)})
            push!(parts,ICNComponent(factory(scalar);bindings,
                inputs=(InputReference(),ComponentReference(1;branches=true)),reductions=_COMPOSED_REDUCTIONS))
        end
        return parameterized_network(parts)
    elseif haskey(parameters, :vals) && parameters.vals isa AbstractMatrix
        size(parameters.vals, 1) > 0 || throw(ArgumentError("construct from a nonempty row signature"))
        sample = first(_repeat_contexts(:vals_rows, Int[], parameters))[2]
        rows = ICNComponent(child_for(sample); bindings, repetitions = (:vals_rows,), reductions = _COMPOSED_REDUCTIONS)
        parts = ICNComponent[rows]
        if haskey(parameters, :bool) && parameters.bool isa Bool
            domain = (; parameters..., vals = view(parameters.vals, :, 1))
            push!(parts, ICNComponent(factory(domain); bindings,
                repetitions = ((:when_bool_true, :vals_domain), (:when_bool_false, :vals_domain)),
                reductions = _COMPOSED_REDUCTIONS))
        end
        if max_depth > 2
            # Larger capacity also exposes independent column projections. The
            # selected column and all operations on it are ordinary genotype bits.
            inputs = Tuple(ParameterReference(:vals, :, j) for j in axes(parameters.vals,2))
            push!(parts, ICNComponent(factory((;));inputs,bindings=((;),)))
        end
        return parameterized_network(parts)
    elseif haskey(parameters, :pair_vars) && _grouped_parameter_rows(parameters.pair_vars)
        child = learnable_composition((; parameters..., pair_vars = first(parameters.pair_vars));
            factory, include_direct, max_depth)
        return parameterized_network([ICNComponent(child; bindings, repetitions = (:pair_vars_groups,),
            reductions = _COMPOSED_REDUCTIONS)])
    elseif haskey(parameters, :pair_vars) &&
           (parameters.pair_vars isa AbstractMatrix || _nested_parameter_rows(parameters.pair_vars))
        size(parameters.pair_vars, 1) > 0 || throw(ArgumentError("construct a row network with a nonempty signature example"))
        sample = first(_repeat_contexts(:pair_vars_rows, Int[], parameters))[2]
        repetitions = (Missing <: eltype(sample.pair_vars) || Nothing <: eltype(sample.pair_vars)) ?
            (:pair_vars_rows, (:pair_vars_rows, :present_pairs)) : (:pair_vars_rows,)
        component = ICNComponent(child_for(sample); repetitions,
            bindings,
            reductions = _COMPOSED_REDUCTIONS)
        return assembled(component)
    elseif get(parameters,:vals,nothing) isa AbstractVector &&
            eltype(parameters.vals) <: Union{AbstractVector,Tuple} &&
            get(parameters,:pair_vars,nothing) isa AbstractVector
        # Nested values induce independent comparisons against each value tuple.
        # Neither tuple membership nor its relation to the paired input is built
        # in: the leaf operations, sources, quantifiers and root reducer are bits.
        sample=Base.structdiff(parameters,NamedTuple{(:vals,)})
        child=factory(sample)
        inputs=(InputReference(),ParameterReference(:pair_vars))
        return parameterized_network([
            ICNComponent(child;bindings),
            ICNComponent(child;inputs,bindings,repetitions=(:paired_vals,),reductions=_COMPOSED_REDUCTIONS),
            ICNComponent(child;inputs,bindings,repetitions=(:paired_vals,),reductions=_COMPOSED_REDUCTIONS)])
    elseif haskey(parameters, :vals) && parameters.vals isa AbstractVector
        isempty(parameters.vals) && throw(ArgumentError("construct a vals network with a nonempty signature example"))
        sample = first(_repeat_contexts(:vals, Int[], parameters))[2]
        component = ICNComponent(child_for(sample); repetitions = (:vals,),
            bindings,
            reductions = _COMPOSED_REDUCTIONS)
        return assembled(component)
    elseif haskey(parameters, :filter_val) && haskey(parameters, :pair_vars) &&
           parameters.pair_vars isa AbstractVector
        # A scalar comparison map followed by an aligned map is available from
        # the filter role and the paired-vector role, independently of a concept.
        mapped = factory((; op = (!=), val = parameters.filter_val))
        map_bindings = Tuple((; op, val = ParameterReference(:filter_val))
            for op in ((==), (!=), (<), (<=), (>), (>=)))
        projected = factory(Base.structdiff(parameters, NamedTuple{(:filter_val,)}))
        parts = ICNComponent[]
        include_direct && push!(parts, ICNComponent(factory(parameters); bindings))
        map_index = length(parts) + 1
        push!(parts, ICNComponent(mapped; bindings = map_bindings,
                    repetitions = (:once, :singletons)))
        push!(parts, ICNComponent(projected; bindings,
                    inputs = (InputReference(), ComponentReference(map_index; branches = true))))
        return parameterized_network(parts)
    end
    if max_depth > 2 && get(parameters,:pair_vars,nothing) isa AbstractVector &&
            haskey(parameters,:op) && haskey(parameters,:val)
        sample=(;parameters...,filter_val=zero(eltype(parameters.pair_vars)))
        return parameterized_network([
            ICNComponent(factory(parameters);bindings),
            ICNComponent(child_for(sample);bindings,repetitions=(:input_values,),reductions=_COMPOSED_REDUCTIONS)])
    end
    return factory(parameters)
end

"""A bounded generic DAG over parameter-compatible ICNs.

No concept identifier is accepted. Each node's operations, backward input edge,
scalar binding, repetition and reduction are genotype choices. The default scalar
ICN factory remains unchanged; this explicitly requested larger search space is
useful when a witness needs several interacting ICNs. Index walks are meaningful
on integer-valued scopes; zero is their invalid-index sentinel, not a penalty.
"""
function learnable_graph(parameters::NamedTuple; nodes::Integer = 5,
        literal_values = (0, 1, 2))
    nodes > 0 || throw(ArgumentError("positive number of graph nodes required"))
    parts = ICNComponent[]
    for i in 1:nodes
        child = _signature_network(parameters)
        inputs = (InputReference(), (ComponentReference(j) for j in 1:(i-1))...,
            (ComponentReference(j; branches = true) for j in 1:(i-1))...,
            (ParameterReference(name) for (name,value) in pairs(parameters) if value isa AbstractVector)...)
        base = (; (name => ParameterReference(name) for name in keys(parameters))...)
        targets = haskey(parameters, :val) ?
            (ParameterReference(:val), literal_values...,
             (ComponentReference(j, 1) for j in 1:(i-1))...) : (nothing,)
        operators = haskey(parameters, :op) ?
            (ParameterReference(:op), (==), (!=), (<), (<=), (>), (>=)) : (nothing,)
        bindings = Tuple(begin
            p = haskey(parameters, :val) ? (; base..., val = target) : base
            haskey(parameters, :op) ? (; p..., op = operator) : p
        end for target in targets for operator in operators)
        repetitions = (:once, :singletons, :prefixes, :strict_prefixes,
            :adjacent_pairs, :successor_walks)
        if haskey(parameters, :id) && parameters.id isa Integer
            repetitions = (repetitions..., :indices, :before_index, :after_index)
            if haskey(parameters, :op) && haskey(parameters, :val)
                repetitions = (repetitions..., (:indices, :where_index_op))
            end
        end
        push!(parts, ICNComponent(child; inputs, bindings,
            repetitions, reductions = _COMPOSED_REDUCTIONS))
    end
    return ComposedICN(parts)
end

struct DecodedComposedICN
    network::ComposedICN
end
composition(network::ComposedICN; name::Symbol = gensym(:composed_icn)) =
    check_weights_validity(network, network.weights) ? DecodedComposedICN(deepcopy(network)) :
    throw(ArgumentError("cannot decode invalid composition weights"))
(decoded::DecodedComposedICN)(x; parameters...) = evaluate(decoded.network, Solution(x); parameters...)
compose(network::ComposedICN; name::Symbol = gensym(:composed_icn)) =
    (composition(network; name), code(network, :maths; name = String(name)))
incremental_supported(::ComposedICN) = false
incremental_supported(::DecodedComposedICN) = false

function code(network::ComposedICN, language::Symbol = :maths; name = "composition", simplified = true)
    language === :maths || throw(ArgumentError("composed ICN source export currently supports :maths"))
    _component_weights!(network)
    terms = String[]
    for (i, part) in enumerate(network.components)
        selected(role, options) = options[_composition_choice(network, i, role)]
        body = code(part.network, :maths; name = "node$i", simplified)
        push!(terms, "$body; input=$(repr(selected(:input, part.inputs))); " *
            "bindings=$(repr(selected(:binding, part.bindings))); " *
            "repeat=$(repr(selected(:repetition, part.repetitions))); " *
            "reduce=$(selected(:reduction, part.reductions))")
    end
    block = only(b for b in network.blocks if b.component == 0 && b.role == :outputs)
    return "$name(x) = composed{" * join(terms, "; ") *
        "; outputs=$(findall(view(network.weights, block.range))); " *
        "reduce=$(network.reductions[_composition_choice(network, 0, :reduction)])}"
end
code(decoded::DecodedComposedICN, args...; kwargs...) = code(decoded.network, args...; kwargs...)
canonical_key(network::ComposedICN; simplified = true) = code(network, :maths; name = "composed", simplified)
canonical_key(decoded::DecodedComposedICN; kwargs...) = canonical_key(decoded.network; kwargs...)
function symbols(network::ComposedICN; simplified = true)
    _component_weights!(network)
    return [(; component = i, operations = symbols(part.network; simplified),
        input = part.inputs[_composition_choice(network, i, :input)],
        binding = part.bindings[_composition_choice(network, i, :binding)],
        repetition = part.repetitions[_composition_choice(network, i, :repetition)],
        reduction = part.reductions[_composition_choice(network, i, :reduction)])
        for (i, part) in enumerate(network.components)]
end
symbols(decoded::DecodedComposedICN; kwargs...) = symbols(decoded.network; kwargs...)
