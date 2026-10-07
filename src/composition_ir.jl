"""One selected ICN layer in a human-readable composition."""
struct CompositionLayer
    name::Symbol
    operations::Vector{Symbol}
end

"""
    CompositionIR

Canonical, serializable representation of a selected ICN. `raw_layers` preserves the
network exactly; `layers` removes only operations that are provably neutral. It is meant
for reporting and semantic deduplication, never for executing the hot path.
"""
struct CompositionIR
    raw_layers::Vector{CompositionLayer}
    layers::Vector{CompositionLayer}
    parameters::Vector{Symbol}
end

"""A compiled composition together with its interpretable representation."""
struct Composition{F <: Function}
    f::F
    ir::CompositionIR
    network::Any
    cache::Dict{Tuple{Symbol, String, Bool}, String}
end

"""A non-negative sum of independently learned ICN components."""
struct AdditiveComposition{C <: Tuple}
    components::C
end

AdditiveComposition(components...) = AdditiveComposition(tuple(components...))

@inline function (composition::AdditiveComposition)(arguments...; parameters...)
    return _additive_value(composition.components, arguments...; parameters...)
end

"""A generic quantifier reduction over independently parameterized components."""
struct GroupedComposition{C <: Tuple, O}
    components::C
    reduction::Symbol
    reduction_op::O
    reduction_val::Int
end

function GroupedComposition(components...;
        reduction::Symbol = :exists_product,
        reduction_op::F = (==),
        reduction_val::Integer = 0,
) where {F}
    reduction in (:exists_min, :exists_product, :forall_sum, :forall_max, :count) ||
        throw(ArgumentError("unsupported grouped reduction: $reduction"))
    return GroupedComposition(
        tuple(components...), reduction, reduction_op, Int(reduction_val),
    )
end


@inline function (composition::GroupedComposition)(values; pair_vars, parameters...)
    length(pair_vars) == length(composition.components) || throw(DimensionMismatch(
        "one parameter group is required per grouped composition component",
    ))
    return reduce_icn_outputs(
        index -> composition.components[index](
            values; pair_vars = pair_vars[index], parameters...,
        ),
        eachindex(composition.components),
        composition.reduction;
        op = composition.reduction_op,
        val = composition.reduction_val,
    )
end

@inline _additive_value(::Tuple{}, arguments...; parameters...) = 0.0
@inline function _additive_value(components::Tuple, arguments...; parameters...)
    return first(components)(arguments...; parameters...) +
           _additive_value(Base.tail(components), arguments...; parameters...)
end

@inline function (composition::Composition)(arguments...; parameters...)
    composition.f(arguments...; parameters...)
end

composition(composition::Composition) = composition.f

function composition(icn::AbstractICN; name::Symbol = gensym(:composition))
    return Composition(first(compose(icn; name)), composition_ir(icn), deepcopy(icn),
        Dict{Tuple{Symbol, String, Bool}, String}())
end

"""
Structural adapter for signatures where `id` and/or `val` are decision variables stored in
the input vector. It is compiler-side binding metadata, never a learnable ICN operation.
"""
struct ParameterBindingComposition{C}
    component::C
end

@inline function _binding_index_violation(index::Integer, length::Integer)
    index < 1 && return Float64(1 - index)
    index > length && return Float64(index - length)
    return 0.0
end

@inline _binding_index_violation(index, length::Integer) = 1.0

function (bound::ParameterBindingComposition)(x;
        id = nothing, op = (==), val = nothing, parameters...)
    Base.require_one_based_indexing(x)
    embedded_id = isnothing(id)
    embedded_val = isnothing(val)
    minimum_length = embedded_id + embedded_val + (embedded_id || embedded_val)
    length(x) >= minimum_length || return 1.0

    selected_id = embedded_id ? first(x) : id
    selected_val = embedded_val ? last(x) : val
    values = if embedded_id && embedded_val
        @view x[2:(end - 1)]
    elseif embedded_id
        @view x[2:end]
    elseif embedded_val
        @view x[1:(end - 1)]
    else
        x
    end
    violation = _binding_index_violation(selected_id, length(values))
    iszero(violation) || return violation
    return bound.component(
        values; id = selected_id, op, val = selected_val, parameters...)
end


@testitem "Parameterized compositions bind indexed input variables by signature" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @test

    function select!(network, operations)
        fill!(network.weights.parent, false)
        offset = 0
        for (layer, selected) in zip(network.layers, operations)
            names = collect(keys(layer.fn))
            for operation in selected
                network.weights.parent[offset + only(findall(==(operation), names))] = true
            end
            offset += length(layer.fn)
        end
        return network
    end

    layers = [CN.SimpleFilter, CN.Transformation, CN.Arithmetic,
        CN.Aggregation, CN.Comparison]
    network = select!(CN.ICN(;
        parameters = [:id, :op, :val],
        parameter_values = (; id = 1, op = (==), val = 0),
        layers,
        connection = UInt32.(eachindex(layers)),
    ), ((:filter_elem,), (:id,), (:sum,), (:sum,), (:condition_residual,)))
    bound = CN.parameterized_composition(
        network, (; id = nothing, op = (==), val = nothing);
        name = :indexed_binding_test,
    )

    @test Base.invokelatest(bound, [1, 3, 2]; id = 2, op = (==), val = 3) == 0.0
    @test Base.invokelatest(bound, [2, 7, 3]; id = nothing, op = (==), val = 3) == 0.0
    @test Base.invokelatest(bound, [7, 2, 2]; id = 2, op = (==), val = nothing) == 0.0
    @test Base.invokelatest(bound, [2, 7, 3, 3];
        id = nothing, op = (==), val = nothing) == 0.0
    @test Base.invokelatest(bound, [0, 7, 3];
        id = nothing, op = (==), val = 3) == 1.0
    @test Base.invokelatest(bound, [5, 7, 3];
        id = nothing, op = (==), val = 3) == 3.0
    @test Base.invokelatest(bound, Int[]; id = 2, op = (==), val = 3) == 2.0
end

"""
    parameterized_composition(icn, parameter_values; name)

Compile an ICN and attach structural input bindings inferred only from keyword names and
runtime types. The current binding rule recognizes the complete `id`/`op`/`val` signature:
`id === nothing` binds the first input and `val === nothing` binds the last input. Other
signatures keep the ordinary compiled composition.
"""
function parameterized_composition(
        icn::AbstractICN, parameter_values::NamedTuple;
        name::Symbol = gensym(:parameterized_composition))
    compiled = composition(icn; name)
    if haskey(parameter_values, :id) && haskey(parameter_values, :op) &&
       haskey(parameter_values, :val)
        return ParameterBindingComposition(compiled)
    end
    return compiled
end

function Composition(f::F, selected::Vector{Vector{Symbol}}) where {F <: Function}
    raw = CompositionLayer[CompositionLayer(Symbol(:Layer, index), copy(operations))
                           for
                           (index, operations) in enumerate(selected)]
    ir = CompositionIR(raw, copy(raw), Symbol[])
    return Composition(f, ir, nothing, Dict{Tuple{Symbol, String, Bool}, String}())
end

function compose(icn::AbstractICN, weights::BitVector;
        name::Symbol = gensym(:composition))
    candidate = deepcopy(icn)
    apply!(candidate, weights) || throw(ArgumentError("invalid ICN weights"))
    return composition(candidate; name)
end

function _raw_composition_layers(icn::AbstractICN)
    selected = _selected_operation_names(icn)
    return CompositionLayer[CompositionLayer(layer.name, copy(operations))
                            for
                            (layer, operations) in zip(icn.layers, selected)]
end

function _identity_layer(layer::CompositionLayer)
    layer.name in (
        :SimpleFilter, :PairedMap, :PairMask, :GroupReduction, :Transformation, :Pointwise,
        :Comparison,
    ) &&
        layer.operations == [:id]
end

"""Remove only identities that cannot change the computed function."""
function _simplify_composition_layers(raw_layers::Vector{CompositionLayer})
    layers = CompositionLayer[]
    vector_width = nothing
    for layer in raw_layers
        operations = copy(layer.operations)
        if layer.name in (:Transformation, :SegmentMap)
            # A non-mutex transformation layer is an unordered set for both supported
            # arithmetic reducers. Sorting makes its canonical identity deterministic.
            sort!(operations)
            vector_width = length(operations)
        end
        current = CompositionLayer(layer.name, operations)
        _identity_layer(current) && continue
        if layer.name === :Arithmetic && vector_width == 1
            # Both sum and product are identities over one transformed vector.
            continue
        end
        push!(layers, current)
    end
    return layers
end

function composition_ir(icn::AbstractICN)
    raw_layers = _raw_composition_layers(icn)
    parameters = sort!(collect(icn.parameters); by = string)
    return CompositionIR(raw_layers, _simplify_composition_layers(raw_layers), parameters)
end

composition_ir(composition::Composition) = composition.ir

function symbols(ir::CompositionIR; simplified::Bool = true)
    layers = simplified ? ir.layers : ir.raw_layers
    return [copy(layer.operations) for layer in layers]
end

function symbols(icn::AbstractICN; simplified::Bool = true)
    symbols(composition_ir(icn); simplified)
end
function symbols(composition::Composition; simplified::Bool = true)
    symbols(composition.ir; simplified)
end

function _canonical_layer(layer::CompositionLayer)
    operations = join(string.(layer.operations), ",")
    return "$(layer.name)[$operations]"
end

"""
    canonical_key(icn; simplified=true)

Stable semantic key used to remove structurally neutral duplicates. Runtime parameter
values are deliberately excluded: the key identifies a generic learned function.
"""
function canonical_key(ir::CompositionIR; simplified::Bool = true)
    layers = simplified ? ir.layers : ir.raw_layers
    layer_key = join(_canonical_layer.(layers), "|")
    return "parameters=$(join(string.(ir.parameters), ','));$layer_key"
end

function canonical_key(icn::AbstractICN; simplified::Bool = true)
    canonical_key(composition_ir(icn); simplified)
end
function canonical_key(composition::Composition; simplified::Bool = true)
    canonical_key(composition.ir; simplified)
end
function canonical_key(composition::AdditiveComposition; simplified::Bool = true)
    keys = sort!(String[canonical_key(component; simplified)
                        for component in composition.components])
    return "add(" * join(keys, ";") * ")"
end
function canonical_key(composition::GroupedComposition; simplified::Bool = true)
    keys = String[canonical_key(component; simplified)
                  for component in composition.components]
    return "grouped[reduction=$(composition.reduction);op=$(composition.reduction_op);" *
           "val=$(composition.reduction_val);components={$(join(keys, ';'))}]"
end

function _operation_call(operation::Symbol, argument::String)
    return operation === :id ? argument : "$(operation)($argument)"
end

function _math_expression(ir::CompositionIR; simplified::Bool = true)
    layers = simplified ? ir.layers : ir.raw_layers
    expression = "x"
    width = 1
    for layer in layers
        operations = layer.operations
        isempty(operations) && continue
        if layer.name in (:Transformation, :SegmentMap)
            transformed = [_operation_call(operation, expression)
                           for operation in operations]
            expression = length(transformed) == 1 ? only(transformed) :
                         "[" * join(transformed, ", ") * "]"
            width = length(transformed)
        elseif layer.name === :Arithmetic
            operation = only(operations)
            if width > 1
                reducer = if operation === :sum
                    "elementwise_sum"
                elseif operation === :product
                    "elementwise_product"
                elseif operation === :difference
                    "elementwise_difference"
                else
                    String(operation)
                end
                expression = "$reducer($expression)"
            end
            width = 1
        else
            # Every remaining current layer is mutually exclusive.
            expression = _operation_call(only(operations), expression)
            width = 1
        end
    end
    return expression
end

function code(ir::CompositionIR, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    language === :maths || throw(ArgumentError(
        "CompositionIR supports :maths; use code(icn, :Julia) for executable Julia code",
    ))
    return "$(name)(x) = $(_math_expression(ir; simplified))"
end

function code(icn::AbstractICN, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    if language === :maths
        return code(composition_ir(icn), language; name, simplified)
    elseif language === :Julia
        generated = last(compose(icn; name = Symbol(name), jlfun = false))
        return format_text(string(generated), SciMLStyle(); pipe_to_function_call = false)
    end
    throw(ArgumentError("unsupported composition language: $language"))
end

function code(composition::Composition, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    key = (language, String(name), simplified)
    return get!(composition.cache, key) do
        if language === :maths
            code(composition.ir, language; name, simplified)
        elseif language === :Julia && !isnothing(composition.network)
            code(composition.network, language; name, simplified)
        else
            throw(ArgumentError(
                "executable source generation requires the originating ICN",
            ))
        end
    end
end

function code(composition::AdditiveComposition, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    language === :maths || throw(ArgumentError(
        "AdditiveComposition currently supports interpretable :maths output",
    ))
    expressions = String[]
    for component in composition.components
        rendered = code(component, :maths; name = "component", simplified)
        push!(expressions, split(rendered, " = "; limit = 2)[2])
    end
    return "$(name)(x) = " * (isempty(expressions) ? "0" : join(expressions, " + "))
end
function code(composition::GroupedComposition, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    language === :maths || throw(ArgumentError(
        "GroupedComposition currently supports interpretable :maths output",
    ))
    expressions = String[]
    for component in composition.components
        rendered = code(component, :maths; name = "group", simplified)
        push!(expressions, split(rendered, " = "; limit = 2)[2])
    end
    return "$(name)(x; pair_vars) = reduce_groups([" * join(expressions, ", ") *
           "]; reduction=$(composition.reduction), op=$(composition.reduction_op), " *
           "val=$(composition.reduction_val))"
end

function composition_to_file!(composition::Composition, path, name,
        language::Symbol = :Julia; simplified::Bool = true)
    open(path, "w") do output
        write(output, code(composition, language; name, simplified))
    end
    return nothing
end

function composition_to_file!(icn::AbstractICN, path, name,
        language::Symbol = :Julia; simplified::Bool = true)
    open(path, "w") do output
        write(output, code(icn, language; name, simplified))
    end
    return nothing
end

@testitem "Composition IR restores interpretability and canonical simplification" begin
    using Test

    function select!(network, operations)
        fill!(network.weights.parent, false)
        offset = 0
        for (layer, selected) in zip(network.layers, operations)
            names = collect(keys(layer.fn))
            for operation in selected
                index = something(findfirst(==(operation), names))
                network.weights.parent[offset + index] = true
            end
            offset += length(layer.fn)
        end
        @test check_weights_validity(network, network.weights)
        return network
    end

    network = ICN(parameters = [:dom_size, :numvars, :val])
    select!(network,
        ([:count_equal_left], [:product], [:count_positive], [:id]))
    ir = composition_ir(network)
    @test symbols(network; simplified = false) == [
        [:count_equal_left], [:product], [:count_positive], [:id]
    ]
    @test symbols(network) == [[:count_equal_left], [:count_positive]]
    @test code(network, :maths; name = "all_different") ==
          "all_different(x) = count_positive(count_equal_left(x))"

    pairwise = ICN(parameters = [:dom_size, :numvars, :val])
    select!(pairwise,
        ([:count_equal_left, :count_less_val], [:sum], [:sum],
            [:max_var_minus_numvars]))
    @test code(pairwise, :maths; name = "no_overlap") ==
          "no_overlap(x) = max_var_minus_numvars(sum(elementwise_sum([count_equal_left(x), count_less_val(x)])))"

    equivalent = deepcopy(network)
    select!(equivalent,
        ([:count_equal_left], [:sum], [:count_positive], [:id]))
    @test canonical_key(network) == canonical_key(equivalent)
    @test canonical_key(network; simplified = false) !=
          canonical_key(equivalent; simplified = false)

    wrapped = composition(network; name = :interpretable_penalty)
    @test symbols(wrapped) == symbols(network)
    @test composition(wrapped)([1, 2, 1]; val = 0, dom_size = 3, numvars = 3) == 1
    symbolic = [:room_a, :room_b, :room_a]
    @test evaluate(network, Solution(symbolic);
        val = 0, dom_size = 3, numvars = 3) == 1
    @test composition(wrapped)(symbolic;
        val = 0, dom_size = 3, numvars = 3) == 1
end
