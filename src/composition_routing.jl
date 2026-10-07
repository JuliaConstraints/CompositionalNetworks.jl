"""
    InputReference(indices...)

Reference the current assignment, or select entries from it. An integer index returns a
scalar; range/vector indices return a view. No indices means the whole assignment.
These are structural bindings, not learnable ICN operations.
"""
struct InputReference{I <: Tuple}
    indices::I
end
InputReference(indices...) = InputReference(indices)

"""
    ParameterReference(name, indices...)

Reference a named runtime keyword, optionally selecting entries (for example a row of
coefficients or one member of an `(operator, target)` tuple). Values are never captured
from a training instance. Missing keywords and invalid indices raise normal errors.
"""
struct ParameterReference{I <: Tuple}
    name::Symbol
    indices::I
end
ParameterReference(name::Symbol, indices...) = ParameterReference(name, indices)

"""Reshape a gathered input/keyword reference without embedding numerical data.

Repeated indices keep variable aliases. A tuple of literal/reference cells can
mix constants and variables. Shape is modeling metadata, not a cost.
"""
struct ReshapedReference{R,D <: Tuple}
    reference::R
    shape::D
end
_reshape_binding(values::AbstractArray,shape) = reshape(values,shape)
_reshape_binding(values::Tuple,shape) = reshape(collect(values),shape)
@inline _resolve_parameter(ref::ReshapedReference, x, parameters) =
    _reshape_binding(_resolve_parameter(ref.reference,x,parameters),ref.shape)

@inline _route_selection(value, ::Tuple{}) = value
@inline _route_selection(value, indices::Tuple) = getindex(value, indices...)
@inline function _route_selection(value::AbstractArray, indices::Tuple)
    return all(index -> index isa Integer, indices) ?
           getindex(value, indices...) : view(value, indices...)
end
# Resolve the empty-index intersection explicitly: it denotes the array, not a 0-D view.
@inline _route_selection(value::AbstractArray, ::Tuple{}) = value

@inline _resolve_parameter(value, x, parameters) = value
@inline _resolve_parameter(values::Union{Tuple, NamedTuple}, x, parameters) =
    map(value -> _resolve_parameter(value, x, parameters), values)
@inline _resolve_parameter(ref::InputReference, x, parameters) =
    _route_selection(x, ref.indices)
@inline _resolve_parameter(ref::ParameterReference, x, parameters) =
    _route_selection(getproperty(parameters, ref.name), ref.indices)

"""
    resolve_parameters(bindings::NamedTuple, x, parameters::NamedTuple)

Resolve literal values and input/keyword references, including references nested in
tuples/named tuples (such as `(operator, InputReference(4))`). Arrays supplied as literal
data are not traversed or copied. The result can be passed to
`structure_for_parameters` / `layers_for_parameters` when constructing a network, and
is resolved afresh during evaluation. Only the explicitly mapped keywords are forwarded.
"""
@inline resolve_parameters(bindings::NamedTuple, x, parameters::NamedTuple) =
    map(binding -> _resolve_parameter(binding, x, parameters), bindings)

"""
    RoutedComposition(component; input=InputReference(), parameters::NamedTuple)

Apply an existing composition to a selected scope with independently bound keywords.
Compose these adapters with `AdditiveComposition` to share a scope while supplying
different coefficients, operators, or targets. Targets may reference decision variables.
Routing changes neither the component's grammar nor its selected weights. Scope/shape
errors are errors, not invented constraint penalties. Incremental evaluation is not yet
supported because input/parameter dependencies need their own invalidation tracking.
"""
struct RoutedComposition{C, I, P <: NamedTuple}
    component::C
    input::I
    parameters::P
end
RoutedComposition(component; input = InputReference(), parameters::NamedTuple) =
    RoutedComposition(component, input, parameters)

@inline function (routed::RoutedComposition)(x; parameters...)
    runtime = (; parameters...)
    values = _resolve_parameter(routed.input, x, runtime)
    bound = resolve_parameters(routed.parameters, x, runtime)
    return routed.component(values; bound...)
end

_routing_label(ref::InputReference) = "input" * repr(ref.indices)
_routing_label(ref::ParameterReference) = "keyword:" * String(ref.name) * repr(ref.indices)
_routing_label(ref::ReshapedReference) = "reshape(" * _routing_label(ref.reference) * "," * repr(ref.shape) * ")"
_routing_label(value) = "literal:" * string(typeof(value)) * ":" * repr(value)
function _routing_parameters(parameters::NamedTuple)
    return join((string(key) * "=" * _routing_label(getproperty(parameters, key))
                 for key in sort!(collect(keys(parameters)); by = string)), ", ")
end

function canonical_key(routed::RoutedComposition; simplified::Bool = true)
    return "routed[input=$(_routing_label(routed.input));" *
           "parameters={$(_routing_parameters(routed.parameters))};" *
           "component={$(canonical_key(routed.component; simplified))}]"
end

symbols(routed::RoutedComposition; simplified::Bool = true) =
    symbols(routed.component; simplified)

function code(routed::RoutedComposition, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    language === :maths || throw(ArgumentError(
        "RoutedComposition currently supports interpretable :maths output"))
    body = split(code(routed.component, :maths; name = "component", simplified),
        " = "; limit = 2)[2]
    return "$(name)(x; parameters...) = bind(" * body *
           "; x <- $(_routing_label(routed.input)); " *
           _routing_parameters(routed.parameters) * ")"
end

incremental_supported(::RoutedComposition) = false

@testitem "Composition routing resolves scope and keyword references without copying" begin
    using Test
    x = [1, 2, 3, 4]
    matrix = [1 2; 3 4]
    bindings = (; val = InputReference(4), pair_vars = ParameterReference(:matrix, 1, :),
        op = (<=))
    resolved = resolve_parameters(bindings, x, (; matrix))
    @test resolved.val == 4
    @test resolved.op === (<=)
    @test resolved.pair_vars == [1, 2]
    @test parent(resolved.pair_vars) === matrix
    matrix[1, 1] = 9
    @test resolved.pair_vars[1] == 9
    @test resolve_parameters((; vals = InputReference()), x, (;)).vals === x
    @test resolve_parameters((; vals = ParameterReference(:matrix)), x, (; matrix)).vals === matrix
    @test resolve_parameters((; val = ParameterReference(:matrix, 2, 1)), x, (; matrix)).val == 3
    @test resolve_parameters((; val = ParameterReference(:condition, 2)), x,
        (; condition = ((<=), 7))).val == 7
    nested = (; condition = ((<=), InputReference(4)),
        profile = (; values = ParameterReference(:matrix), target = InputReference(1)))
    result = resolve_parameters(nested, x, (; matrix))
    @test result.condition == ((<=), 4)
    @test result.profile.values === matrix
    @test result.profile.target == 1
    @test_throws ErrorException resolve_parameters(bindings, x, (;))
    @test_throws BoundsError resolve_parameters(bindings, x[1:3], (; matrix))
    @test_throws BoundsError resolve_parameters(
        (; val = ParameterReference(:matrix, 9, 1)), x, (; matrix))
    # Only mapped keywords cross the adapter, even with unrelated input metadata.
    component(values; val) = sum(values) - val
    routed = RoutedComposition(component; input = InputReference(1:3),
        parameters = (; val = InputReference(4)))
    @test routed(x; ignored = :metadata) == 2
    x[4] = 6
    @test routed(x) == 0
    @test !incremental_supported(routed)
end
