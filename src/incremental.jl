"""
    IncrementalComposition

A compiled standard ICN topology whose selected primitive operations are part of its concrete
type. The object remains an ordinary callable cost function; `incremental_state` additionally
creates mutable state for candidate evaluation and commits.
"""
abstract type AbstractIncrementalComposition <: Function end

struct IncrementalComposition{T, A, G, C} <: AbstractIncrementalComposition
    transformations::T
    arithmetic::A
    aggregation::G
    comparison::C
end

"""Compiled residual for a pairwise family of grouped affine disjunctions."""
struct PairwiseDisjunctionComposition{A} <: AbstractIncrementalComposition
    aggregation::A
end

"""Compiled reduction of operations applied to aligned decision/parameter pairs."""
struct AlignedPairComposition{O, A} <: AbstractIncrementalComposition
    operation::O
    aggregation::A
end

"""Compiled L1 residual for cyclic indirect indexing over equal-sized blocks."""
struct CyclicIndexComposition <: AbstractIncrementalComposition end

"""Compiled L1 residual to the indicator vector selected by a one-based index."""
struct IndicatorIndexComposition <: AbstractIncrementalComposition end

"""Incremental sum of atomic functional-graph and active-cardinality components."""
struct FunctionalGraphComposition{C} <: AbstractIncrementalComposition
    components::C
end

FunctionalGraphComposition() = FunctionalGraphComposition(nothing)

"""Compiled dynamic-programming distance for an arbitrary `AbstractLanguage`."""
struct LanguageDistanceComposition{C} <: AbstractIncrementalComposition
    component::C
end

"""Compiled duplicate count over distances of disjoint adjacent pairs."""
struct PairDistanceCollisionComposition{C} <: AbstractIncrementalComposition
    component::C
end

"""Compiled maximum-load residual for a weighted interval profile."""
struct EventProfileComposition{C} <: AbstractIncrementalComposition
    comparison::C
end

"""Compiled integral residual for a piecewise-constant weighted interval profile."""
struct EventAreaComposition{R} <: AbstractIncrementalComposition
    reduction::R
end

"""Compiled distinct-count composition preceded by a value-local filter."""
struct FilteredDistinctComposition{F, C} <: AbstractIncrementalComposition
    filter::F
    comparison::C
end

const _INCREMENTAL_TRANSFORMATIONS = (:id, :count_equal_left, :count_equal_right)
const _INCREMENTAL_ARITHMETIC = (:sum, :product)
const _INCREMENTAL_AGGREGATIONS = (:sum, :count_zero, :count_positive)
const _INCREMENTAL_COMPARISONS = (
    :id,
    :condition_residual,
    :abs_val,
    :val_minus_var,
    :var_minus_val,
    :euclidean_val,
    :euclidean_val_op,
    :euclidean,
    :euclidean_op,
    :var_minus_numvars,
    :max_numvars_minus_var,
    :max_var_minus_numvars,
    :vals_minus_var_gele,
    :vals_minus_var_gl
)

function _pairwise_disjunction_incremental_supported(icn::AbstractICN)
    Tuple(layer.name for layer in icn.layers) ==
        (:PairedMap, :PairMask, :GroupReduction, :Transformation,
         :Arithmetic, :Aggregation, :Comparison) ||
        return false
    selected = _selected_operation_names(icn)
    return selected[1] == [:pairwise_oriented_affine_margins] &&
           selected[2] == [:zero_extent_groups] &&
           selected[3] == [:minimum] &&
           selected[5] == [:sum] &&
           ((selected[4] == [:positive_part] &&
             only(selected[6]) in (:sum, :count_positive)) ||
            (selected[4] == [:id] && selected[6] == [:count_positive])) &&
           selected[7] == [:id]
end

function _aligned_pair_incremental_supported(icn::AbstractICN)
    length(icn.layers) == 5 || return false
    Tuple(layer.name for layer in icn.layers) ==
        (:PairedMap, :Transformation, :Arithmetic, :Aggregation, :Comparison) ||
        return false
    selected = _selected_operation_names(icn)
    front_and_transformation =
        (selected[1] in ([:sub], [:aligned_not_equal]) && selected[2] == [:id]) ||
        (selected[1] == [:aligned_difference] && selected[2] == [:absolute])
    return front_and_transformation && selected[3] == [:sum] &&
           length(selected[4]) == 1 &&
           only(selected[4]) in (:sum, :count_positive) && selected[5] == [:id]
end

function _event_profile_incremental_supported(icn::AbstractICN)
    Tuple(layer.name for layer in icn.layers) ==
        (:EventMap, :SegmentMap, :Arithmetic, :Aggregation, :Comparison) ||
        return false
    selected = _selected_operation_names(icn)
    return selected[1] == [:weighted_interval_segments] &&
           selected[2] == [:loads] && selected[3] == [:sum] &&
           selected[4] == [:maximum] &&
           only(selected[5]) in _INCREMENTAL_COMPARISONS
end

function _event_area_incremental_supported(icn::AbstractICN)
    Tuple(layer.name for layer in icn.layers) ==
        (:EventMap, :SegmentMap, :Arithmetic, :Aggregation, :Comparison) ||
        return false
    selected = _selected_operation_names(icn)
    return selected[1] == [:weighted_interval_segments] &&
           Set(selected[2]) == Set((:widths, :condition_residuals)) &&
           selected[3] == [:product] &&
           selected[4] == [:sum] &&
           selected[5] == [:id]
end

const _INCREMENTAL_VALUE_FILTERS = (
    :id,
    :filter_unique,
    :filter_op_val,
    :filter_equal_val,
    :filter_ge_val,
    :filter_great_val,
    :filter_less_val,
    :filter_le_val,
    :filter_ne_val,
    :filter_equal_filter_val,
    :filter_op_vals,
    :filter_equal_vals,
    :filter_ne_vals,
)

function _filtered_distinct_incremental_supported(icn::AbstractICN)
    length(icn.layers) == 5 || return false
    Tuple(layer.name for layer in icn.layers) ==
        (:SimpleFilter, :Transformation, :Arithmetic, :Aggregation, :Comparison) ||
        return false
    selected = _selected_operation_names(icn)
    return length(selected[1]) == 1 && only(selected[1]) in _INCREMENTAL_VALUE_FILTERS &&
           length(selected[2]) == 1 &&
           only(selected[2]) in (:count_equal_left, :count_equal_right) &&
           selected[3] == [:sum] &&
           selected[4] == [:count_zero] &&
           length(selected[5]) == 1 && only(selected[5]) in _INCREMENTAL_COMPARISONS
end

function _language_distance_incremental_supported(icn::AbstractICN)
    length(icn.layers) == 5 || return false
    Tuple(layer.name for layer in icn.layers) ==
        (:Language, :Transformation, :Arithmetic, :Aggregation, :Comparison) ||
        return false
    selected = _selected_operation_names(icn)
    return selected[1] == [:distance] && selected[2] == [:id] &&
           length(selected[3]) == 1 && only(selected[3]) in (:sum, :product) &&
           selected[4] == [:sum] && selected[5] == [:id]
end

function incremental_supported(icn::AbstractICN)
    structural_kernel = _index_relation_kernel(icn)
    structural_kernel isa Union{
        Val{:cyclic}, Val{:indicator}, Val{:pair_distance_collisions},
    } && return true
    _filtered_distinct_incremental_supported(icn) && return true
    _language_distance_incremental_supported(icn) && return true
    _event_area_incremental_supported(icn) && return true
    _event_profile_incremental_supported(icn) && return true
    _pairwise_disjunction_incremental_supported(icn) && return true
    _aligned_pair_incremental_supported(icn) && return true
    _supports_inplace_compilation(icn) || return false
    selected = _selected_operation_names(icn)
    return !isempty(selected[1]) &&
           all(operation -> operation in _INCREMENTAL_TRANSFORMATIONS, selected[1]) &&
           only(selected[2]) in _INCREMENTAL_ARITHMETIC &&
           only(selected[3]) in _INCREMENTAL_AGGREGATIONS &&
           only(selected[4]) in _INCREMENTAL_COMPARISONS
end

incremental_supported(::MatrixRowsComposition) = true
incremental_supported(::FunctionalGraphComposition) = true
incremental_supported(::ParameterRowsComposition) = false
incremental_supported(
    ::ParameterRowsComposition{R, Val{:aligned_mismatch}},
) where {R} = true

function _functional_graph_additive_supported(composition::AdditiveComposition)
    length(composition.components) == 3 || return false
    kernels = Set{Symbol}()
    for component in composition.components
        component isa Composition || return false
        component.network isa AbstractICN || return false
        kernel = _index_relation_kernel(component.network)
        kernel isa Val{:functional_graph_predecessors} && push!(kernels, :predecessors)
        kernel isa Val{:functional_graph_orbit} && push!(kernels, :orbit)
        kernel isa Val{:nonfixed_condition} && push!(kernels, :size)
    end
    return kernels == Set((:predecessors, :orbit, :size))
end

incremental_supported(composition::AdditiveComposition) =
    _functional_graph_additive_supported(composition)
incremental_supported(composition::GroupedComposition) =
    all(incremental_supported, composition.components)

"""Compile the supported primitive operations selected by a standard four-layer ICN."""
function incremental_composition(icn::AbstractICN)
    incremental_supported(icn) || throw(ArgumentError(
        "the selected ICN primitives do not yet support incremental evaluation"))
    index_relation = _index_relation_kernel(icn)
    index_relation isa Val{:cyclic} && return CyclicIndexComposition()
    index_relation isa Val{:indicator} && return IndicatorIndexComposition()
    index_relation isa Val{:pair_distance_collisions} &&
        return PairDistanceCollisionComposition(
            composition(icn; name = :pair_distance_collisions),
        )
    selected = _selected_operation_names(icn)
    if _filtered_distinct_incremental_supported(icn)
        return FilteredDistinctComposition(
            Val(only(selected[1])), Val(only(selected[5])))
    end
    if _language_distance_incremental_supported(icn)
        return LanguageDistanceComposition(composition(icn; name = :language_distance))
    end
    if _event_area_incremental_supported(icn)
        return EventAreaComposition(Val(:segment_condition_area))
    end
    if _event_profile_incremental_supported(icn)
        return EventProfileComposition(Val(only(selected[5])))
    end
    if _pairwise_disjunction_incremental_supported(icn)
        return PairwiseDisjunctionComposition(Val(only(selected[6])))
    end
    if _aligned_pair_incremental_supported(icn)
        operation = selected[1] == [:aligned_difference] ? :sub : only(selected[1])
        return AlignedPairComposition(
            Val(operation), Val(only(selected[4])))
    end
    transformations = Tuple(Val(operation) for operation in selected[1])
    return IncrementalComposition(
        transformations,
        Val(only(selected[2])),
        Val(only(selected[3])),
        Val(only(selected[4]))
    )
end

function incremental_composition(composition::AdditiveComposition)
    incremental_supported(composition) || throw(ArgumentError(
        "the additive ICN components do not yet support incremental evaluation",
    ))
    return FunctionalGraphComposition(composition)
end
function incremental_composition(composition::GroupedComposition)
    incremental_supported(composition) || throw(ArgumentError(
        "one or more grouped ICN components do not support incremental evaluation",
    ))
    return composition
end


incremental_composition(composition::FunctionalGraphComposition) = composition
incremental_composition(composition::LanguageDistanceComposition) = composition
incremental_composition(composition::PairDistanceCollisionComposition) = composition
incremental_composition(composition::ParameterRowsComposition) =
    incremental_supported(composition) ? composition : throw(ArgumentError(
        "the selected parameter-row primitives do not yet support incremental evaluation",
    ))

function canonical_key(composition::FunctionalGraphComposition; simplified::Bool = true)
    isnothing(composition.components) && return "functional_graph"
    return canonical_key(composition.components; simplified)
end

function code(composition::FunctionalGraphComposition, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    isnothing(composition.components) && return "$(name)(x) = functional_graph_residual(x)"
    return code(composition.components, language; name, simplified)
end

canonical_key(composition::LanguageDistanceComposition; simplified::Bool = true) =
    canonical_key(composition.component; simplified)
function code(composition::LanguageDistanceComposition, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    return code(composition.component, language; name, simplified)
end

canonical_key(composition::PairDistanceCollisionComposition; simplified::Bool = true) =
    canonical_key(composition.component; simplified)
function code(composition::PairDistanceCollisionComposition, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    return code(composition.component, language; name, simplified)
end

@inline function (composition::PairDistanceCollisionComposition)(x; parameters...)
    return _pair_distance_collision_penalty(x; parameters...)
end

function (composition::LanguageDistanceComposition)(x;
        language, language_workspace = nothing, X = nothing, parameters...)
    workspace = isnothing(X) ? language_workspace : X
    return Float64(isnothing(workspace) ?
        ConstraintCommons.language_distance(language, x) :
        ConstraintCommons.language_distance(language, x, workspace))
end

function (composition::FilteredDistinctComposition)(x; parameters...)
    distinct = _aggregate_specialized(
        Val(:SimpleFilter), composition.filter, Val(:count_equal_left),
        Val(:count_zero), x; parameters...)
    return Float64(_compare(composition.comparison, distinct; parameters...))
end


function (composition::EventAreaComposition)(x;
        X = nothing, pair_vars, op, val, parameters...)
    workspace = X isa EventProfileWorkspace ? X : incremental_workspace(
        composition, x; pair_vars, op, val, parameters...,
    )
    return _weighted_interval_condition_area!(
        workspace.events, composition.reduction, x, pair_vars, op, val)
end

function (composition::PairwiseDisjunctionComposition)(x;
        pair_vars, dim = 1, bool = true, parameters...)
    return Float64(_aggregate_pairwise_disjunction(
        composition.aggregation, x; pair_vars, dim, bool))
end

function (composition::AlignedPairComposition)(x; pair_vars, parameters...)
    return Float64(_aggregate_paired(
        composition.operation, composition.aggregation, x; pair_vars, parameters...))
end


(::CyclicIndexComposition)(x; dim, parameters...) =
    _cyclic_index_l1(x; dim, parameters...)

(::IndicatorIndexComposition)(x; id, parameters...) =
    _indicator_index_l1(x; id, parameters...)

function (::FunctionalGraphComposition)(x; X = nothing, op::F = (>=), val = 2) where {F}
    if X isa FunctionalGraphWorkspace
        return _functional_graph_penalty!(X, x, op, val)
    end
    return _functional_graph_predecessor_penalty(x) +
           _functional_graph_orbit_penalty(x) +
           _nonfixed_condition_penalty(x, op, val)
end

function (composition::EventProfileComposition)(x;
        X = nothing, pair_vars, parameters...)
    workspace = X isa EventProfileWorkspace ? X : incremental_workspace(
        composition, x; pair_vars, parameters...,
    )
    maximum_load = _maximum_weighted_interval_load!(workspace.events, x, pair_vars)
    return Float64(_compare(composition.comparison, maximum_load; parameters...))
end

@inline _transform_columns!(::Tuple{}, workspace, x, column; parameters...) = workspace
@inline function _transform_columns!(operations::Tuple, workspace, x, column;
        parameters...)
    _transform!(first(operations), _WorkspaceColumn(workspace, column, length(x)), x;
        parameters...)
    return _transform_columns!(Base.tail(operations), workspace, x, column + 1;
        parameters...)
end

function (composition::IncrementalComposition)(x;
        X = composition_workspace(length(composition.transformations), x), parameters...)
    rows = length(x)
    columns = length(composition.transformations)
    size(X, 1) >= rows && size(X, 2) >= columns || throw(DimensionMismatch(
        "composition workspace must have at least $(rows)x$(columns) elements"))
    _transform_columns!(composition.transformations, X, x, 1; parameters...)
    columns > 1 && _combine_rows!(composition.arithmetic, X, rows, columns)
    aggregate = _aggregate(composition.aggregation, X, rows; parameters...)
    return Float64(_compare(composition.comparison, aggregate; parameters...))
end

"""Persistent buffers for an incremental composition. They may be owned by the caller."""
abstract type AbstractIncrementalWorkspace end

"Caller-owned occurrence counters for matrix-row compositions."
struct MatrixRowsWorkspace{V <: AbstractVector{Int}} <: AbstractIncrementalWorkspace
    counts::V
end

"Caller-owned row mismatch counters for a collection-valued parameter."
struct ParameterRowsWorkspace{V <: AbstractVector{Int}} <: AbstractIncrementalWorkspace
    mismatches::V
end

function incremental_workspace(
        ::MatrixRowsComposition, values::AbstractVector; vals::AbstractMatrix,
        parameters...)
    return MatrixRowsWorkspace(zeros(Int, size(vals, 1)))
end

function incremental_workspace(
        ::ParameterRowsComposition{R, Val{:aligned_mismatch}},
        values::AbstractVector;
        pair_vars,
        parameters...,
) where {R}
    return ParameterRowsWorkspace(zeros(Int, _parameter_row_count(pair_vars)))
end

function incremental_workspace(
        ::LanguageDistanceComposition,
        values::AbstractVector;
        language,
        parameters...,
)
    return ConstraintCommons.language_distance_workspace(language)
end

struct IncrementalWorkspace{M <: AbstractMatrix, V <: AbstractVector} <:
       AbstractIncrementalWorkspace
    transformations::M
    combined::V
end

"Caller-owned frequencies for excess-value or distinct-value composition states."
struct ValueExcessWorkspace{D <: AbstractDict} <: AbstractIncrementalWorkspace
    counts::D
end

"Caller-owned marker selecting the scalar state for a sum of identity-transformed values."
struct SumWorkspace{T} <: AbstractIncrementalWorkspace end

"Caller-owned marker for a scalar aligned-pair reduction."
struct AlignedPairWorkspace <: AbstractIncrementalWorkspace end

"Caller-owned marker for a signature-derived parameter-binding adapter."
struct ParameterBindingWorkspace <: AbstractIncrementalWorkspace end

"""Allocation-free marker for an indexed-relation incremental state."""
struct IndexRelationWorkspace <: AbstractIncrementalWorkspace end

"""Caller-owned counters and visitation marks for a functional graph."""
struct FunctionalGraphWorkspace{
    C <: AbstractVector{Int}, V <: AbstractVector{Bool},
} <: AbstractIncrementalWorkspace
    counts::C
    visited::V
end

"""Reverse adjacency lists for cyclic indirect-index dependencies."""
struct CyclicIndexWorkspace{
    M <: AbstractMatrix{Int},
    V <: AbstractVector{Int},
} <: AbstractIncrementalWorkspace
    heads::M
    next::V
    previous::V
end

incremental_workspace(
    ::AlignedPairComposition, values::AbstractVector; parameters...) = AlignedPairWorkspace()
incremental_workspace(
    ::ParameterBindingComposition, values::AbstractVector; parameters...) =
    ParameterBindingWorkspace()
incremental_workspace(
    ::IndicatorIndexComposition,
    values::AbstractVector;
    parameters...,
) = IndexRelationWorkspace()
incremental_workspace(
    ::PairDistanceCollisionComposition,
    values::AbstractVector;
    parameters...,
) = IndexRelationWorkspace()

incremental_workspace(
    ::FunctionalGraphComposition,
    values::AbstractVector;
    parameters...,
) = FunctionalGraphWorkspace(zeros(Int, length(values)), falses(length(values)))

function incremental_workspace(
        ::CyclicIndexComposition,
        values::AbstractVector;
        dim,
        parameters...,
)
    blocks, width = _block_layout(values, Int(dim))
    return CyclicIndexWorkspace(
        zeros(Int, blocks, width),
        zeros(Int, length(values)),
        zeros(Int, length(values)),
    )
end


@testitem "Disjoint pair-distance composition stays exact and allocation-free" begin
    using Test
    import CompositionalNetworks as CN

    layers = [
        CN.Transformation, CN.Arithmetic, CN.Pointwise, CN.Transformation,
        CN.Arithmetic, CN.Aggregation, CN.Comparison,
    ]
    network = CN.ICN(; layers, connection = UInt32.(1:length(layers)))
    fill!(network.weights.parent, false)
    selections = (
        :disjoint_pair_differences,
        :sum,
        :absolute,
        :count_equal_left,
        :sum,
        :count_positive,
        :id,
    )
    let offset = 0
        for (layer, selected) in zip(network.layers, selections)
            names = collect(keys(layer.fn))
            network.weights.parent[(offset + findfirst(==(selected), names))] = true
            offset += length(layer.fn)
        end
    end
    compiled = CN.incremental_composition(network)
    @test compiled isa CN.PairDistanceCollisionComposition
    values = [1, 4, -2, 1]
    state = CN.incremental_state(compiled, values)
    @test CN.incremental_value(state) == compiled(values) == 1.0
    @test CN.incremental_update!(state, 4, 2) == compiled([1, 4, -2, 2]) == 0.0
    @test CN.incremental_update!(state, 4, 1) == 1.0
    CN.incremental_update!(state, 4, 2)
    update_allocations(state) = @allocated CN.incremental_update!(state, 4, 1)
    @test update_allocations(state) == 0
end

"Caller-owned pair residuals for a grouped affine-disjunction composition."
struct PairwiseDisjunctionWorkspace{V <: AbstractVector} <: AbstractIncrementalWorkspace
    residuals::V
end

"Caller-owned sweep events for a weighted interval profile."
struct EventProfileWorkspace{V <: AbstractVector} <: AbstractIncrementalWorkspace
    events::V
end

function _maximum_weighted_interval_load!(events, values, pair_vars)
    _check_weighted_interval_arguments(values, pair_vars)
    length(events) == 2length(values) || throw(DimensionMismatch(
        "event workspace must contain exactly two events per origin",
    ))
    @inbounds for task in eachindex(values)
        event = 2task - 1
        start = values[task]
        height = pair_vars[2, task]
        events[event] = (start, height)
        events[event + 1] = (start + pair_vars[1, task], -height)
    end
    sort!(events; alg = Base.Sort.QuickSort)

    usage = 0.0
    maximum_load = 0.0
    event = firstindex(events)
    @inbounds while event <= lastindex(events)
        time = events[event][1]
        while event <= lastindex(events) && events[event][1] == time
            usage += events[event][2]
            event += 1
        end
        maximum_load = max(maximum_load, usage)
    end
    return maximum_load
end

function incremental_workspace(composition::IncrementalComposition, input_length::Integer;
        type::Type = Float64)
    transformations = Matrix{type}(
        undef, input_length, length(composition.transformations))
    return IncrementalWorkspace(transformations, Vector{type}(undef, input_length))
end

function incremental_workspace(
        ::FilteredDistinctComposition,
        values::AbstractVector{T};
        type::Type = Float64,
        parameters...,
) where {T}
    return ValueExcessWorkspace(Dict{T, Int}())
end

function incremental_workspace(composition::IncrementalComposition, values::AbstractVector;
        type::Type = Float64)
    return incremental_workspace(composition, length(values); type)
end

function incremental_workspace(
        ::PairwiseDisjunctionComposition,
        input_length::Integer;
        dim = 1,
        type::Type = Float64,
        parameters...,
)
    dimensions = Int(dim)
    dimensions > 0 || throw(ArgumentError("pairwise dimension must be positive"))
    input_length % dimensions == 0 || throw(DimensionMismatch(
        "pairwise coordinates must be divisible by the dimension"))
    tasks = input_length ÷ dimensions
    return PairwiseDisjunctionWorkspace(
        Vector{type}(undef, tasks * (tasks - 1) ÷ 2))
end

function incremental_workspace(
        composition::PairwiseDisjunctionComposition,
        values::AbstractVector;
        parameters...,
)
    return incremental_workspace(composition, length(values); parameters...)
end

function incremental_workspace(
        ::EventProfileComposition,
        input_length::Integer;
        type::Type = Float64,
        parameters...,
)
    return EventProfileWorkspace(
        Vector{Tuple{type, type}}(undef, 2input_length),
    )
end


function incremental_workspace(
        ::EventAreaComposition,
        input_length::Integer;
        type::Type = Float64,
        parameters...,
)
    return EventProfileWorkspace(
        Vector{Tuple{type, type}}(undef, 2input_length),
    )
end

function incremental_workspace(
        composition::EventAreaComposition,
        values::AbstractVector;
        parameters...,
)
    return incremental_workspace(composition, length(values); parameters...)
end

function incremental_workspace(
        composition::EventProfileComposition,
        values::AbstractVector;
        parameters...,
)
    return incremental_workspace(composition, length(values); parameters...)
end

function incremental_workspace(
        ::IncrementalComposition{
            Tuple{Val{:count_equal_left}}, Val{:sum}, Val{:count_positive}, Val{:id}},
        values::AbstractVector{T}; type::Type = Float64) where {T}
    return ValueExcessWorkspace(Dict{T, Int}())
end

function incremental_workspace(
        ::IncrementalComposition{
            Tuple{Val{:count_equal_right}}, Val{:sum}, Val{:count_positive}, Val{:id}},
        values::AbstractVector{T}; type::Type = Float64) where {T}
    return ValueExcessWorkspace(Dict{T, Int}())
end

function incremental_workspace(
        ::IncrementalComposition{
            Tuple{E}, A, Val{:count_zero}, C},
        values::AbstractVector{T}; type::Type = Float64,
) where {E <: Union{Val{:count_equal_left}, Val{:count_equal_right}}, A, C, T}
    return ValueExcessWorkspace(Dict{T, Int}())
end

function incremental_workspace(
        ::IncrementalComposition{Tuple{Val{:id}}, A, Val{:sum}, C},
        values::AbstractVector; type::Type = Float64) where {A, C}
    return SumWorkspace{type}()
end

abstract type AbstractIncrementalCompositionState end

mutable struct IncrementalCompositionState{
    C <: IncrementalComposition, V <: AbstractVector, W <: IncrementalWorkspace, P, A} <:
               AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    parameters::P
    aggregate::A
    current::Float64
end

mutable struct ValueExcessCompositionState{
    C <: IncrementalComposition, V <: AbstractVector, W <: ValueExcessWorkspace, P} <:
               AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    parameters::P
    duplicates::Int
    current::Float64
end

mutable struct DistinctCountCompositionState{
    C <: IncrementalComposition, V <: AbstractVector, W <: ValueExcessWorkspace, P} <:
               AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    parameters::P
    distinct::Int
    current::Float64
end

mutable struct FilteredDistinctCompositionState{
    C <: FilteredDistinctComposition,
    V <: AbstractVector,
    W <: ValueExcessWorkspace,
    P,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    parameters::P
    distinct::Int
    current::Float64
end

mutable struct SumCompositionState{
    C <: IncrementalComposition, V <: AbstractVector, W <: SumWorkspace, P, A} <:
               AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    parameters::P
    aggregate::A
    current::Float64
end

mutable struct PairwiseDisjunctionCompositionState{
    C <: PairwiseDisjunctionComposition,
    V <: AbstractVector,
    W <: PairwiseDisjunctionWorkspace,
    P,
    A,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    parameters::P
    aggregate::A
    current::Float64
end

mutable struct AlignedPairCompositionState{
    C <: AlignedPairComposition,
    V <: AbstractVector,
    W <: AlignedPairWorkspace,
    P,
    A,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    parameters::P
    aggregate::A
    current::Float64
end

mutable struct ParameterBindingCompositionState{
    C <: ParameterBindingComposition,
    V <: AbstractVector,
    W <: ParameterBindingWorkspace,
    P,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    parameters::P
    current::Float64
end


mutable struct IndexRelationCompositionState{
    C <: Union{CyclicIndexComposition, IndicatorIndexComposition},
    V <: AbstractVector,
    W <: Union{IndexRelationWorkspace, CyclicIndexWorkspace},
    P,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    parameters::P
    current::Float64
end

mutable struct FunctionalGraphCompositionState{
    V <: AbstractVector,
    W <: FunctionalGraphWorkspace,
    P,
} <: AbstractIncrementalCompositionState
    composition::FunctionalGraphComposition
    values::V
    workspace::W
    parameters::P
    current::Float64
end

mutable struct LanguageDistanceCompositionState{
    C <: LanguageDistanceComposition,
    V <: AbstractVector,
    L,
    W,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    language::L
    workspace::W
    current::Float64
end


mutable struct PairDistanceCollisionCompositionState{
    C <: PairDistanceCollisionComposition,
    V <: AbstractVector,
    W <: IndexRelationWorkspace,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    current::Float64
end

mutable struct EventProfileCompositionState{
    C <: Union{EventProfileComposition, EventAreaComposition},
    V <: AbstractVector,
    W <: EventProfileWorkspace,
    P,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    workspace::W
    parameters::P
    current::Float64
end

mutable struct MatrixRowsCompositionState{
    C <: MatrixRowsComposition,
    V <: AbstractVector,
    M <: AbstractMatrix,
    W <: MatrixRowsWorkspace,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    vals::M
    bool::Bool
    workspace::W
    outside::Int
    current::Float64
end

mutable struct ParameterRowsCompositionState{
    C <: ParameterRowsComposition,
    V <: AbstractVector,
    P,
    W <: ParameterRowsWorkspace,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    pair_vars::P
    workspace::W
    current::Float64
end
mutable struct GroupedCompositionState{
    C <: GroupedComposition,
    V <: AbstractVector,
    S <: Tuple,
} <: AbstractIncrementalCompositionState
    composition::C
    values::V
    states::S
    current::Float64
end

@inline function _combined_row(::Val{:sum}, transformations, row, columns)
    value = zero(eltype(transformations))
    @inbounds for column in 1:columns
        value += transformations[row, column]
    end
    return value
end

@inline function _combined_row(::Val{:product}, transformations, row, columns)
    value = one(eltype(transformations))
    @inbounds for column in 1:columns
        value *= transformations[row, column]
    end
    return value
end

function _initialize_combined!(combined, arithmetic, transformations, rows, columns)
    @inbounds for row in 1:rows
        combined[row] = _combined_row(arithmetic, transformations, row, columns)
    end
    return combined
end

function _initialize_combined!(combined, arithmetic, transformations, rows, ::Val{1})
    @inbounds for row in 1:rows
        combined[row] = transformations[row, 1]
    end
    return combined
end

function _initialize_combined!(combined, arithmetic, transformations, rows,
        ::Val{C}) where {C}
    return _initialize_combined!(combined, arithmetic, transformations, rows, C)
end

function _initial_aggregate(::Val{:sum}, combined, rows)
    value = zero(eltype(combined))
    @inbounds for row in 1:rows
        value += combined[row]
    end
    return value
end

function _initial_aggregate(::Val{:count_positive}, combined, rows)
    value = 0
    @inbounds for row in 1:rows
        value += combined[row] > 0
    end
    return value
end

function _initial_aggregate(::Val{:count_zero}, combined, rows)
    value = 0
    @inbounds for row in 1:rows
        value += iszero(combined[row])
    end
    return value
end

@inline _update_aggregate(::Val{:sum}, aggregate, old, new) = aggregate + new - old
@inline _update_aggregate(::Val{:count_zero}, aggregate, old, new) = aggregate +
                                                                      iszero(new) -
                                                                      iszero(old)
@inline _update_aggregate(::Val{:count_positive}, aggregate, old, new) = aggregate +
                                                                         (new > 0) -
                                                                         (old > 0)

function incremental_state(composition::IncrementalComposition, values;
        workspace = incremental_workspace(composition, values), parameters...)
    return _incremental_state(composition, values, workspace, (; parameters...))
end

function incremental_state(
        composition::MatrixRowsComposition,
        values;
        vals::AbstractMatrix,
        bool::Bool = false,
        workspace = incremental_workspace(composition, values; vals, bool),
        parameters...,
)
    size(vals, 2) >= 1 || throw(ArgumentError(
        "matrix-valued vals requires at least one column",
    ))
    length(workspace.counts) == size(vals, 1) || throw(DimensionMismatch(
        "matrix-row workspace requires one counter per vals row",
    ))
    state = MatrixRowsCompositionState(
        composition, collect(values), vals, bool, workspace, 0, 0.0,
    )
    incremental_rebuild!(state, values)
    return state
end

function incremental_state(
        composition::ParameterRowsComposition{R, Val{:aligned_mismatch}},
        values;
        pair_vars,
        workspace = incremental_workspace(composition, values; pair_vars),
        parameters...,
) where {R}
    Base.require_one_based_indexing(values, pair_vars)
    length(workspace.mismatches) == _parameter_row_count(pair_vars) ||
        throw(DimensionMismatch(
            "parameter-row workspace requires one counter per parameter row",
        ))
    state = ParameterRowsCompositionState(
        composition, collect(values), pair_vars, workspace, 0.0,
    )
    incremental_rebuild!(state, values)
    return state
end

function incremental_state(
        composition::LanguageDistanceComposition,
        values;
        language,
        workspace = incremental_workspace(composition, values; language),
        parameters...,
)
    state = LanguageDistanceCompositionState(
        composition, collect(values), language, workspace, 0.0,
    )
    incremental_rebuild!(state, values)
    return state
end


function incremental_state(
        composition::PairDistanceCollisionComposition,
        values;
        workspace = IndexRelationWorkspace(),
        parameters...,
)
    workspace isa IndexRelationWorkspace || throw(ArgumentError(
        "pair-distance compositions require an IndexRelationWorkspace",
    ))
    state = PairDistanceCollisionCompositionState(
        composition, collect(values), workspace, 0.0,
    )
    incremental_rebuild!(state, values)
    return state
end

function incremental_state(
        composition::GroupedComposition,
        values;
        pair_vars,
        parameters...,
)
    length(pair_vars) == length(composition.components) || throw(DimensionMismatch(
        "one parameter group is required per grouped composition component",
    ))
    states = ntuple(length(composition.components)) do index
        incremental_state(
            composition.components[index], values;
            pair_vars = pair_vars[index], parameters...,
        )
    end
    state = GroupedCompositionState(composition, collect(values), states, 0.0)
    state.current = _grouped_state_value(composition, states)
    return state
end

function incremental_state(composition::FilteredDistinctComposition, values;
        workspace = nothing, parameters...)
    actual_workspace = isnothing(workspace) ?
                       incremental_workspace(composition, values; parameters...) : workspace
    return _incremental_state(
        composition, values, actual_workspace, (; parameters...))
end

function incremental_state(
        composition::PairwiseDisjunctionComposition,
        values;
        workspace = nothing,
        parameters...,
)
    params = (; parameters...)
    actual_workspace = isnothing(workspace) ?
                       incremental_workspace(composition, values; parameters...) : workspace
    actual_workspace isa PairwiseDisjunctionWorkspace || throw(ArgumentError(
        "pairwise disjunction compositions require a PairwiseDisjunctionWorkspace"))
    state = PairwiseDisjunctionCompositionState(
        composition,
        collect(values),
        actual_workspace,
        params,
        composition.aggregation isa Val{:sum} ? 0.0 : 0,
        0.0,
    )
    incremental_rebuild!(state, values)
    return state
end

function incremental_state(
        composition::AlignedPairComposition,
        values;
        pair_vars,
        workspace = incremental_workspace(composition, values; pair_vars),
        parameters...,
)
    params = (; pair_vars, parameters...)
    axes(values) == axes(pair_vars) || throw(DimensionMismatch(
        "aligned values and pair_vars must have the same axes"))
    workspace isa AlignedPairWorkspace || throw(ArgumentError(
        "aligned-pair compositions require an AlignedPairWorkspace"))
    aggregate = composition.aggregation isa Val{:sum} ? 0.0 : 0
    state = AlignedPairCompositionState(
        composition, collect(values), workspace, params, aggregate, 0.0)
    incremental_rebuild!(state, values)
    return state
end

function incremental_state(
        composition::ParameterBindingComposition,
        values;
        workspace = incremental_workspace(composition, values),
        parameters...,
)
    workspace isa ParameterBindingWorkspace || throw(ArgumentError(
        "parameter-binding compositions require a ParameterBindingWorkspace"))
    state = ParameterBindingCompositionState(
        composition, collect(values), workspace, (; parameters...), 0.0)
    incremental_rebuild!(state, values)
    return state
end


function incremental_state(
        composition::Union{CyclicIndexComposition, IndicatorIndexComposition},
        values;
        workspace = nothing,
        parameters...,
)
    actual_workspace = isnothing(workspace) ?
                       incremental_workspace(composition, values; parameters...) : workspace
    if composition isa CyclicIndexComposition
        actual_workspace isa CyclicIndexWorkspace || throw(ArgumentError(
            "cyclic indexed relations require a CyclicIndexWorkspace"))
    else
        actual_workspace isa IndexRelationWorkspace || throw(ArgumentError(
            "indicator indexed relations require an IndexRelationWorkspace"))
    end
    state = IndexRelationCompositionState(
        composition, collect(values), actual_workspace, (; parameters...), 0.0)
    incremental_rebuild!(state, values)
    return state
end

function incremental_state(
        composition::FunctionalGraphComposition,
        values;
        workspace = incremental_workspace(composition, values),
        op = (>=),
        val = 2,
        parameters...,
)
    workspace isa FunctionalGraphWorkspace || throw(ArgumentError(
        "functional-graph compositions require a FunctionalGraphWorkspace",
    ))
    state = FunctionalGraphCompositionState(
        composition, collect(values), workspace, (; op, val, parameters...), 0.0,
    )
    incremental_rebuild!(state, values)
    return state
end

function incremental_state(
        composition::Union{EventProfileComposition, EventAreaComposition},
        values;
        workspace = nothing,
        parameters...,
)
    params = (; parameters...)
    haskey(params, :pair_vars) || throw(ArgumentError(
        "weighted interval profiles require pair_vars",
    ))
    actual_workspace = isnothing(workspace) ?
                       incremental_workspace(composition, values; parameters...) : workspace
    actual_workspace isa EventProfileWorkspace || throw(ArgumentError(
        "weighted interval profiles require an EventProfileWorkspace",
    ))
    _check_weighted_interval_arguments(values, params.pair_vars)
    length(actual_workspace.events) == 2length(values) || throw(DimensionMismatch(
        "weighted interval event workspace must contain two events per origin",
    ))
    state = EventProfileCompositionState(
        composition, collect(values), actual_workspace, params, 0.0,
    )
    incremental_rebuild!(state, values)
    return state
end

function _incremental_state(composition::IncrementalComposition, values,
        workspace::IncrementalWorkspace, parameters)
    length(workspace.combined) >= length(values) || throw(DimensionMismatch(
        "incremental workspace must contain at least $(length(values)) rows"))
    size(workspace.transformations, 1) >= length(values) &&
    size(workspace.transformations, 2) >= length(composition.transformations) ||
        throw(DimensionMismatch("incremental transformation workspace is too small"))
    state = IncrementalCompositionState(
        composition,
        collect(values),
        workspace,
        parameters,
        zero(eltype(workspace.combined)),
        0.0
    )
    incremental_rebuild!(state, values)
    return state
end

function _incremental_state(
        composition::FilteredDistinctComposition,
        values,
        workspace::ValueExcessWorkspace,
        parameters,
)
    state = FilteredDistinctCompositionState(
        composition, collect(values), workspace, parameters, 0, 0.0)
    incremental_rebuild!(state, values)
    return state
end

function _incremental_state(
        composition::IncrementalComposition{
            Tuple{Val{:count_equal_left}}, Val{:sum}, Val{:count_positive}, Val{:id}},
        values, workspace::ValueExcessWorkspace, parameters)
    state = ValueExcessCompositionState(
        composition,
        collect(values),
        workspace,
        parameters,
        0,
        0.0
    )
    incremental_rebuild!(state, values)
    return state
end

function _incremental_state(
        composition::IncrementalComposition{
            Tuple{Val{:count_equal_right}}, Val{:sum}, Val{:count_positive}, Val{:id}},
        values, workspace::ValueExcessWorkspace, parameters)
    state = ValueExcessCompositionState(
        composition, collect(values), workspace, parameters, 0, 0.0)
    incremental_rebuild!(state, values)
    return state
end

function _incremental_state(
        composition::IncrementalComposition{Tuple{E}, A, Val{:count_zero}, C},
        values, workspace::ValueExcessWorkspace, parameters,
) where {E <: Union{Val{:count_equal_left}, Val{:count_equal_right}}, A, C}
    state = DistinctCountCompositionState(
        composition, collect(values), workspace, parameters, 0, 0.0)
    incremental_rebuild!(state, values)
    return state
end

function _incremental_state(
        composition::IncrementalComposition{Tuple{Val{:id}}, A, Val{:sum}, C},
        values, workspace::SumWorkspace{T}, parameters) where {A, C, T}
    state = SumCompositionState(
        composition,
        collect(values),
        workspace,
        parameters,
        zero(T),
        0.0
    )
    incremental_rebuild!(state, values)
    return state
end

incremental_value(state::AbstractIncrementalCompositionState) = state.current

@inline function _matrix_row_count_penalty(vals, row, count)
    columns = size(vals, 2)
    columns == 1 && return Float64(abs(count - 1))
    target = @inbounds vals[row, 2]
    columns == 2 && return Float64(abs(count - target))
    upper = @inbounds vals[row, 3]
    minimum_count = min(target, upper)
    maximum_count = max(target, upper)
    return Float64(count < minimum_count ? minimum_count - count :
                   count > maximum_count ? count - maximum_count : zero(count))
end

function incremental_rebuild!(state::MatrixRowsCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change",
    ))
    copyto!(state.values, values)
    counts = state.workspace.counts
    fill!(counts, 0)
    outside = 0
    @inbounds for value in state.values
        found = false
        for row in axes(state.vals, 1)
            if value == state.vals[row, 1]
                counts[row] += 1
                found = true
            end
        end
        outside += !found
    end
    current = state.bool ? Float64(outside) : 0.0
    @inbounds for row in eachindex(counts)
        current += _matrix_row_count_penalty(state.vals, row, counts[row])
    end
    state.outside = outside
    state.current = current
    return current
end

@inline function _parameter_rows_state_value(composition, mismatches)
    if composition.reduction === :exists_min
        minimum_distance = typemax(Int)
        @inbounds for distance in mismatches
            distance < 0 && continue
            minimum_distance = min(minimum_distance, distance)
            iszero(minimum_distance) && return 0.0
        end
        return minimum_distance == typemax(Int) ? 1.0 : Float64(minimum_distance)
    elseif composition.reduction === :exists_product
        product = 1.0
        found = false
        @inbounds for distance in mismatches
            distance < 0 && continue
            found = true
            product *= distance
            iszero(product) && return 0.0
        end
        return found ? product : 1.0
    elseif composition.reduction === :forall_sum
        total = 0.0
        @inbounds for distance in mismatches
            distance < 0 || (total += distance)
        end
        return total
    elseif composition.reduction === :forall_max
        maximum_distance = 0
        @inbounds for distance in mismatches
            distance < 0 || (maximum_distance = max(maximum_distance, distance))
        end
        return Float64(maximum_distance)
    end
    matches = count(iszero, mismatches)
    return Float64(_count_violation(
        matches, composition.reduction_op, composition.reduction_val,
    ))
end

function incremental_rebuild!(state::ParameterRowsCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change",
    ))
    copyto!(state.values, values)
    @inbounds for row in eachindex(state.workspace.mismatches)
        state.workspace.mismatches[row] =
            _aligned_mismatch_count(state.values, state.pair_vars, row)
    end
    state.current = _parameter_rows_state_value(
        state.composition, state.workspace.mismatches,
    )
    return state.current
end

function incremental_rebuild!(state::LanguageDistanceCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change",
    ))
    copyto!(state.values, values)
    state.current = Float64(ConstraintCommons.language_distance(
        state.language, state.values, state.workspace,
    ))
    return state.current
end


function incremental_rebuild!(state::PairDistanceCollisionCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change",
    ))
    copyto!(state.values, values)
    state.current = _pair_distance_collision_penalty(state.values)
    return state.current
end

@inline _grouped_state_min(::Tuple{}) = Inf
@inline function _grouped_state_min(states::Tuple)
    value = incremental_value(first(states))
    iszero(value) && return value
    return min(value, _grouped_state_min(Base.tail(states)))
end

@inline _grouped_state_product(::Tuple{}) = 1.0
@inline function _grouped_state_product(states::Tuple)
    value = incremental_value(first(states))
    iszero(value) && return value
    return value * _grouped_state_product(Base.tail(states))
end

@inline _grouped_state_max(::Tuple{}) = -Inf
@inline _grouped_state_max(states::Tuple) = max(
    incremental_value(first(states)), _grouped_state_max(Base.tail(states)),
)

@inline _grouped_state_sum(::Tuple{}) = 0.0
@inline _grouped_state_sum(states::Tuple) =
    incremental_value(first(states)) + _grouped_state_sum(Base.tail(states))

@inline _grouped_state_zero_count(::Tuple{}) = 0
@inline _grouped_state_zero_count(states::Tuple) =
    iszero(incremental_value(first(states))) +
    _grouped_state_zero_count(Base.tail(states))

@inline function _grouped_state_value(composition, states::Tuple)
    reduction = composition.reduction
    isempty(states) && return _empty_reduction(
        reduction, composition.reduction_op, composition.reduction_val,
    )
    if reduction === :exists_min
        return _grouped_state_min(states)
    elseif reduction === :exists_product
        return _grouped_state_product(states)
    elseif reduction === :forall_max
        return _grouped_state_max(states)
    elseif reduction === :forall_sum || reduction === :mean
        result = _grouped_state_sum(states)
        return reduction === :mean ? result / length(states) : result
    elseif reduction === :count
        return _count_violation(
            _grouped_state_zero_count(states),
            composition.reduction_op,
            composition.reduction_val,
        )
    end
    throw(ArgumentError("unknown ICN output reduction: $reduction"))
end

@inline _grouped_rebuild!(::Tuple{}, values) = nothing
@inline function _grouped_rebuild!(states::Tuple, values)
    incremental_rebuild!(first(states), values)
    _grouped_rebuild!(Base.tail(states), values)
    return nothing
end

@inline _grouped_update!(::Tuple{}, position, new_value) = nothing
@inline function _grouped_update!(states::Tuple, position, new_value)
    incremental_update!(first(states), position, new_value)
    _grouped_update!(Base.tail(states), position, new_value)
    return nothing
end

function incremental_rebuild!(state::GroupedCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change",
    ))
    copyto!(state.values, values)
    _grouped_rebuild!(state.states, values)
    state.current = _grouped_state_value(state.composition, state.states)
    return state.current
end

function _functional_graph_penalty!(workspace::FunctionalGraphWorkspace,
        values, op::F, val) where {F}
    counts = workspace.counts
    visited = workspace.visited
    length(counts) == length(values) == length(visited) || throw(DimensionMismatch(
        "functional-graph workspace must match the input length",
    ))
    fill!(counts, 0)
    fill!(visited, false)

    first_active = 0
    active_count = 0
    @inbounds for index in eachindex(values)
        successor = values[index]
        if successor isa Integer && 1 <= successor <= length(values)
            counts[Int(successor)] += 1
        end
        if successor != index
            iszero(first_active) && (first_active = index)
            active_count += 1
        end
    end

    missing_predecessors = 0
    @inbounds for count in counts
        missing_predecessors += iszero(count)
    end
    exclusions = 0
    if iszero(first_active)
        exclusions = !isempty(values)
    else
        current = first_active
        while current isa Integer && 1 <= current <= length(values) &&
              !visited[Int(current)] && values[Int(current)] != current
            visited[Int(current)] = true
            current = values[Int(current)]
        end
        @inbounds for index in eachindex(values)
            exclusions += values[index] != index && !visited[index]
        end
    end
    return Float64(missing_predecessors + exclusions) +
           _condition_residual(active_count, val, op)
end

function incremental_rebuild!(state::FunctionalGraphCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change",
    ))
    values === state.values || copyto!(state.values, values)
    state.current = _functional_graph_penalty!(
        state.workspace,
        state.values,
        state.parameters.op,
        state.parameters.val,
    )
    return state.current
end


function incremental_rebuild!(
        state::IndexRelationCompositionState{<:IndicatorIndexComposition}, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change"))
    copyto!(state.values, values)
    state.current = Float64(state.composition(state.values; state.parameters...))
    return state.current
end


@inline function _cyclic_list_insert!(workspace, block, target, position)
    old_head = @inbounds workspace.heads[block, target]
    @inbounds begin
        workspace.previous[position] = 0
        workspace.next[position] = old_head
        workspace.heads[block, target] = position
        iszero(old_head) || (workspace.previous[old_head] = position)
    end
    return nothing
end


@inline function _cyclic_list_remove!(workspace, block, target, position)
    previous = @inbounds workspace.previous[position]
    next = @inbounds workspace.next[position]
    @inbounds if iszero(previous)
        workspace.heads[block, target] = next
    else
        workspace.next[previous] = next
    end
    @inbounds iszero(next) || (workspace.previous[next] = previous)
    @inbounds begin
        workspace.previous[position] = 0
        workspace.next[position] = 0
    end
    return nothing
end


function incremental_rebuild!(
        state::IndexRelationCompositionState{<:CyclicIndexComposition}, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change"))
    copyto!(state.values, values)
    blocks, width = _block_layout(state.values, Int(state.parameters.dim))
    workspace = state.workspace
    size(workspace.heads) == (blocks, width) || throw(DimensionMismatch(
        "cyclic reverse-index workspace has the wrong block layout"))
    fill!(workspace.heads, 0)
    fill!(workspace.next, 0)
    fill!(workspace.previous, 0)
    @inbounds for position in eachindex(state.values)
        target = state.values[position]
        target isa Integer && 1 ≤ target ≤ width || continue
        block = (position - 1) ÷ width + 1
        _cyclic_list_insert!(workspace, block, Int(target), position)
    end
    state.current = Float64(state.composition(state.values; state.parameters...))
    return state.current
end

function incremental_rebuild!(state::IncrementalCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change"))
    copyto!(state.values, values)
    composition = state.composition
    workspace = state.workspace
    rows = length(values)
    columns = length(composition.transformations)
    _transform_columns!(composition.transformations, workspace.transformations,
        state.values, 1; state.parameters...)
    _initialize_combined!(workspace.combined, composition.arithmetic,
        workspace.transformations, rows, Val(columns))
    state.aggregate = _initial_aggregate(
        composition.aggregation, workspace.combined, rows)
    state.current = Float64(
        _compare(composition.comparison, state.aggregate; state.parameters...))
    return state.current
end

function incremental_rebuild!(state::ValueExcessCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change"))
    copyto!(state.values, values)
    counts = state.workspace.counts
    empty!(counts)
    duplicates = 0
    for value in state.values
        count = get(counts, value, 0)
        duplicates += !iszero(count)
        counts[value] = count + 1
    end
    state.duplicates = duplicates
    state.current = Float64(duplicates)
    return state.current
end

function incremental_rebuild!(state::DistinctCountCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change"))
    copyto!(state.values, values)
    counts = state.workspace.counts
    empty!(counts)
    for value in state.values
        counts[value] = get(counts, value, 0) + 1
    end
    state.distinct = length(counts)
    state.current = Float64(_compare(
        state.composition.comparison, state.distinct; state.parameters...))
    return state.current
end

@inline _filter_value_keep(::Val{:id}, value; parameters...) = true
@inline _filter_value_keep(::Val{:filter_unique}, value; parameters...) = true
@inline _filter_value_keep(::Val{:filter_op_val}, value; val, op, parameters...) =
    op(value, val)
@inline _filter_value_keep(::Val{:filter_equal_val}, value; val, parameters...) = value == val
@inline _filter_value_keep(::Val{:filter_ge_val}, value; val, parameters...) = value >= val
@inline _filter_value_keep(::Val{:filter_great_val}, value; val, parameters...) = value > val
@inline _filter_value_keep(::Val{:filter_less_val}, value; val, parameters...) = value < val
@inline _filter_value_keep(::Val{:filter_le_val}, value; val, parameters...) = value <= val
@inline _filter_value_keep(::Val{:filter_ne_val}, value; val, parameters...) = value != val
@inline _filter_value_keep(
    ::Val{:filter_equal_filter_val}, value; filter_val, parameters...) = value == filter_val
@inline _filter_value_keep(::Val{:filter_equal_vals}, value; vals, parameters...) = value in vals
@inline _filter_value_keep(::Val{:filter_ne_vals}, value; vals, parameters...) = !(value in vals)
@inline function _filter_value_keep(
        ::Val{:filter_op_vals}, value; vals, op, parameters...)
    @inbounds for parameter in vals
        op(value, parameter) || return false
    end
    return true
end

function incremental_rebuild!(state::FilteredDistinctCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change"))
    copyto!(state.values, values)
    counts = state.workspace.counts
    empty!(counts)
    for value in state.values
        _filter_value_keep(state.composition.filter, value; state.parameters...) || continue
        counts[value] = get(counts, value, 0) + 1
    end
    state.distinct = length(counts)
    state.current = Float64(_compare(
        state.composition.comparison, state.distinct; state.parameters...))
    return state.current
end

function incremental_rebuild!(state::SumCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change"))
    copyto!(state.values, values)
    total = zero(typeof(state.aggregate))
    @inbounds for value in state.values
        total += value
    end
    state.aggregate = total
    state.current = Float64(
        _compare(state.composition.comparison, total; state.parameters...))
    return state.current
end

function incremental_rebuild!(state::PairwiseDisjunctionCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change"))
    copyto!(state.values, values)
    haskey(state.parameters, :pair_vars) || throw(ArgumentError(
        "pairwise disjunction state requires pair_vars"))
    paired = _paired_parameters(state.parameters.pair_vars)
    dimensions = Int(get(state.parameters, :dim, 1))
    zero_ignored = Bool(get(state.parameters, :bool, true))
    tasks = _check_pairwise_geometry_arguments(state.values, paired, dimensions)
    residuals = state.workspace.residuals
    required = tasks * (tasks - 1) ÷ 2
    length(residuals) >= required || throw(DimensionMismatch(
        "pairwise disjunction workspace must contain at least $required residuals"))
    aggregate = state.composition.aggregation isa Val{:sum} ? 0.0 : 0
    pair_index = 1
    @inbounds for first_task in 1:(tasks - 1)
        for second_task in (first_task + 1):tasks
            residual = _pairwise_disjunction_residual(
                state.values,
                paired,
                dimensions,
                first_task,
                second_task,
                zero_ignored,
            )
            residuals[pair_index] = residual
            aggregate = _update_aggregate(
                state.composition.aggregation, aggregate, 0.0, residual)
            pair_index += 1
        end
    end
    state.aggregate = aggregate
    state.current = Float64(aggregate)
    return state.current
end

function incremental_rebuild!(state::AlignedPairCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change"))
    pair_vars = state.parameters.pair_vars
    axes(values) == axes(pair_vars) || throw(DimensionMismatch(
        "aligned values and pair_vars must have the same axes"))
    copyto!(state.values, values)
    aggregate = state.composition.aggregation isa Val{:sum} ? 0.0 : 0
    @inbounds for index in eachindex(state.values, pair_vars)
        contribution = _paired_scalar(
            state.composition.operation, state.values[index], pair_vars[index])
        aggregate = _update_aggregate(
            state.composition.aggregation, aggregate, 0, contribution)
    end
    state.aggregate = aggregate
    state.current = Float64(aggregate)
    return state.current
end

function incremental_rebuild!(state::ParameterBindingCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change"))
    copyto!(state.values, values)
    state.current = Float64(state.composition(state.values; state.parameters...))
    return state.current
end

function incremental_rebuild!(state::EventProfileCompositionState, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change",
    ))
    copyto!(state.values, values)
    maximum_load = _maximum_weighted_interval_load!(
        state.workspace.events, state.values, state.parameters.pair_vars,
    )
    state.current = Float64(
        _compare(state.composition.comparison, maximum_load; state.parameters...),
    )
    return state.current
end


function incremental_rebuild!(
        state::EventProfileCompositionState{<:EventAreaComposition}, values)
    length(values) == length(state.values) || throw(DimensionMismatch(
        "incremental state input length cannot change",
    ))
    copyto!(state.values, values)
    state.current = _weighted_interval_condition_area_unchecked!(
        state.workspace.events,
        state.composition.reduction,
        state.values,
        state.parameters.pair_vars,
        state.parameters.op,
        state.parameters.val,
    )
    return state.current
end

@inline function _pairwise_residual_index(tasks::Int, first_task::Int, second_task::Int)
    first_task < second_task || throw(ArgumentError("pair indices must be ordered"))
    return (first_task - 1) * (2 * tasks - first_task) ÷ 2 +
           (second_task - first_task)
end

@inline function _update_transform!(::Val{:id}, transformations, column, values,
        position, old_value, new_value)
    @inbounds transformations[position, column] = new_value
    return position, position
end

@inline function _update_transform!(::Val{:count_equal_left}, transformations, column,
        values, position, old_value, new_value)
    count = 0
    @inbounds for row in firstindex(values):(position - 1)
        count += values[row] == new_value
    end
    @inbounds transformations[position, column] = count
    @inbounds for row in (position + 1):lastindex(values)
        transformations[row, column] += (values[row] == new_value) -
                                        (values[row] == old_value)
    end
    return position, lastindex(values)
end

@inline function _update_transform!(::Val{:count_equal_right}, transformations, column,
        values, position, old_value, new_value)
    count = 0
    @inbounds for row in (position + 1):lastindex(values)
        count += values[row] == new_value
    end
    @inbounds transformations[position, column] = count
    @inbounds for row in firstindex(values):(position - 1)
        transformations[row, column] += (values[row] == new_value) -
                                        (values[row] == old_value)
    end
    return firstindex(values), position
end

@inline _update_transform_columns!(::Tuple{}, transformations, column, values, position,
    old_value, new_value, first_row, last_row) = (first_row, last_row)

@inline function _update_transform_columns!(operations::Tuple, transformations, column,
        values, position, old_value, new_value, first_row, last_row)
    first_changed, last_changed = _update_transform!(first(operations), transformations,
        column, values, position, old_value, new_value)
    return _update_transform_columns!(Base.tail(operations), transformations, column + 1,
        values, position, old_value, new_value,
        min(first_row, first_changed), max(last_row, last_changed))
end

@inline function _update_single_transform!(::Tuple{Val{:id}}, aggregation, workspace,
        values, position, old_value, new_value, aggregate)
    old_combined = @inbounds workspace.combined[position]
    @inbounds workspace.transformations[position, 1] = new_value
    @inbounds workspace.combined[position] = new_value
    return _update_aggregate(aggregation, aggregate, old_combined, new_value)
end

@inline function _update_single_transform!(::Tuple{Val{:count_equal_left}}, aggregation,
        workspace, values, position, old_value, new_value, aggregate)
    count = 0
    @inbounds for row in firstindex(values):(position - 1)
        count += values[row] == new_value
    end
    old_combined = @inbounds workspace.combined[position]
    @inbounds workspace.transformations[position, 1] = count
    @inbounds workspace.combined[position] = count
    aggregate = _update_aggregate(
        aggregation, aggregate, old_combined, count)
    @inbounds for row in (position + 1):lastindex(values)
        old_combined = workspace.combined[row]
        new_combined = old_combined + (values[row] == new_value) -
                       (values[row] == old_value)
        workspace.transformations[row, 1] = new_combined
        workspace.combined[row] = new_combined
        aggregate = _update_aggregate(
            aggregation, aggregate, old_combined, new_combined)
    end
    return aggregate
end

@inline function _update_single_transform!(::Tuple{Val{:count_equal_right}}, aggregation,
        workspace, values, position, old_value, new_value, aggregate)
    count = 0
    @inbounds for row in (position + 1):lastindex(values)
        count += values[row] == new_value
    end
    old_combined = @inbounds workspace.combined[position]
    @inbounds workspace.transformations[position, 1] = count
    @inbounds workspace.combined[position] = count
    aggregate = _update_aggregate(aggregation, aggregate, old_combined, count)
    @inbounds for row in firstindex(values):(position - 1)
        old_combined = workspace.combined[row]
        new_combined = old_combined + (values[row] == new_value) -
                       (values[row] == old_value)
        workspace.transformations[row, 1] = new_combined
        workspace.combined[row] = new_combined
        aggregate = _update_aggregate(
            aggregation, aggregate, old_combined, new_combined)
    end
    return aggregate
end

@inline function _update_composition!(transformations::Tuple{Val{:id}}, composition,
        workspace, values, position, old_value, new_value, aggregate)
    return _update_single_transform!(transformations, composition.aggregation,
        workspace, values, position, old_value, new_value, aggregate)
end

@inline function _update_composition!(
        transformations::Tuple{Val{:count_equal_left}}, composition,
        workspace, values, position, old_value, new_value, aggregate)
    return _update_single_transform!(transformations, composition.aggregation,
        workspace, values, position, old_value, new_value, aggregate)
end

@inline function _update_composition!(
        transformations::Tuple{Val{:count_equal_right}}, composition,
        workspace, values, position, old_value, new_value, aggregate)
    return _update_single_transform!(transformations, composition.aggregation,
        workspace, values, position, old_value, new_value, aggregate)
end

@inline function _update_composition!(transformations::Tuple, composition, workspace,
        values, position, old_value, new_value, aggregate)
    first_row, last_row = _update_transform_columns!(transformations,
        workspace.transformations, 1, values, position, old_value, new_value,
        lastindex(values), firstindex(values))
    columns = length(transformations)
    @inbounds for row in first_row:last_row
        old_combined = workspace.combined[row]
        new_combined = _combined_row(
            composition.arithmetic, workspace.transformations, row, columns)
        workspace.combined[row] = new_combined
        aggregate = _update_aggregate(
            composition.aggregation, aggregate, old_combined, new_combined)
    end
    return aggregate
end

"""Apply one local value replacement to persistent ICN state and return its new cost."""
Base.@propagate_inbounds function incremental_update!(
        state::MatrixRowsCompositionState, position::Integer, new_value)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    counts = state.workspace.counts
    old_found = false
    new_found = false
    current = state.current
    @inbounds for row in axes(state.vals, 1)
        target = state.vals[row, 1]
        if old_value == target
            old_found = true
            old_count = counts[row]
            current -= _matrix_row_count_penalty(state.vals, row, old_count)
            counts[row] = old_count - 1
            current += _matrix_row_count_penalty(state.vals, row, old_count - 1)
        end
        if new_value == target
            new_found = true
            old_count = counts[row]
            current -= _matrix_row_count_penalty(state.vals, row, old_count)
            counts[row] = old_count + 1
            current += _matrix_row_count_penalty(state.vals, row, old_count + 1)
        end
    end
    if !old_found
        state.outside -= 1
        state.bool && (current -= 1.0)
    end
    if !new_found
        state.outside += 1
        state.bool && (current += 1.0)
    end
    @inbounds values[position] = new_value
    state.current = current
    return current
end

Base.@propagate_inbounds function incremental_update!(
        state::ParameterRowsCompositionState, position::Integer, new_value)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    @inbounds for row in eachindex(state.workspace.mismatches)
        mismatch = state.workspace.mismatches[row]
        mismatch < 0 && continue
        row_value = _parameter_row_value(state.pair_vars, row, Int(position))
        state.workspace.mismatches[row] = mismatch +
                                          (new_value != row_value) -
                                          (old_value != row_value)
    end
    @inbounds values[position] = new_value
    state.current = _parameter_rows_state_value(
        state.composition, state.workspace.mismatches,
    )
    return state.current
end

Base.@propagate_inbounds function incremental_update!(
        state::LanguageDistanceCompositionState, position::Integer, new_value)
    @boundscheck checkbounds(state.values, position)
    @inbounds state.values[position] = new_value
    state.current = Float64(ConstraintCommons.language_distance(
        state.language, state.values, state.workspace,
    ))
    return state.current
end


Base.@propagate_inbounds function incremental_update!(
        state::PairDistanceCollisionCompositionState,
        position::Integer,
        new_value,
)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    pair_count = length(values) ÷ 2
    if position > 2pair_count
        @inbounds values[position] = new_value
        return state.current
    end
    pair_index = (Int(position) + 1) ÷ 2
    left = 2pair_index - 1
    right = left + 1
    old_distance = @inbounds abs(values[left] - values[right])
    new_distance = @inbounds position == left ?
                   abs(new_value - values[right]) : abs(values[left] - new_value)
    if old_distance != new_distance
        old_other_count = 0
        new_other_count = 0
        @inbounds for other_pair in 1:pair_count
            other_pair == pair_index && continue
            other_left = 2other_pair - 1
            distance = abs(values[other_left] - values[other_left + 1])
            old_other_count += distance == old_distance
            new_other_count += distance == new_distance
        end
        state.current += (new_other_count > 0) - (old_other_count > 0)
    end
    @inbounds values[position] = new_value
    return state.current
end

Base.@propagate_inbounds function incremental_update!(
        state::GroupedCompositionState, position::Integer, new_value)
    @boundscheck checkbounds(state.values, position)
    _grouped_update!(state.states, position, new_value)
    @inbounds state.values[position] = new_value
    state.current = _grouped_state_value(state.composition, state.states)
    return state.current
end

Base.@propagate_inbounds function incremental_update!(
        state::AlignedPairCompositionState, position::Integer, new_value)
    values = state.values
    pair_vars = state.parameters.pair_vars
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    pair_value = @inbounds pair_vars[position]
    old_contribution = _paired_scalar(
        state.composition.operation, old_value, pair_value)
    new_contribution = _paired_scalar(
        state.composition.operation, new_value, pair_value)
    aggregate = _update_aggregate(
        state.composition.aggregation, state.aggregate,
        old_contribution, new_contribution)
    @inbounds values[position] = new_value
    state.aggregate = aggregate
    state.current = Float64(aggregate)
    return state.current
end


Base.@propagate_inbounds function incremental_update!(
        state::IndexRelationCompositionState{<:IndicatorIndexComposition},
        position::Integer,
        new_value,
)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    id = state.parameters.id
    target = position == id
    state.current += abs(Float64(new_value) - target) -
                     abs(Float64(old_value) - target)
    @inbounds values[position] = new_value
    return state.current
end

Base.@propagate_inbounds function incremental_update!(
        state::FunctionalGraphCompositionState,
        position::Integer,
        new_value,
)
    @boundscheck checkbounds(state.values, position)
    old_value = @inbounds state.values[position]
    old_value == new_value && return state.current
    @inbounds state.values[position] = new_value
    return incremental_rebuild!(state, state.values)
end


@inline function _cyclic_index_contribution(
        values,
        blocks::Int,
        width::Int,
        source_position::Int,
        changed_position::Int,
        changed_value,
)
    source_block = (source_position - 1) ÷ width
    local_index = (source_position - 1) % width + 1
    source_value = source_position == changed_position ? changed_value :
                   (@inbounds values[source_position])
    source_value isa Integer && 1 ≤ source_value ≤ width ||
        return Float64(local_index)
    target_block = (source_block + 1) % blocks
    target_position = target_block * width + Int(source_value)
    target_value = target_position == changed_position ? changed_value :
                   (@inbounds values[target_position])
    return abs(Float64(target_value - local_index))
end


Base.@propagate_inbounds function incremental_update!(
        state::IndexRelationCompositionState{<:CyclicIndexComposition},
        position::Integer,
        new_value,
)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    blocks, width = _block_layout(values, Int(state.parameters.dim))
    changed_position = Int(position)
    current = state.current

    old_contribution = _cyclic_index_contribution(
        values, blocks, width, changed_position, 0, old_value)
    new_contribution = _cyclic_index_contribution(
        values, blocks, width, changed_position, changed_position, new_value)
    current += new_contribution - old_contribution

    changed_block = (changed_position - 1) ÷ width
    changed_local_index = (changed_position - 1) % width + 1
    previous_block = (changed_block - 1 + blocks) % blocks
    workspace = state.workspace
    source_position = @inbounds workspace.heads[previous_block + 1, changed_local_index]
    while !iszero(source_position)
        next_source = @inbounds workspace.next[source_position]
        if source_position == changed_position
            source_position = next_source
            continue
        end
        old_contribution = _cyclic_index_contribution(
            values, blocks, width, source_position, 0, old_value)
        new_contribution = _cyclic_index_contribution(
            values, blocks, width, source_position, changed_position, new_value)
        current += new_contribution - old_contribution
        source_position = next_source
    end

    source_block = changed_block + 1
    if old_value isa Integer && 1 ≤ old_value ≤ width
        _cyclic_list_remove!(workspace, source_block, Int(old_value), changed_position)
    end
    if new_value isa Integer && 1 ≤ new_value ≤ width
        _cyclic_list_insert!(workspace, source_block, Int(new_value), changed_position)
    end
    @inbounds values[changed_position] = new_value
    state.current = current
    return current
end

Base.@propagate_inbounds function incremental_update!(
        state::ParameterBindingCompositionState, position::Integer, new_value)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    @inbounds values[position] = new_value
    state.current = Float64(state.composition(values; state.parameters...))
    return state.current
end

Base.@propagate_inbounds function incremental_update!(
        state::IncrementalCompositionState, position::Integer,
        new_value)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    composition = state.composition
    workspace = state.workspace
    aggregate = _update_composition!(composition.transformations, composition,
        workspace, values, position, old_value, new_value, state.aggregate)
    @inbounds values[position] = new_value
    state.aggregate = aggregate
    state.current = Float64(
        _compare(composition.comparison, aggregate; state.parameters...))
    return state.current
end

Base.@propagate_inbounds function incremental_update!(
        state::ValueExcessCompositionState, position::Integer,
        new_value)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    counts = state.workspace.counts
    old_count = counts[old_value]
    state.duplicates -= old_count > 1
    if old_count == 1
        delete!(counts, old_value)
    else
        counts[old_value] = old_count - 1
    end
    new_count = get(counts, new_value, 0)
    state.duplicates += !iszero(new_count)
    counts[new_value] = new_count + 1
    @inbounds values[position] = new_value
    state.current = Float64(state.duplicates)
    return state.current
end

Base.@propagate_inbounds function incremental_update!(
        state::DistinctCountCompositionState, position::Integer, new_value)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    counts = state.workspace.counts
    old_count = counts[old_value]
    if old_count == 1
        delete!(counts, old_value)
        state.distinct -= 1
    else
        counts[old_value] = old_count - 1
    end
    new_count = get(counts, new_value, 0)
    state.distinct += iszero(new_count)
    counts[new_value] = new_count + 1
    @inbounds values[position] = new_value
    state.current = Float64(_compare(
        state.composition.comparison, state.distinct; state.parameters...))
    return state.current
end

Base.@propagate_inbounds function incremental_update!(
        state::FilteredDistinctCompositionState, position::Integer, new_value)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    counts = state.workspace.counts
    if _filter_value_keep(state.composition.filter, old_value; state.parameters...)
        old_count = counts[old_value]
        if old_count == 1
            delete!(counts, old_value)
            state.distinct -= 1
        else
            counts[old_value] = old_count - 1
        end
    end
    if _filter_value_keep(state.composition.filter, new_value; state.parameters...)
        new_count = get(counts, new_value, 0)
        state.distinct += iszero(new_count)
        counts[new_value] = new_count + 1
    end
    @inbounds values[position] = new_value
    state.current = Float64(_compare(
        state.composition.comparison, state.distinct; state.parameters...))
    return state.current
end

Base.@propagate_inbounds function incremental_update!(
        state::SumCompositionState, position::Integer, new_value)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    state.aggregate += new_value - old_value
    @inbounds values[position] = new_value
    state.current = Float64(_compare(
        state.composition.comparison, state.aggregate; state.parameters...))
    return state.current
end


Base.@propagate_inbounds function incremental_update!(
        state::PairwiseDisjunctionCompositionState,
        position::Integer,
        new_value,
)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    paired = state.parameters.pair_vars
    dimensions = Int(get(state.parameters, :dim, 1))
    zero_ignored = Bool(get(state.parameters, :bool, true))
    tasks = length(values) ÷ dimensions
    changed_task = (Int(position) - 1) ÷ dimensions + 1
    residuals = state.workspace.residuals
    aggregate = state.aggregate
    @inbounds values[position] = new_value
    @inbounds for other_task in 1:tasks
        other_task == changed_task && continue
        first_task = min(changed_task, other_task)
        second_task = max(changed_task, other_task)
        pair_index = _pairwise_residual_index(tasks, first_task, second_task)
        old_residual = residuals[pair_index]
        new_residual = _pairwise_disjunction_residual(
            values,
            paired,
            dimensions,
            first_task,
            second_task,
            zero_ignored,
        )
        residuals[pair_index] = new_residual
        aggregate = _update_aggregate(
            state.composition.aggregation, aggregate, old_residual, new_residual)
    end
    state.aggregate = aggregate
    state.current = Float64(aggregate)
    return state.current
end

Base.@propagate_inbounds function incremental_update!(
        state::EventProfileCompositionState,
        position::Integer,
        new_value,
)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    @inbounds values[position] = new_value
    maximum_load = _maximum_weighted_interval_load!(
        state.workspace.events, values, state.parameters.pair_vars,
    )
    state.current = Float64(
        _compare(state.composition.comparison, maximum_load; state.parameters...),
    )
    return state.current
end


Base.@propagate_inbounds function incremental_update!(
        state::EventProfileCompositionState{<:EventAreaComposition},
        position::Integer,
        new_value,
)
    values = state.values
    @boundscheck checkbounds(values, position)
    old_value = @inbounds values[position]
    old_value == new_value && return state.current
    @inbounds values[position] = new_value
    state.current = _weighted_interval_condition_area_unchecked!(
        state.workspace.events,
        state.composition.reduction,
        values,
        state.parameters.pair_vars,
        state.parameters.op,
        state.parameters.val,
    )
    return state.current
end

"""
Evaluate a batch of temporary replacements and restore the incremental state.

The generic transaction restores derived state by applying the inverse replacements. Fully
recomputed event profiles specialize this path: their caller-owned event buffer may remain
stale, so only the values and cached cost need restoring and the original profile is not
sorted a second time.
"""
function incremental_candidate_value!(
        state::AbstractIncrementalCompositionState,
        changes::Union{Tuple, AbstractVector},
)
    applied = 0
    try
        for change in changes
            incremental_update!(state, change.position, change.new_value)
            applied += 1
        end
        return incremental_value(state)
    finally
        @inbounds for index in applied:-1:1
            change = changes[index]
            incremental_update!(state, change.position, change.old_value)
        end
    end
end

function incremental_candidate_value!(
        state::EventProfileCompositionState,
        changes::Union{Tuple, AbstractVector},
)
    isempty(changes) && return state.current
    old_current = state.current
    applied = 0
    try
        for change in changes
            @boundscheck checkbounds(state.values, change.position)
            @inbounds state.values[change.position] = change.new_value
            applied += 1
        end
        return incremental_rebuild!(state, state.values)
    finally
        @inbounds for index in applied:-1:1
            change = changes[index]
            state.values[change.position] = change.old_value
        end
        state.current = old_current
    end
end

@testitem "Incremental primitive compositions" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @inferred, @test, @test_throws

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

    network = select!(CN.ICN(), (
        (:count_equal_left,), (:sum,), (:count_positive,), (:id,)))
    @test CN.incremental_supported(network)
    composition = CN.incremental_composition(network)
    workspace = CN.incremental_workspace(composition, [1, 2, 3, 4])
    state = CN.incremental_state(composition, [1, 2, 3, 4]; workspace)
    @test state.workspace === workspace
    @test state isa CN.ValueExcessCompositionState
    @test CN.incremental_value(state) == composition([1, 2, 3, 4]) == 0.0
    @test_throws BoundsError CN.incremental_update!(state, 0, 1)

    @test @inferred(CN.incremental_update!(state, 4, 2)) == 1.0
    @test CN.incremental_value(state) == composition([1, 2, 3, 2])
    @test CN.incremental_update!(state, 4, 4) == 0.0
    temporary_duplicate = ((position = 4, old_value = 4, new_value = 2),)
    @test CN.incremental_candidate_value!(state, temporary_duplicate) == 1.0
    @test CN.incremental_value(state) == 0.0
    @test state.values == [1, 2, 3, 4]

    CN.incremental_update!(state, 1, 2)
    @test CN.incremental_update!(state, 2, 1) == composition([2, 1, 3, 4]) == 0.0
    CN.incremental_update!(state, 2, 2)
    @test CN.incremental_update!(state, 1, 1) == 0.0

    strings = CN.incremental_state(composition, ["a", "b", "a"])
    @test CN.incremental_value(strings) == 1.0
    @test CN.incremental_update!(strings, 3, "c") == 0.0
    @test CN.incremental_rebuild!(strings, ["z", "z", "z"]) == 2.0

    distinct_network = select!(CN.ICN(
        parameters = [:op, :val],
        parameter_values = (; op = (==), val = 3),
    ), ((:count_equal_right,), (:sum,), (:count_zero,), (:condition_residual,)))
    @test CN.incremental_supported(distinct_network)
    distinct_composition = CN.incremental_composition(distinct_network)
    distinct_state = CN.incremental_state(
        distinct_composition, [1, 2, 2, 3]; op = (==), val = 3)
    @test distinct_state isa CN.DistinctCountCompositionState
    @test CN.incremental_value(distinct_state) == 0.0
    @test CN.incremental_update!(distinct_state, 4, 2) == 1.0
    @test CN.incremental_update!(distinct_state, 4, 3) == 0.0
    distinct_candidate = ((position = 1, old_value = 1, new_value = 2),)
    @test CN.incremental_candidate_value!(distinct_state, distinct_candidate) == 1.0
    @test CN.incremental_value(distinct_state) == 0.0
    @test distinct_state.values == [1, 2, 2, 3]

    filtered_layers = [CN.SimpleFilter, CN.Transformation, CN.Arithmetic,
        CN.Aggregation, CN.Comparison]
    filtered_network = select!(CN.ICN(;
        parameters = [:vals, :op, :val],
        parameter_values = (; vals = [0], op = (==), val = 2),
        layers = filtered_layers,
        connection = UInt32.(eachindex(filtered_layers)),
    ), ((:filter_ne_vals,), (:count_equal_right,), (:sum,),
        (:count_zero,), (:condition_residual,)))
    @test CN.check_weights_validity(filtered_network, filtered_network.weights)
    @test CN.incremental_supported(filtered_network)
    filtered_composition = CN.incremental_composition(filtered_network)
    filtered_state = CN.incremental_state(
        filtered_composition, [0, 1, 1, 2]; vals = [0], op = (==), val = 2)
    @test filtered_state isa CN.FilteredDistinctCompositionState
    @test CN.incremental_value(filtered_state) == 0.0
    @test CN.incremental_update!(filtered_state, 4, 1) == 1.0
    @test CN.incremental_update!(filtered_state, 4, 2) == 0.0
    filtered_candidate = ((position = 4, old_value = 2, new_value = 0),)
    @test CN.incremental_candidate_value!(filtered_state, filtered_candidate) == 1.0
    @test CN.incremental_value(filtered_state) == 0.0
    @test filtered_state.values == [0, 1, 1, 2]

    paired_layers = [CN.PairedMap, CN.Transformation, CN.Arithmetic,
        CN.Aggregation, CN.Comparison]
    hamming_network = select!(CN.ICN(;
        parameters = [:pair_vars],
        parameter_values = (; pair_vars = ["a", "b", "c"]),
        layers = paired_layers,
        connection = UInt32.(eachindex(paired_layers)),
    ), ((:aligned_not_equal,), (:id,), (:sum,), (:sum,), (:id,)))
    @test CN.incremental_supported(hamming_network)
    hamming = CN.incremental_composition(hamming_network)
    hamming_state = CN.incremental_state(
        hamming, ["a", "z", "c"]; pair_vars = ["a", "b", "c"])
    @test hamming(["a", "z", "c"]; pair_vars = ["a", "b", "c"]) == 1.0
    @test CN.incremental_update!(hamming_state, 2, "b") == 0.0
    hamming_candidate = ((position = 1, old_value = "a", new_value = "z"),)
    @test CN.incremental_candidate_value!(hamming_state, hamming_candidate) == 1.0
    @test CN.incremental_value(hamming_state) == 0.0
    @test hamming_state.values == ["a", "b", "c"]

    manhattan_network = select!(CN.ICN(;
        parameters = [:pair_vars],
        parameter_values = (; pair_vars = [1, 2, 3]),
        layers = paired_layers,
        connection = UInt32.(eachindex(paired_layers)),
    ), ((:aligned_difference,), (:absolute,), (:sum,), (:sum,), (:id,)))
    @test CN.incremental_supported(manhattan_network)
    manhattan = CN.incremental_composition(manhattan_network)
    manhattan_state = CN.incremental_state(
        manhattan, [4, 2, -1]; pair_vars = [1, 2, 3])
    @test manhattan([4, 2, -1]; pair_vars = [1, 2, 3]) == 7.0
    @test CN.incremental_update!(manhattan_state, 1, 1) == 4.0
    @test_throws DimensionMismatch CN.incremental_state(
        manhattan, [1, 2]; pair_vars = [1])

    function aligned_update_restore_allocations(state, position, replacement)
        old = state.values[position]
        CN.incremental_update!(state, position, replacement)
        CN.incremental_update!(state, position, old)
        return @allocated begin
            CN.incremental_update!(state, position, replacement)
            CN.incremental_update!(state, position, old)
        end
    end
    @test aligned_update_restore_allocations(hamming_state, 1, "z") == 0
    @test aligned_update_restore_allocations(manhattan_state, 2, 9) == 0

    identity_network = select!(CN.ICN(), (
        (:id,), (:product,), (:sum,), (:id,)))
    identity_composition = CN.incremental_composition(identity_network)
    identity_workspace = CN.incremental_workspace(identity_composition, 4)
    identity_state = CN.incremental_state(
        identity_composition, [1, 2, 3, 4]; workspace = identity_workspace)
    @test_throws BoundsError CN.incremental_update!(identity_state, 5, 1)
    @test CN.incremental_update!(identity_state, 2, 7) ==
          identity_composition([1, 7, 3, 4]) == 15.0
    @test identity_state.workspace.combined == [1, 7, 3, 4]

    multiple_network = select!(CN.ICN(), (
        (:id, :count_equal_left), (:sum,), (:sum,), (:id,)))
    multiple_composition = CN.incremental_composition(multiple_network)
    multiple_values = [1, 2, 1, 3]
    multiple_state = CN.incremental_state(multiple_composition, multiple_values)
    for (position, new_value) in ((2, 1), (4, 2), (1, 3), (3, 4))
        multiple_values[position] = new_value
        @test CN.incremental_update!(multiple_state, position, new_value) ==
              multiple_composition(multiple_values)
    end

    function update_and_restore!(state)
        old = state.values[4]
        value = CN.incremental_update!(state, 4, 2)
        CN.incremental_update!(state, 4, old)
        return value
    end
    update_and_restore!(state)
    @test @allocated(update_and_restore!(state)) == 0

    matrix_workspace = CN.incremental_workspace(composition, 4)
    matrix_state = CN.incremental_state(
        composition, [1, 2, 3, 4]; workspace = matrix_workspace)
    @test matrix_state isa CN.IncrementalCompositionState
    @test CN.incremental_update!(matrix_state, 4, 2) == 1.0

    sum_network = select!(CN.ICN(parameters = [:val]), (
        (:id,), (:sum,), (:sum,), (:abs_val,)))
    sum_composition = CN.incremental_composition(sum_network)
    sum_workspace = CN.incremental_workspace(sum_composition, [1, 2, 3, 4])
    sum_state = CN.incremental_state(
        sum_composition, [1, 2, 3, 4]; workspace = sum_workspace, val = 11)
    @test sum_state isa CN.SumCompositionState
    @test sum_state.workspace === sum_workspace
    @test CN.incremental_value(sum_state) == sum_composition([1, 2, 3, 4]; val = 11) == 1.0
    @test_throws BoundsError CN.incremental_update!(sum_state, 5, 1)
    @test @inferred(CN.incremental_update!(sum_state, 1, 2)) == 0.0
    @test CN.incremental_update!(sum_state, 1, 1) == 1.0

    for assignment in Iterators.product(ntuple(_ -> 0:3, 4)...)
        values = collect(assignment)
        trial = CN.incremental_state(sum_composition, values; val = 5)
        @test CN.incremental_value(trial) == sum_composition(values; val = 5)
        for position in eachindex(values), replacement in 0:3

            expected = copy(values)
            expected[position] = replacement
            @test CN.incremental_update!(trial, position, replacement) ==
                  sum_composition(expected; val = 5)
            @test CN.incremental_update!(trial, position, values[position]) ==
                  sum_composition(values; val = 5)
        end
    end

    sum_matrix_workspace = CN.incremental_workspace(sum_composition, 4)
    sum_matrix_state = CN.incremental_state(
        sum_composition, [1, 2, 3, 4]; workspace = sum_matrix_workspace, val = 11)
    @test sum_matrix_state isa CN.IncrementalCompositionState
    @test CN.incremental_update!(sum_matrix_state, 1, 2) == 0.0

    pairwise = select!(CN.ICN(
        parameters = [:dom_size, :numvars, :pair_vars, :dim, :bool],
        layers = [CN.PairedMap, CN.PairMask, CN.GroupReduction,
            CN.Transformation, CN.Arithmetic, CN.Aggregation, CN.Comparison],
        connection = UInt32.(1:7),
    ), (
        (:pairwise_oriented_affine_margins,), (:zero_extent_groups,), (:minimum,),
        (:positive_part,), (:sum,), (:sum,), (:id,),
    ))
    @test CN.incremental_supported(pairwise)
    pairwise_composition = CN.incremental_composition(pairwise)
    pairwise_values = [0, 1, 4]
    pairwise_parameters = (;
        pair_vars = [2, 2, 1], dim = 1, bool = true, numvars = 3, dom_size = 5)
    pairwise_state = CN.incremental_state(
        pairwise_composition, pairwise_values; pairwise_parameters...)
    @test pairwise_state isa CN.PairwiseDisjunctionCompositionState
    @test CN.incremental_value(pairwise_state) ==
          pairwise_composition(pairwise_values; pairwise_parameters...)
    for position in eachindex(pairwise_values), replacement in 0:5
        expected = copy(pairwise_values)
        expected[position] = replacement
        @test CN.incremental_update!(pairwise_state, position, replacement) ==
              pairwise_composition(expected; pairwise_parameters...)
        @test CN.incremental_update!(
            pairwise_state, position, pairwise_values[position]) ==
              pairwise_composition(pairwise_values; pairwise_parameters...)
    end

    box_values = [0, 0, 1, 3, 4, 0]
    box_parameters = (;
        pair_vars = [2, 2, 2, 2, 0, 3], dim = 2, bool = true,
        numvars = 3, dom_size = 5,
    )
    box_state = CN.incremental_state(
        pairwise_composition, box_values; box_parameters...)
    for position in eachindex(box_values), replacement in 0:4
        expected = copy(box_values)
        expected[position] = replacement
        @test CN.incremental_update!(box_state, position, replacement) ==
              pairwise_composition(expected; box_parameters...)
        @test CN.incremental_update!(box_state, position, box_values[position]) ==
              pairwise_composition(box_values; box_parameters...)
    end

    function update_and_restore_pairwise!(state)
        old = state.values[2]
        CN.incremental_update!(state, 2, old + 1)
        CN.incremental_update!(state, 2, old)
        return nothing
    end
    update_and_restore_pairwise!(pairwise_state)
    @test @allocated(update_and_restore_pairwise!(pairwise_state)) == 0

    event_network = select!(CN.ICN(
        parameters = [:dom_size, :numvars, :pair_vars, :op, :val],
        layers = [CN.EventMap, CN.SegmentMap, CN.Arithmetic,
            CN.Aggregation, CN.Comparison],
        connection = UInt32.(1:5),
    ), (
        (:weighted_interval_segments,), (:loads,), (:sum,), (:maximum,),
        (:var_minus_val,),
    ))
    @test CN.incremental_supported(event_network)
    event_composition = CN.incremental_composition(event_network)
    event_parameters = (;
        pair_vars = [2 2 1; 2 1 2], op = (<=), val = 3,
        numvars = 3, dom_size = 5,
    )
    event_values = [0, 1, 3]
    event_workspace = CN.incremental_workspace(
        event_composition, event_values; event_parameters...,
    )
    event_state = CN.incremental_state(
        event_composition, event_values; workspace = event_workspace,
        event_parameters...,
    )
    @test event_state isa CN.EventProfileCompositionState
    @test CN.incremental_value(event_state) ==
          event_composition(event_values; event_parameters...) == 0.0
    @test event_composition(event_values; X = zeros(2, 2), event_parameters...) == 0.0
    for assignment in Iterators.product(ntuple(_ -> 0:3, 3)...)
        values = collect(assignment)
        expected = max(0.0, maximum(getproperty.(
            CN.EventMap.fn[:weighted_interval_segments](
                values; pair_vars = event_parameters.pair_vars,
            ), :load,
        )) - event_parameters.val)
        @test CN.incremental_rebuild!(event_state, values) == expected
        for position in eachindex(values), replacement in 0:3
            candidate = copy(values)
            candidate[position] = replacement
            candidate_expected = max(0.0, maximum(getproperty.(
                CN.EventMap.fn[:weighted_interval_segments](
                    candidate; pair_vars = event_parameters.pair_vars,
                ), :load,
            )) - event_parameters.val)
            change = ((position = position, old_value = values[position],
                new_value = replacement),)
            @test CN.incremental_candidate_value!(event_state, change) ==
                  candidate_expected
            @test CN.incremental_value(event_state) == expected
            @test event_state.values == values
            @test CN.incremental_update!(event_state, position, replacement) ==
                  candidate_expected
            @test CN.incremental_update!(event_state, position, values[position]) == expected
        end
    end

    function update_and_restore_event!(state)
        old = state.values[2]
        CN.incremental_update!(state, 2, old + 1)
        CN.incremental_update!(state, 2, old)
        return nothing
    end
    function evaluate_temporary_event!(state, changes)
        CN.incremental_candidate_value!(state, changes)
        return nothing
    end
    CN.incremental_rebuild!(event_state, event_values)
    event_candidate = ((position = 2, old_value = event_values[2], new_value = 0),)
    evaluate_temporary_event!(event_state, event_candidate)
    @test @allocated(evaluate_temporary_event!(event_state, event_candidate)) == 0
    update_and_restore_event!(event_state)
    @test @allocated(update_and_restore_event!(event_state)) == 0

    compiled_event = first(CN.compose(event_network; name = :weighted_event_test))
    compiled_workspace = CN.composition_workspace(event_network, length(event_values))
    @test compiled_workspace isa CN.EventProfileWorkspace
    @test compiled_event(event_values; X = compiled_workspace, event_parameters...) == 0.0
    function compiled_event_allocations(compiled, values, workspace, parameters)
        compiled(values; X = workspace, parameters...)
        return @allocated compiled(values; X = workspace, parameters...)
    end
    @test compiled_event_allocations(
        compiled_event, event_values, compiled_workspace, event_parameters) == 0

    function event_area_expected(values, parameters)
        segments = CN.EventMap.fn[:weighted_interval_segments](
            values; pair_vars = parameters.pair_vars)
        area = 0.0
        for segment in segments
            residual = max(0.0, segment.load - parameters.val)
            area += segment.width * residual
        end
        return area
    end
    function update_and_restore_event_area!(state)
        old = state.values[2]
        CN.incremental_update!(state, 2, old + 1)
        CN.incremental_update!(state, 2, old)
        return nothing
    end
    area_network = select!(CN.ICN(
        parameters = [:dom_size, :numvars, :pair_vars, :op, :val],
        layers = [CN.EventMap, CN.SegmentMap, CN.Arithmetic,
            CN.Aggregation, CN.Comparison],
        connection = UInt32.(1:5),
    ), (
        (:weighted_interval_segments,), (:widths, :condition_residuals),
        (:product,), (:sum,), (:id,),
    ))
        @test CN.incremental_supported(area_network)
        area_composition = CN.incremental_composition(area_network)
        area_workspace = CN.incremental_workspace(
            area_composition, event_values; event_parameters...)
        area_state = CN.incremental_state(
            area_composition, event_values; workspace = area_workspace,
            event_parameters...)
        @test area_state isa CN.EventProfileCompositionState
        @test CN.incremental_value(area_state) ==
              event_area_expected(event_values, event_parameters)
        @test area_composition(
            event_values; X = zeros(2, 2), event_parameters...) ==
              event_area_expected(event_values, event_parameters)

        for assignment in Iterators.product(ntuple(_ -> 0:3, 3)...)
            values = collect(assignment)
            expected = event_area_expected(values, event_parameters)
            @test CN.incremental_rebuild!(area_state, values) == expected
            for position in eachindex(values), replacement in 0:3
                candidate = copy(values)
                candidate[position] = replacement
                candidate_expected = event_area_expected(candidate, event_parameters)
                change = ((position = position, old_value = values[position],
                    new_value = replacement),)
                @test CN.incremental_candidate_value!(area_state, change) ==
                      candidate_expected
                @test CN.incremental_value(area_state) == expected
                @test area_state.values == values
                @test CN.incremental_update!(area_state, position, replacement) ==
                      candidate_expected
                @test CN.incremental_update!(area_state, position, values[position]) == expected
            end
        end

        CN.incremental_rebuild!(area_state, event_values)
        area_candidate = ((
            position = 2, old_value = event_values[2], new_value = 0),)
        evaluate_temporary_event!(area_state, area_candidate)
        @test @allocated(evaluate_temporary_event!(area_state, area_candidate)) == 0
        update_and_restore_event_area!(area_state)
        @test @allocated(update_and_restore_event_area!(area_state)) == 0

        compiled_area = first(CN.compose(
            area_network; name = :weighted_event_linear_area_test))
        compiled_area_workspace = CN.composition_workspace(
            area_network, length(event_values))
        @test compiled_area_workspace isa CN.EventProfileWorkspace
        @test Base.invokelatest(compiled_area, event_values;
            X = compiled_area_workspace, event_parameters...) ==
            event_area_expected(event_values, event_parameters)

    unsupported = select!(CN.ICN(), (
        (:count_equal,), (:sum,), (:count_positive,), (:id,)))
    @test !CN.incremental_supported(unsupported)
    @test_throws ArgumentError CN.incremental_composition(unsupported)
end
