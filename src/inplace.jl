"Maximum number of reusable columns required by the built-in ICN topology."
max_icn_length() = length(Transformation.fn)
function max_icn_length(icn::AbstractICN)
    maximum(
        (layer.mutex ? 1 : length(layer.fn) for layer in icn.layers); init = 1)
end

"Allocate the reusable matrix consumed by an in-place compiled composition."
function composition_workspace(icn::AbstractICN, input_length::Integer;
        type::Type = Float64)
    if length(icn.layers) == 5 && first(icn.layers).name === :EventMap
        return EventProfileWorkspace(
            Vector{Tuple{type, type}}(undef, 2input_length),
        )
    end
    return Matrix{type}(undef, input_length, max_icn_length(icn))
end

function composition_workspace(columns::Integer, x::AbstractVector)
    Matrix{Float64}(undef, length(x), columns)
end

struct _WorkspaceColumn{T, M <: AbstractMatrix{T}} <: AbstractVector{T}
    workspace::M
    column::Int
    rows::Int
end

Base.IndexStyle(::Type{<:_WorkspaceColumn}) = IndexLinear()
Base.size(column::_WorkspaceColumn) = (column.rows,)
@inline Base.getindex(column::_WorkspaceColumn, index::Int) =
    @inbounds column.workspace[index, column.column]
@inline function Base.setindex!(column::_WorkspaceColumn, value, index::Int)
    @inbounds column.workspace[index, column.column] = value
    return value
end

@inline function _count_relation(predicate, x, value, first_index, last_index)
    count = 0
    @inbounds for j in first_index:last_index
        count += predicate(x[j], value)
    end
    return count
end

@inline function _count_relation_except(predicate, x, value, excluded_index)
    count = 0
    @inbounds for j in eachindex(x)
        j == excluded_index && continue
        count += predicate(x[j], value)
    end
    return count
end

@inline function _transform!(::Val{:id}, output, x; parameters...)
    copyto!(output, x)
    return output
end

@inline function _transform!(::Val{:absolute}, output, x; parameters...)
    @inbounds for i in eachindex(x, output)
        output[i] = abs(x[i])
    end
    return output
end

@inline function _transform!(::Val{:positive_part}, output, x; parameters...)
    @inbounds for i in eachindex(x, output)
        output[i] = max(zero(x[i]), x[i])
    end
    return output
end

@inline function _transform!(::Val{:index_indicator}, output, x; id, parameters...)
    @inbounds for i in eachindex(x, output)
        output[i] = i == id
    end
    return output
end

function _transform!(::Val{:cyclic_indirect_values}, output, x; dim, parameters...)
    blocks, width = _block_layout(x, Int(dim))
    fill!(output, zero(eltype(output)))
    @inbounds for block in 0:(blocks - 1)
        source_offset = block * width
        target_offset = ((block + 1) % blocks) * width
        for local_index in 1:width
            indirect_index = x[source_offset + local_index]
            if indirect_index isa Integer && 1 ≤ indirect_index ≤ width
                output[source_offset + local_index] =
                    x[target_offset + Int(indirect_index)]
            end
        end
    end
    return output
end

function _transform!(::Val{:block_local_indices}, output, x; dim, parameters...)
    blocks, width = _block_layout(x, Int(dim))
    @inbounds for block in 0:(blocks - 1), local_index in 1:width
        output[block * width + local_index] = local_index
    end
    return output
end

function _transform!(::Val{:predecessor_counts}, output, x; parameters...)
    fill!(output, zero(eltype(output)))
    @inbounds for successor in x
        successor isa Integer && 1 <= successor <= length(x) || continue
        output[Int(successor)] += 1
    end
    return output
end

function _transform!(::Val{:orbit_exclusion_indicators}, output, x; parameters...)
    fill!(output, zero(eltype(output)))
    first_active = findfirst(index -> x[index] != index, eachindex(x))
    if isnothing(first_active)
        isempty(output) || (output[firstindex(output)] = 1)
        return output
    end

    current = first_active
    while current isa Integer && 1 <= current <= length(x) &&
          iszero(output[Int(current)]) && x[Int(current)] != current
        output[Int(current)] = 1
        current = x[Int(current)]
    end
    @inbounds for index in eachindex(x, output)
        output[index] = x[index] != index && iszero(output[index])
    end
    return output
end

function _transform!(::Val{:nonfixed_indicators}, output, x; parameters...)
    @inbounds for index in eachindex(x, output)
        output[index] = x[index] != index
    end
    return output
end

function _cyclic_index_l1(x; dim, parameters...)
    blocks, width = _block_layout(x, Int(dim))
    violation = 0.0
    @inbounds for block in 0:(blocks - 1)
        source_offset = block * width
        target_offset = ((block + 1) % blocks) * width
        for local_index in 1:width
            indirect_index = x[source_offset + local_index]
            if indirect_index isa Integer && 1 ≤ indirect_index ≤ width
                violation += abs(Float64(
                    x[target_offset + Int(indirect_index)] - local_index))
            else
                violation += local_index
            end
        end
    end
    return violation
end

function _indicator_index_l1(x; id, parameters...)
    violation = 0.0
    @inbounds for index in eachindex(x)
        violation += abs(Float64(x[index]) - (index == id))
    end
    return violation
end

"""Workspace-free count of nodes without a predecessor."""
function _functional_graph_predecessor_penalty(
        x; op = nothing, val = nothing, parameters...)
    Base.require_one_based_indexing(x)
    missing = 0
    @inbounds for target in eachindex(x)
        predecessors = 0
        for successor in x
            predecessors += successor isa Integer && successor == target
        end
        missing += iszero(predecessors)
    end
    return Float64(missing)
end

"""Workspace-free count of active nodes outside the first active orbit."""
function _functional_graph_orbit_penalty(
        x; op = nothing, val = nothing, parameters...)
    Base.require_one_based_indexing(x)
    n = length(x)
    first_active = findfirst(index -> x[index] != index, eachindex(x))
    isnothing(first_active) && return Float64(!isempty(x))
    exclusions = 0
    @inbounds for target in eachindex(x)
        x[target] == target && continue
        reached = false
        current = first_active
        for _ in 1:n
            current isa Integer && 1 <= current <= n || break
            if current == target
                reached = true
                break
            end
            x[Int(current)] == current && break
            current = x[Int(current)]
        end
        exclusions += !reached
    end
    return Float64(exclusions)
end

function _nonfixed_condition_penalty(x, op::F, val) where {F}
    active = 0
    @inbounds for index in eachindex(x)
        active += x[index] != index
    end
    return _condition_residual(active, val, op)
end

_nonfixed_condition_penalty(x; op, val, parameters...) =
    _nonfixed_condition_penalty(x, op, val)

"""Duplicate count among absolute distances of disjoint adjacent pairs."""
function _pair_distance_collision_penalty(x; parameters...)
    Base.require_one_based_indexing(x)
    pairs = length(x) ÷ 2
    duplicates = 0
    @inbounds for right_pair in 2:pairs
        right = 2right_pair - 1
        right_distance = abs(x[right] - x[right + 1])
        repeated = false
        for left_pair in 1:(right_pair - 1)
            left = 2left_pair - 1
            if abs(x[left] - x[left + 1]) == right_distance
                repeated = true
                break
            end
        end
        duplicates += repeated
    end
    return Float64(duplicates)
end

for (name, predicate, side) in (
    (:count_equal_right, :(==), :right),
    (:count_less_right, :(<), :right),
    (:count_great_right, :(>), :right),
    (:count_equal_left, :(==), :left),
    (:count_less_left, :(<), :left),
    (:count_great_left, :(>), :left)
)
    @eval function _transform!(::Val{$(QuoteNode(name))}, output, x;
            parameters...)
        @inbounds for i in eachindex(x, output)
            first_index = $(QuoteNode(side)) === :right ? i + 1 : firstindex(x)
            last_index = $(QuoteNode(side)) === :right ? lastindex(x) : i - 1
            output[i] = _count_relation($predicate, x, x[i], first_index, last_index)
        end
        return output
    end
end

for (name, predicate) in (
    (:count_equal, :(==)),
    (:count_less, :(<)),
    (:count_great, :(>))
)
    @eval function _transform!(::Val{$(QuoteNode(name))}, output, x;
            parameters...)
        @inbounds for i in eachindex(x, output)
            output[i] = _count_relation_except($predicate, x, x[i], i)
        end
        return output
    end
end

for (name, predicate) in (
    (:count_equal_val, :(==)),
    (:count_less_val, :(<)),
    (:count_great_val, :(>))
)
    @eval function _transform!(::Val{$(QuoteNode(name))}, output, x; val,
            parameters...)
        @inbounds for i in eachindex(x, output)
            output[i] = _count_relation_except($predicate, x, x[i] + val, i)
        end
        return output
    end
end

function _transform!(::Val{:var_minus_val}, output, x; val, parameters...)
    @inbounds for i in eachindex(x, output)
        output[i] = max(0, x[i] - val)
    end
    return output
end

function _transform!(::Val{:val_minus_var}, output, x; val, parameters...)
    @inbounds for i in eachindex(x, output)
        output[i] = max(0, val - x[i])
    end
    return output
end

function _transform!(::Val{:contiguous_vars_minus}, output, x; parameters...)
    last = lastindex(x)
    @inbounds for i in eachindex(x, output)
        output[i] = i == last ? 0 : max(0, x[i] - x[i + 1])
    end
    return output
end

function _transform!(::Val{:contiguous_vars_minus_rev}, output, x; parameters...)
    last = lastindex(x)
    @inbounds for i in eachindex(x, output)
        output[i] = i == last ? 0 : max(0, x[i + 1] - x[i])
    end
    return output
end

function _transform!(::Val{:count_bounding_val}, output, x; val, parameters...)
    @inbounds for i in eachindex(x, output)
        lower = x[i]
        upper = lower + val
        count = 0
        for j in eachindex(x)
            j == i && continue
            count += lower <= x[j] <= upper
        end
        output[i] = count
    end
    return output
end

function _transform!(::Val{:var_minus_vals}, output, x; vals, parameters...)
    @inbounds for i in eachindex(x, output)
        value = zero(eltype(output))
        for parameter in vals
            value = max(value, x[i] - parameter)
        end
        output[i] = value
    end
    return output
end

function _transform!(::Val{:vals_minus_var}, output, x; vals, parameters...)
    @inbounds for i in eachindex(x, output)
        value = zero(eltype(output))
        for parameter in vals
            value = max(value, parameter - x[i])
        end
        output[i] = value
    end
    return output
end

function _combine_rows!(::Val{:sum}, workspace, rows::Int, columns::Int)
    @inbounds for row in 1:rows
        value = zero(eltype(workspace))
        for column in 1:columns
            value += workspace[row, column]
        end
        workspace[row, 1] = value
    end
    return workspace
end

function _combine_rows!(::Val{:product}, workspace, rows::Int, columns::Int)
    @inbounds for row in 1:rows
        value = one(eltype(workspace))
        for column in 1:columns
            value *= workspace[row, column]
        end
        workspace[row, 1] = value
    end
    return workspace
end

function _combine_rows!(::Val{:difference}, workspace, rows::Int, columns::Int)
    columns == 2 || throw(DimensionMismatch(
        "an aligned vector difference requires exactly two operands"))
    @inbounds for row in 1:rows
        workspace[row, 1] -= workspace[row, 2]
    end
    return workspace
end

function _aggregate(::Val{:sum}, workspace, rows::Int; parameters...)
    value = zero(eltype(workspace))
    @inbounds for row in 1:rows
        value += workspace[row, 1]
    end
    return value
end

_aggregate(::Val{:count_elements}, workspace, rows::Int; parameters...) = rows

function _aggregate(::Val{:count_zero}, workspace, rows::Int; parameters...)
    value = 0
    @inbounds for row in 1:rows
        value += iszero(workspace[row, 1])
    end
    return value
end

function _aggregate(::Val{:count_positive}, workspace, rows::Int; parameters...)
    value = 0
    @inbounds for row in 1:rows
        value += workspace[row, 1] > 0
    end
    return value
end

function _aggregate(::Val{:count_op_val}, workspace, rows::Int; val, op,
        parameters...)
    value = 0
    @inbounds for row in 1:rows
        value += op(workspace[row, 1], val)
    end
    return value
end

function _aggregate(::Val{:maximum}, workspace, rows::Int; parameters...)
    rows == 0 && return typemax(eltype(workspace))
    value = workspace[1, 1]
    @inbounds for row in 2:rows
        value = max(value, workspace[row, 1])
    end
    return value
end

function _aggregate(::Val{:minimum}, workspace, rows::Int; parameters...)
    rows == 0 && return typemax(eltype(workspace))
    value = workspace[1, 1]
    @inbounds for row in 2:rows
        value = min(value, workspace[row, 1])
    end
    return value
end

"Aggregate an identity-transformed input without copying it into a workspace."
function _aggregate_input(::Val{:sum}, x; parameters...)
    value = zero(eltype(x))
    @inbounds for item in x
        value += item
    end
    return value
end

_aggregate_input(::Val{:count_elements}, x; parameters...) = length(x)
_aggregate_input(::Val{:first_or_zero}, x; parameters...) = isempty(x) ? zero(eltype(x)) : first(x)
_aggregate_input(::Val{:argmin}, x; parameters...) = isempty(x) ? 0 : argmin(x)
_aggregate_input(::Val{:argmax}, x; parameters...) = isempty(x) ? 0 : argmax(x)

function _aggregate_input(::Val{:count_zero}, x; parameters...)
    value = 0
    @inbounds for item in x
        value += iszero(item)
    end
    return value
end

function _aggregate_input(::Val{:count_positive}, x; parameters...)
    value = 0
    @inbounds for item in x
        value += item > 0
    end
    return value
end

function _aggregate_input(::Val{:count_op_val}, x; val, op, parameters...)
    value = 0
    @inbounds for item in x
        value += op(item, val)
    end
    return value
end


function _aggregate_input(::Val{:maximum}, x; parameters...)
    isempty(x) && return typemax(eltype(x))
    value = @inbounds x[firstindex(x)]
    @inbounds for index in (firstindex(x) + 1):lastindex(x)
        value = max(value, x[index])
    end
    return value
end

function _aggregate_input(::Val{:minimum}, x; parameters...)
    isempty(x) && return typemax(eltype(x))
    value = @inbounds x[firstindex(x)]
    @inbounds for index in (firstindex(x) + 1):lastindex(x)
        value = min(value, x[index])
    end
    return value
end

_supports_fused_aggregation(::Val, ::Val) = false
_supports_fused_aggregation(::Val{:count_equal_left}, ::Val{:count_zero}) = true
_supports_fused_aggregation(::Val{:count_equal_right}, ::Val{:count_zero}) = true
_supports_fused_aggregation(::Val{:count_equal_left}, ::Val{:count_positive}) = true
_supports_fused_aggregation(::Val{:count_equal_right}, ::Val{:count_positive}) = true
_supports_fused_aggregation(::Val{:val_minus_var}, ::Val{:count_positive}) = true
_supports_fused_aggregation(::Val{:contiguous_vars_minus}, ::Val{:sum}) = true

_is_pairwise_count(::Val) = false
for operation in (
    :count_equal_right,
    :count_less_right,
    :count_great_right,
    :count_equal_left,
    :count_less_left,
    :count_great_left,
    :count_equal,
    :count_less,
    :count_great,
    :count_equal_val,
    :count_less_val,
    :count_great_val,
    :count_bounding_val,
)
    @eval _is_pairwise_count(::Val{$(QuoteNode(operation))}) = true
end
_is_pairwise_count(operation::Symbol) = _is_pairwise_count(Val(operation))

@inline _pairwise_count(
    ::Val{:count_equal_right}, x, i, j; parameters...
) = j > i && x[j] == x[i]
@inline _pairwise_count(
    ::Val{:count_less_right}, x, i, j; parameters...
) = j > i && x[j] < x[i]
@inline _pairwise_count(
    ::Val{:count_great_right}, x, i, j; parameters...
) = j > i && x[j] > x[i]
@inline _pairwise_count(
    ::Val{:count_equal_left}, x, i, j; parameters...
) = j < i && x[j] == x[i]
@inline _pairwise_count(
    ::Val{:count_less_left}, x, i, j; parameters...
) = j < i && x[j] < x[i]
@inline _pairwise_count(
    ::Val{:count_great_left}, x, i, j; parameters...
) = j < i && x[j] > x[i]
@inline _pairwise_count(
    ::Val{:count_equal}, x, i, j; parameters...
) = j != i && x[j] == x[i]
@inline _pairwise_count(
    ::Val{:count_less}, x, i, j; parameters...
) = j != i && x[j] < x[i]
@inline _pairwise_count(
    ::Val{:count_great}, x, i, j; parameters...
) = j != i && x[j] > x[i]
@inline _pairwise_count(
    ::Val{:count_equal_val}, x, i, j; val, parameters...
) = j != i && x[j] == x[i] + val
@inline _pairwise_count(
    ::Val{:count_less_val}, x, i, j; val, parameters...
) = j != i && x[j] < x[i] + val
@inline _pairwise_count(
    ::Val{:count_great_val}, x, i, j; val, parameters...
) = j != i && x[j] > x[i] + val
@inline _pairwise_count(
    ::Val{:count_bounding_val}, x, i, j; val, parameters...
) = j != i && x[i] <= x[j] <= x[i] + val

@inline _pairwise_counts(::Tuple{}, x, i, j; parameters...) = 0
@inline function _pairwise_counts(operations::Tuple, x, i, j; parameters...)
    operation = first(operations)
    return _pairwise_count(Val(operation), x, i, j; parameters...) +
           _pairwise_counts(Base.tail(operations), x, i, j; parameters...)
end

"Fuse any learned sum of pairwise-count transformations into one nested loop."
function _aggregate_pairwise_sum(::Val{operations}, x; parameters...) where {operations}
    total = 0
    indices = eachindex(x)
    # Directional counts are zero outside their strict index triangle. The
    # operation tuple is part of the compiled decoder's type, so this choice
    # does not add a branch to each pair or retain a mutable learning network.
    if indices isa AbstractUnitRange{<:Integer} && all(operation -> operation in
            (:count_equal_left, :count_less_left, :count_great_left), operations)
        @inbounds for i in indices, j in first(indices):(i - 1)
            total += _pairwise_counts(operations, x, i, j; parameters...)
        end
    elseif indices isa AbstractUnitRange{<:Integer} && all(operation -> operation in
            (:count_equal_right, :count_less_right, :count_great_right), operations)
        @inbounds for i in indices, j in (i + 1):last(indices)
            total += _pairwise_counts(operations, x, i, j; parameters...)
        end
    else
        @inbounds for i in eachindex(x), j in eachindex(x)
            total += _pairwise_counts(operations, x, i, j; parameters...)
        end
    end
    return total
end

_is_elementwise(::Val) = false
for operation in (
    :id,
    :positive_part,
    :var_minus_val,
    :val_minus_var,
    :contiguous_vars_minus,
    :contiguous_vars_minus_rev,
    :var_minus_vals,
    :vals_minus_var,
)
    @eval _is_elementwise(::Val{$(QuoteNode(operation))}) = true
end
_is_elementwise(operation::Symbol) = _is_elementwise(Val(operation))

@inline _elementwise_value(::Val{:id}, x, i; parameters...) = @inbounds x[i]
@inline _elementwise_value(::Val{:positive_part}, x, i; parameters...) =
    @inbounds max(zero(x[i]), x[i])
@inline _elementwise_value(::Val{:var_minus_val}, x, i; val, parameters...) =
    @inbounds max(0, x[i] - val)
@inline _elementwise_value(::Val{:val_minus_var}, x, i; val, parameters...) =
    @inbounds max(0, val - x[i])
@inline function _elementwise_value(
        ::Val{:contiguous_vars_minus}, x, i; parameters...)
    return i == lastindex(x) ? 0 : @inbounds max(0, x[i] - x[i + 1])
end
@inline function _elementwise_value(
        ::Val{:contiguous_vars_minus_rev}, x, i; parameters...)
    return i == lastindex(x) ? 0 : @inbounds max(0, x[i + 1] - x[i])
end
@inline function _elementwise_value(::Val{:var_minus_vals}, x, i; vals, parameters...)
    value = zero(eltype(x))
    @inbounds for parameter in vals
        value = max(value, x[i] - parameter)
    end
    return value
end
@inline function _elementwise_value(::Val{:vals_minus_var}, x, i; vals, parameters...)
    value = zero(eltype(x))
    @inbounds for parameter in vals
        value = max(value, parameter - x[i])
    end
    return value
end
@inline _elementwise_combine(::Tuple{}, ::Val{:sum}, x, i; parameters...) = 0
@inline _elementwise_combine(::Tuple{}, ::Val{:product}, x, i; parameters...) = 1
@inline function _elementwise_combine(
        operations::Tuple, arithmetic::Val{:sum}, x, i; parameters...)
    return _elementwise_value(Val(first(operations)), x, i; parameters...) +
           _elementwise_combine(Base.tail(operations), arithmetic, x, i; parameters...)
end
@inline function _elementwise_combine(
        operations::Tuple, arithmetic::Val{:product}, x, i; parameters...)
    return _elementwise_value(Val(first(operations)), x, i; parameters...) *
           _elementwise_combine(Base.tail(operations), arithmetic, x, i; parameters...)
end

function _aggregate_elementwise(
        ::Val{operations}, arithmetic, ::Val{:sum}, x; parameters...
) where {operations}
    value = zero(eltype(x))
    @inbounds for i in eachindex(x)
        value += _elementwise_combine(operations, arithmetic, x, i; parameters...)
    end
    return value
end
function _aggregate_elementwise(
        ::Val{operations}, arithmetic, ::Val{:count_positive}, x; parameters...
) where {operations}
    value = 0
    @inbounds for i in eachindex(x)
        value += _elementwise_combine(operations, arithmetic, x, i; parameters...) > 0
    end
    return value
end
function _aggregate_elementwise(
        ::Val{operations}, arithmetic, ::Val{:count_zero}, x; parameters...
) where {operations}
    value = 0
    @inbounds for i in eachindex(x)
        value += iszero(_elementwise_combine(operations, arithmetic, x, i; parameters...))
    end
    return value
end
function _aggregate_elementwise(
        ::Val{operations}, arithmetic, ::Val{:count_op_val}, x; val, op, parameters...
) where {operations}
    value = 0
    @inbounds for i in eachindex(x)
        item = _elementwise_combine(operations, arithmetic, x, i; val, op, parameters...)
        value += op(item, val)
    end
    return value
end
function _aggregate_elementwise(
        ::Val{operations}, arithmetic,
        aggregation::Union{Val{:minimum}, Val{:maximum}}, x; parameters...
) where {operations}
    isempty(x) && return typemax(eltype(x))
    first = firstindex(x)
    value = _elementwise_combine(operations, arithmetic, x, first; parameters...)
    @inbounds for i = (first + 1):lastindex(x)
        item = _elementwise_combine(operations, arithmetic, x, i; parameters...)
        value = aggregation isa Val{:minimum} ? min(value, item) : max(value, item)
    end
    return value
end

@inline _mixed_elementwise(::Tuple{}, x, i; parameters...) = 0
@inline function _mixed_elementwise(operations::Tuple, x, i; parameters...)
    operation = Val(first(operations))
    value = _is_elementwise(operation) ?
            _elementwise_value(operation, x, i; parameters...) : 0
    return value + _mixed_elementwise(Base.tail(operations), x, i; parameters...)
end
@inline _mixed_pairwise(::Tuple{}, x, i, j; parameters...) = 0
@inline function _mixed_pairwise(operations::Tuple, x, i, j; parameters...)
    operation = Val(first(operations))
    value = _is_pairwise_count(operation) ?
            _pairwise_count(operation, x, i, j; parameters...) : 0
    return value + _mixed_pairwise(Base.tail(operations), x, i, j; parameters...)
end

function _aggregate_mixed_sum(::Val{operations}, x; parameters...) where {operations}
    total = 0
    @inbounds for i in eachindex(x)
        total += _mixed_elementwise(operations, x, i; parameters...)
        for j in eachindex(x)
            total += _mixed_pairwise(operations, x, i, j; parameters...)
        end
    end
    return total
end

@inline _filter_keep(::Val{:id}, x, i; parameters...) = true
@inline function _filter_keep(::Val{:filter_unique}, x, i; parameters...)
    @inbounds for j = firstindex(x):(i - 1)
        x[j] == x[i] && return false
    end
    return true
end
@inline _filter_keep(::Val{:filter_elem}, x, i; id, parameters...) = i == id
@inline _filter_keep(::Val{:filter_op_val}, x, i; val, op, parameters...) = op(x[i], val)
@inline _filter_keep(::Val{:filter_equal_val}, x, i; val, parameters...) = x[i] == val
@inline _filter_keep(::Val{:filter_ge_val}, x, i; val, parameters...) = x[i] >= val
@inline _filter_keep(::Val{:filter_great_val}, x, i; val, parameters...) = x[i] > val
@inline _filter_keep(::Val{:filter_less_val}, x, i; val, parameters...) = x[i] < val
@inline _filter_keep(::Val{:filter_le_val}, x, i; val, parameters...) = x[i] <= val
@inline _filter_keep(::Val{:filter_ne_val}, x, i; val, parameters...) = x[i] != val
@inline _filter_keep(::Val{:filter_equal_filter_val}, x, i; filter_val, parameters...) =
    x[i] == filter_val
@inline _filter_keep(::Val{:filter_equal_vals}, x, i; vals, parameters...) = x[i] in vals
@inline _filter_keep(::Val{:filter_ne_vals}, x, i; vals, parameters...) = !(x[i] in vals)
@inline function _filter_keep(::Val{:filter_op_vals}, x, i; vals, op, parameters...)
    @inbounds for parameter in vals
        op(x[i], parameter) || return false
    end
    return true
end

function _aggregate_filter(operation, ::Val{:sum}, x; parameters...)
    value = zero(eltype(x))
    @inbounds for i in eachindex(x)
        _filter_keep(operation, x, i; parameters...) && (value += x[i])
    end
    return value
end

function _aggregate_filter(operation, ::Val{:count_elements}, x; parameters...)
    value = 0
    @inbounds for i in eachindex(x)
        value += _filter_keep(operation, x, i; parameters...)
    end
    return value
end

function _aggregate_filter(operation, ::Val{:count_zero}, x; parameters...)
    value = 0
    @inbounds for i in eachindex(x)
        value += _filter_keep(operation, x, i; parameters...) && iszero(x[i])
    end
    return value
end

function _aggregate_filter(operation, ::Val{:count_positive}, x; parameters...)
    value = 0
    @inbounds for i in eachindex(x)
        value += _filter_keep(operation, x, i; parameters...) && x[i] > 0
    end
    return value
end

function _aggregate_filter(operation, ::Val{:count_op_val}, x; val, op, parameters...)
    value = 0
    @inbounds for i in eachindex(x)
        value += _filter_keep(operation, x, i; val, op, parameters...) && op(x[i], val)
    end
    return value
end

function _aggregate_filter(operation, aggregation::Union{Val{:minimum}, Val{:maximum}}, x;
        parameters...)
    found = false
    value = typemax(eltype(x))
    @inbounds for i in eachindex(x)
        _filter_keep(operation, x, i; parameters...) || continue
        value = found ? (aggregation isa Val{:minimum} ? min(value, x[i]) : max(value, x[i])) : x[i]
        found = true
    end
    return found ? value : typemax(eltype(x))
end

@inline _paired_scalar(::Val{:id}, value, pair_value) = value
@inline _paired_scalar(::Val{:sub}, value, pair_value) = abs(value - pair_value)
@inline _paired_scalar(::Val{:sum}, value, pair_value) = value + pair_value
@inline _paired_scalar(::Val{:prod}, value, pair_value) = value * pair_value
@inline _paired_scalar(::Val{:aligned_not_equal}, value, pair_value) = value != pair_value
@inline _paired_scalar(::Val{:aligned_difference}, value, pair_value) = value - pair_value
@inline _paired_value(operation, x, pair_vars, i) =
    @inbounds _paired_scalar(operation, x[i], pair_vars[i])

function _aggregate_adjacent_left_affine(
        aggregation, x; pair_vars, parameters...)
    axes(x) == axes(pair_vars) || throw(DimensionMismatch(
        "adjacent affine values and parameters must have the same axes",
    ))
    Base.require_one_based_indexing(x, pair_vars)
    length(x) < 2 && return 0.0
    if aggregation isa Val{:minimum}
        value = Inf
    else
        value = 0.0
    end
    @inbounds for index in 1:(length(x) - 1)
        margin = x[index] + pair_vars[index] - x[index + 1]
        value_at_index = Float64(margin)
        if aggregation isa Val{:sum}
            value += value_at_index
        elseif aggregation isa Val{:count_zero}
            value += iszero(value_at_index)
        elseif aggregation isa Val{:count_positive}
            value += value_at_index > 0
        elseif aggregation isa Val{:maximum}
            value = max(value, value_at_index)
        elseif aggregation isa Val{:minimum}
            value = min(value, value_at_index)
        else
            throw(ArgumentError(
                "unsupported adjacent affine aggregation $(typeof(aggregation))",
            ))
        end
    end
    return value
end


function _aggregate_adjacent_paired(aggregation, x; pair_vars, parameters...)
    return _aggregate_adjacent_left_affine(
        aggregation, x; pair_vars, parameters...)
end

for aggregation in (:sum, :count_zero, :count_positive, :minimum, :maximum)
    @eval function _aggregate_paired(
            ::Val{:adjacent_left_affine_margins}, operation::Val{$(QuoteNode(aggregation))}, x;
            pair_vars, parameters...)
        return _aggregate_adjacent_paired(operation, x; pair_vars, parameters...)
    end
end

function _aggregate_paired(operation, ::Val{:sum}, x; pair_vars, parameters...)
    axes(x) == axes(pair_vars) || throw(DimensionMismatch("x and pair_vars must align"))
    value = zero(promote_type(eltype(x), eltype(pair_vars)))
    @inbounds for i in eachindex(x, pair_vars)
        value += _paired_value(operation, x, pair_vars, i)
    end
    return value
end

function _aggregate_paired(
        operation::Val{:aligned_not_equal}, ::Val{:sum}, x; pair_vars, parameters...)
    axes(x) == axes(pair_vars) || throw(DimensionMismatch("x and pair_vars must align"))
    value = 0
    @inbounds for i in eachindex(x, pair_vars)
        value += _paired_value(operation, x, pair_vars, i)
    end
    return value
end

function _aggregate_paired(operation, ::Val{:count_positive}, x; pair_vars, parameters...)
    axes(x) == axes(pair_vars) || throw(DimensionMismatch("x and pair_vars must align"))
    value = 0
    @inbounds for i in eachindex(x, pair_vars)
        value += _paired_value(operation, x, pair_vars, i) > 0
    end
    return value
end

function _aggregate_paired(operation, ::Val{:count_zero}, x; pair_vars, parameters...)
    axes(x) == axes(pair_vars) || throw(DimensionMismatch("x and pair_vars must align"))
    value = 0
    @inbounds for i in eachindex(x, pair_vars)
        value += iszero(_paired_value(operation, x, pair_vars, i))
    end
    return value
end

function _aggregate_paired(operation, ::Val{:count_op_val}, x; pair_vars, val, op,
        parameters...)
    axes(x) == axes(pair_vars) || throw(DimensionMismatch("x and pair_vars must align"))
    value = 0
    @inbounds for i in eachindex(x, pair_vars)
        value += op(_paired_value(operation, x, pair_vars, i), val)
    end
    return value
end

function _aggregate_paired(
        operation, aggregation::Union{Val{:minimum}, Val{:maximum}}, x;
        pair_vars, parameters...)
    axes(x) == axes(pair_vars) || throw(DimensionMismatch("x and pair_vars must align"))
    isempty(x) && return typemax(promote_type(eltype(x), eltype(pair_vars)))
    first = firstindex(x)
    value = _paired_value(operation, x, pair_vars, first)
    @inbounds for i = (first + 1):lastindex(x)
        item = _paired_value(operation, x, pair_vars, i)
        value = aggregation isa Val{:minimum} ? min(value, item) : max(value, item)
    end
    return value
end

@inline function _pairwise_disjunction_residual(
        x,
        pair_vars,
        dimensions::Int,
        first_task::Int,
        second_task::Int,
        zero_ignored::Bool,
)
    _pair_is_disabled(
        pair_vars, dimensions, first_task, second_task, zero_ignored,
    ) && return 0.0
    first_offset = (first_task - 1) * dimensions
    second_offset = (second_task - 1) * dimensions
    minimum_margin = Inf
    @inbounds for dimension in 1:dimensions
        first_index = first_offset + dimension
        second_index = second_offset + dimension
        minimum_margin = min(
            minimum_margin,
            Float64(x[first_index] + pair_vars[first_index] - x[second_index]),
            Float64(x[second_index] + pair_vars[second_index] - x[first_index]),
        )
    end
    return max(0.0, minimum_margin)
end

@inline function _aggregate_pairwise_disjunction(
        aggregation,
        x;
        pair_vars,
        dim = 1,
        bool = true,
        parameters...,
)
    dimensions = Int(dim)
    zero_ignored = Bool(bool)
    paired = _paired_parameters(pair_vars)
    tasks = _check_pairwise_geometry_arguments(x, paired, dimensions)
    tasks < 2 && return aggregation isa Union{Val{:minimum}, Val{:maximum}} ? Inf : 0.0
    result = 0.0
    @inbounds for first_task in 1:(tasks - 1)
        for second_task in (first_task + 1):tasks
            residual = _pairwise_disjunction_residual(
                x, paired, dimensions, first_task, second_task, zero_ignored,
            )
            if aggregation isa Val{:sum}
                result += residual
            elseif aggregation isa Val{:count_positive}
                result += !iszero(residual)
            elseif aggregation isa Val{:maximum}
                result = max(result, residual)
            else
                # The explicit grouped representation contains zero padding, so
                # its minimum is zero whenever at least one pair exists.
                return 0.0
            end
        end
    end
    return result
end

function _aggregate_specialized(
        ::Val{:SimpleFilter}, ::Val{:filter_elem}, aggregation, x; id, parameters...)
    value = x[id]
    return _aggregate_input(aggregation, (value,); parameters...)
end
function _aggregate_specialized(
        ::Val{:SimpleFilter}, ::Val{:filter_id}, aggregation, x; id, parameters...)
    first = x[id]
    second = 0 < first <= length(x) ? -(@inbounds x[first]) : typemax(eltype(x))
    return _aggregate_input(aggregation, (first, second); parameters...)
end
function _aggregate_specialized(
        ::Val{:SimpleFilter}, operation, aggregation, x; parameters...)
    return _aggregate_filter(operation, aggregation, x; parameters...)
end

function _aggregate_specialized(
        ::Val{:SimpleFilter},
        filter_operation,
        equal_counts::Union{Val{:count_equal_left}, Val{:count_equal_right}},
        ::Val{:count_zero},
        x;
        parameters...,
)
    distinct = 0
    @inbounds for i in eachindex(x)
        _filter_keep(filter_operation, x, i; parameters...) || continue
        first_occurrence = true
        for j in firstindex(x):(i - 1)
            if _filter_keep(filter_operation, x, j; parameters...) && x[j] == x[i]
                first_occurrence = false
                break
            end
        end
        distinct += first_occurrence
    end
    return distinct
end
function _aggregate_specialized(
        ::Val{:PairedMap}, operation, aggregation, x; parameters...)
    return _aggregate_paired(operation, aggregation, x; parameters...)
end


function _aggregate_specialized(
        ::Val{:EventMap},
        ::Val{:weighted_interval_profile},
        ::Val{:maximum},
        x;
        pair_vars,
        X = nothing,
        parameters...,
)
    workspace = isnothing(X) ? EventProfileWorkspace(
        Vector{Tuple{Float64, Float64}}(undef, 2length(x)),
    ) : X
    workspace isa EventProfileWorkspace || throw(ArgumentError(
        "weighted interval profiles require an EventProfileWorkspace",
    ))
    return _maximum_weighted_interval_load!(workspace.events, x, pair_vars)
end

@inline _condition_area_term(::Val{:segment_condition_area}, width, residual) =
    Float64(width) * residual
@inline _condition_area_term(::Val{:segment_squared_condition_area}, width, residual) =
    Float64(width) * residual * residual

function _weighted_interval_le_area_unchecked!(
        events, ::Val{Power}, values, pair_vars, val) where {Power}
    @inbounds for task in eachindex(values)
        event = 2task - 1
        height = pair_vars[2, task]
        events[event] = (values[task], height)
        events[event + 1] = (values[task] + pair_vars[1, task], -height)
    end
    sort!(events; alg = Base.Sort.QuickSort)

    usage = 0.0
    area = 0.0
    event = firstindex(events)
    @inbounds while event <= lastindex(events)
        time = events[event][1]
        while event <= lastindex(events) && events[event][1] == time
            usage += events[event][2]
            event += 1
        end
        if event <= lastindex(events)
            width = events[event][1] - time
            residual = max(0.0, usage - val)
            if Power === :linear
                area += width * residual
            else
                area += width * residual * residual
            end
        end
    end
    outside_residual = max(0.0, -Float64(val))
    if Power === :linear
        area += outside_residual
    else
        area += outside_residual * outside_residual
    end
    return area
end

function _weighted_interval_le_area!(events, power, values, pair_vars, val)
    _check_weighted_interval_arguments(values, pair_vars)
    length(events) == 2length(values) || throw(DimensionMismatch(
        "weighted interval event workspace must contain two events per origin"))
    return _weighted_interval_le_area_unchecked!(events, power, values, pair_vars, val)
end

function _weighted_interval_condition_area!(events,
        ::Val{:segment_condition_area}, values, pair_vars, ::typeof(<=), val)
    return _weighted_interval_le_area!(events, Val(:linear), values, pair_vars, val)
end

function _weighted_interval_condition_area!(events,
        ::Val{:segment_squared_condition_area}, values, pair_vars, ::typeof(<=), val)
    return _weighted_interval_le_area!(events, Val(:quadratic), values, pair_vars, val)
end


function _weighted_interval_condition_area_unchecked!(events,
        ::Val{:segment_condition_area}, values, pair_vars, ::typeof(<=), val)
    return _weighted_interval_le_area_unchecked!(
        events, Val(:linear), values, pair_vars, val)
end


function _weighted_interval_condition_area_unchecked!(events,
        ::Val{:segment_squared_condition_area}, values, pair_vars, ::typeof(<=), val)
    return _weighted_interval_le_area_unchecked!(
        events, Val(:quadratic), values, pair_vars, val)
end


function _weighted_interval_condition_area_unchecked!(
        events, reduction, values, pair_vars, op, val)
    return _weighted_interval_condition_area!(
        events, reduction, values, pair_vars, op, val)
end

function _weighted_interval_condition_area!(
        events,
        reduction,
        values,
        pair_vars,
        op,
        val,
)
    _check_weighted_interval_arguments(values, pair_vars)
    length(events) == 2length(values) || throw(DimensionMismatch(
        "weighted interval event workspace must contain two events per origin"))
    @inbounds for task in eachindex(values)
        event = 2task - 1
        height = pair_vars[2, task]
        events[event] = (values[task], height)
        events[event + 1] = (values[task] + pair_vars[1, task], -height)
    end
    sort!(events; alg = Base.Sort.QuickSort)

    # Workspaces default to Float64 even when task data are integral. Keeping the
    # accumulator in the workspace's load type avoids a Union{Int,Float64} and
    # one boxed value per event after the first update.
    usage = 0.0
    area = 0.0
    operator = _condition_operator(op)
    event = firstindex(events)
    @inbounds while event <= lastindex(events)
        time = events[event][1]
        while event <= lastindex(events) && events[event][1] == time
            usage += events[event][2]
            event += 1
        end
        if event <= lastindex(events)
            width = events[event][1] - time
            residual = operator isa Val{:generic} ?
                       _condition_residual(usage, val, operator, op) :
                       _condition_residual(usage, val, operator)
            area += _condition_area_term(reduction, width, residual)
        end
    end
    outside_residual = operator isa Val{:generic} ?
                       _condition_residual(0.0, val, operator, op) :
                       _condition_residual(0.0, val, operator)
    area += _condition_area_term(reduction, 1.0, outside_residual)
    return area
end

function _aggregate_specialized(
        ::Val{:EventMap},
        ::Val{:weighted_interval_segments},
        reduction::Union{Val{:segment_condition_area},
            Val{:segment_squared_condition_area}},
        ::Val{:sum},
        x;
        pair_vars,
        op,
        val,
        X = nothing,
        parameters...,
)
    workspace = X isa EventProfileWorkspace ? X : EventProfileWorkspace(
        Vector{Tuple{Float64, Float64}}(undef, 2length(x)),
    )
    return _weighted_interval_condition_area!(
        workspace.events, reduction, x, pair_vars, op, val)
end

function _aggregate_specialized(
        front,
        operation,
        ::Val{:id},
        aggregation,
        x;
        parameters...,
)
    return _aggregate_specialized(front, operation, aggregation, x; parameters...)
end

function _aggregate_specialized(
        ::Val{:PairedMap},
        ::Val{:aligned_difference},
        ::Val{:absolute},
        aggregation,
        x;
        parameters...,
)
    # `sub` is retained as a legacy public operation. The compiler may fuse the two
    # selected atomic operations, but that fused alias is not exposed as a learned weight.
    return _aggregate_paired(Val(:sub), aggregation, x; parameters...)
end

function _aggregate_specialized(
        ::Val{:PairedMap},
        ::Val{:pairwise_oriented_affine_margins},
        ::Val{:grouped_min_positive},
        aggregation,
        x;
        pair_vars,
        dim = 1,
        bool = true,
        parameters...,
)
    return _aggregate_pairwise_disjunction(
        aggregation,
        x;
        pair_vars,
        dim,
        bool,
    )
end
function _aggregate_specialized(
        ::Val{:Language}, operation, aggregation, x;
        language, language_workspace = nothing, parameters...)
    value = if operation isa Val{:accept}
        Float64(!ConstraintCommons.accept(language, x))
    elseif operation isa Val{:reject}
        Float64(ConstraintCommons.accept(language, x))
    else
        Float64(_language_distance(language, x, language_workspace))
    end
    return _aggregate_input(aggregation, (value,); parameters...)
end

_language_distance(language, x, ::Nothing) =
    ConstraintCommons.language_distance(language, x)
_language_distance(language, x, workspace) =
    ConstraintCommons.language_distance(language, x, workspace)

@inline function _absolute_minimum(x, value)
    isempty(x) && return typemax(eltype(x))
    best = typemax(promote_type(eltype(x), typeof(value)))
    @inbounds for item in x
        best = min(best, abs(item - value))
        iszero(best) && return best
    end
    return best
end

@inline function _reduce_absolute_minimum_values(
        x,
        values,
        reduction::Symbol;
        op::Function = (==),
        val::Integer = 0,
        weights = nothing,
)
    return reduce_icn_outputs(
        value -> _absolute_minimum(x, value),
        values,
        reduction;
        op,
        val,
        weights,
    )
end

"Evaluate common learned transformation-aggregation pairs without materializing a vector."
function _aggregate_transform(
        ::Val{:count_equal_left}, ::Val{:count_positive}, x; parameters...)
    violations = 0
    @inbounds for i in eachindex(x)
        duplicate = false
        for j in firstindex(x):(i - 1)
            if x[j] == x[i]
                duplicate = true
                break
            end
        end
        violations += duplicate
    end
    return violations
end

function _aggregate_transform(
        ::Val{:count_equal_left}, ::Val{:count_zero}, x; parameters...)
    distinct = 0
    @inbounds for i in eachindex(x)
        first_occurrence = true
        for j in firstindex(x):(i - 1)
            if x[j] == x[i]
                first_occurrence = false
                break
            end
        end
        distinct += first_occurrence
    end
    return distinct
end

function _aggregate_transform(
        ::Val{:count_equal_right}, ::Val{:count_zero}, x; parameters...)
    distinct = 0
    @inbounds for i in eachindex(x)
        last_occurrence = true
        for j in (i + 1):lastindex(x)
            if x[j] == x[i]
                last_occurrence = false
                break
            end
        end
        distinct += last_occurrence
    end
    return distinct
end

function _aggregate_transform(
        ::Val{:count_equal_right}, ::Val{:count_positive}, x; parameters...)
    violations = 0
    @inbounds for i in eachindex(x)
        duplicate = false
        for j in (i + 1):lastindex(x)
            if x[j] == x[i]
                duplicate = true
                break
            end
        end
        violations += duplicate
    end
    return violations
end

function _aggregate_transform(
        ::Val{:val_minus_var}, ::Val{:count_positive}, x; val, parameters...)
    violations = 0
    @inbounds for value in x
        violations += value < val
    end
    return violations
end

function _aggregate_transform(
        ::Val{:contiguous_vars_minus}, ::Val{:sum}, x; parameters...)
    total = zero(eltype(x))
    @inbounds for i in firstindex(x):(lastindex(x) - 1)
        total += max(0, x[i] - x[i + 1])
    end
    return total
end

_compare(::Val{:id}, value; parameters...) = value
_compare(::Val{:condition_residual}, value; op, val, parameters...) =
    _condition_residual(value, val, op)
_compare(::Val{:abs_val}, value; val, parameters...) = abs(value - val)
_compare(::Val{:val_minus_var}, value; val, parameters...) = max(0, val - value)
_compare(::Val{:var_minus_val}, value; val, parameters...) = max(0, value - val)
function _compare(::Val{:euclidean_val}, value; val, dom_size, parameters...)
    value == val ? 0 : 1 + abs(value - val) / dom_size
end
function _compare(::Val{:euclidean_val_op}, value; op, val, dom_size, parameters...)
    op(value, val) ? 0 : 1 + abs(value - val) / dom_size
end
function _compare(::Val{:euclidean}, value; dom_size, parameters...)
    iszero(value) ? 0 : 1 + value / dom_size
end
function _compare(::Val{:euclidean_op}, value; op, dom_size, parameters...)
    op(value, 0) ? 0 : 1 + value / dom_size
end
_compare(::Val{:var_minus_numvars}, value; numvars, parameters...) = abs(value - numvars)
function _compare(::Val{:max_numvars_minus_var}, value; numvars, parameters...)
    max(0, numvars - value)
end
function _compare(::Val{:max_var_minus_numvars}, value; numvars, parameters...)
    max(value - numvars, 0)
end
function _compare(::Val{:vals_minus_var_gele}, value; vals, parameters...)
    length(vals) == 2 || return typemax(typeof(value))
    return vals[1] <= value <= vals[2] ? 0 :
           min(abs(value - vals[1]), abs(value - vals[2]))
end

function _compare(::Val{:vals_minus_var_gl}, value; vals, parameters...)
    length(vals) == 2 || return typemax(typeof(value))
    return vals[1] < value < vals[2] ? 0 :
           min(abs(value - vals[1]), abs(value - vals[2]))
end

@testitem "Compiled compositions reuse caller-owned workspace" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @test, @testset

    function select!(icn, operations)
        fill!(icn.weights.parent, false)
        offset = 0
        for (layer, selected) in zip(icn.layers, operations)
            names = collect(keys(layer.fn))
            for operation in selected
                icn.weights.parent[offset + only(findall(==(operation), names))] = true
            end
            offset += length(layer.fn)
        end
        return icn
    end

    icn = CN.ICN(parameters = [:dom_size, :numvars, :val, :vals])
    select!(icn, ((:id, :count_equal_right, :count_equal_left),
        (:sum,), (:sum,), (:id,)))
    @test CN.check_weights_validity(icn, icn.weights)

    compiled = first(CN.compose(icn; name = :workspace_composition_test))
    input = [1, 2, 1, 3]
    workspace = CN.composition_workspace(icn, length(input))
    parameters = (; dom_size = 3, numvars = 4, val = 2, vals = (1, 3))
    expected = CN.evaluate(icn, CN.Solution(input); parameters...)
    @test Base.invokelatest(compiled, input; X = workspace, parameters...) == expected
    @test size(workspace) == (length(input), CN.max_icn_length(icn))
    # This selection is fully fused: the generated function deliberately ignores
    # an obsolete/undersized matrix instead of validating storage it never reads.
    @test Base.invokelatest(
        compiled, input; X = zeros(length(input) - 1, 3), parameters...) == expected

    function composition_allocations(f, input, workspace)
        f(input; X = workspace, dom_size = 3, numvars = 4, val = 2,
            vals = (1, 3))
        return @allocated(f(
            input; X = workspace, dom_size = 3, numvars = 4, val = 2,
            vals = (1, 3)))
    end
    @test composition_allocations(compiled, input, workspace) == 0

    inputs = ([1, 3, 2, 1], [1], [0, 0, 0, 0], [4, 1, 4, 2, 3, 2])
    workspace = CN.composition_workspace(icn, maximum(length, inputs))
    function available_operations(icn)
        available = Vector{Vector{Symbol}}()
        compact_offset = 0
        full_offset = 0
        for (i, layer) in enumerate(icn.layers)
            compact_range = compact_offset .+ (1:icn.weightlen[i])
            relative_indices = icn.weights.indices[1][compact_range] .- full_offset
            push!(available, collect(keys(layer.fn))[relative_indices])
            compact_offset += icn.weightlen[i]
            full_offset += length(layer.fn)
        end
        return available
    end
    available = available_operations(icn)

    @testset "built-in operations match generic evaluation" begin
        for transformation in available[1]
            select!(icn, ([transformation], [:sum], [:sum], [:id]))
            compiled = first(CN.compose(icn))
            for values in inputs
                parameters = (; dom_size = 4, numvars = length(values), val = 2,
                    vals = (1, 3))
                observed = if transformation === :disjoint_pair_differences
                    Base.invokelatest(compiled, values; parameters...)
                else
                    Base.invokelatest(compiled, values; X = workspace, parameters...)
                end
                @test observed == CN.evaluate(icn, CN.Solution(values); parameters...)
            end
        end
        for arithmetic in available[2]
            select!(icn, ([:id, :count_equal], [arithmetic], [:sum], [:id]))
            compiled = first(CN.compose(icn))
            for values in inputs
                parameters = (; dom_size = 4, numvars = length(values), val = 2,
                    vals = (1, 3))
                @test Base.invokelatest(
                    compiled, values; X = workspace, parameters...) ==
                      CN.evaluate(icn, CN.Solution(values); parameters...)
            end
        end
        for aggregation in available[3]
            select!(icn, ([:id], [:sum], [aggregation], [:id]))
            compiled = first(CN.compose(icn))
            for values in inputs
                parameters = (; dom_size = 4, numvars = length(values), val = 2,
                    vals = (1, 3))
                @test Base.invokelatest(
                    compiled, values; X = workspace, parameters...) ==
                      CN.evaluate(icn, CN.Solution(values); parameters...)
            end
        end
        for comparison in available[4]
            select!(icn, ([:id], [:sum], [:sum], [comparison]))
            compiled = first(CN.compose(icn))
            for values in inputs
                parameters = (; dom_size = 4, numvars = length(values), val = 2,
                    vals = (1, 3))
                @test Base.invokelatest(
                    compiled, values; X = workspace, parameters...) ==
                      CN.evaluate(icn, CN.Solution(values); parameters...)
            end
        end
    end
end
