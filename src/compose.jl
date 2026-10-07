"""
Generate a julia function for a given ICN

Example usage:
```julia
compose(ICN(), name = :hopefullyworkingfunction)
```
"""
function compose(
        icn::AbstractICN;
        name::Symbol = gensym(),
        jlfun = true,
        fname = "",
        dbg = false
)
    index_relation = _index_relation_kernel(icn)
    if !(index_relation isa Val{:generic})
        return _compose_index_relation(icn, index_relation; name, jlfun, fname, dbg)
    elseif _supports_inplace_compilation(icn)
        return _compose_inplace(icn; name, jlfun, fname, dbg)
    elseif _supports_specialized_compilation(icn)
        return _compose_specialized(icn; name, jlfun, fname, dbg)
    end

    f = JLFunction()
    f.name = name
    f.args = [:x]
    # Generic decoding retains the caller-owned-workspace call contract, even
    # when a data-dependent shape currently needs an allocating fallback.
    f.kwargs = Any[Expr(:kw, :X, :nothing), (p for p in icn.parameters if p != :X)...]

    fns = []
    _start = 1
    weights = icn.weights.parent
    for (i, layer) in enumerate(icn.layers)
        j = findall(weights[_start:(_start - 1 + length(layer.fn))])
        if layer.mutex
            push!(fns, :($(layer.name) = x = $(layer.fnexprs[j[1]].body)))
        else
            temp = xtuple([layer.fnexprs[k].body for k in j]...)
            push!(
                fns,
                :(
                    $(layer.name) = x = $(temp) |>
                                        ifelse(
                    isempty(x), r -> collect(typeof(x), r), collect)
                )
            )
        end
        if dbg
            push!(
                fns,
                :(@info($(string(layer.name)), $(layer.name), typeof($(layer.name))))
            )
        end
        _start += length(layer.fn)
    end
    f.body = Expr(:block, push!(fns, :(return x))...)
    if !isempty(fname)
        open(fname, "w") do fio
            write(fio, sprint_expr(f))
        end
    end
    return (eval(codegen_ast(f)), jlfun ? f : codegen_ast(f))
end

function _selected_operation_names(icn::AbstractICN)
    return _selected_operation_names(icn, icn.weights)
end

function _index_relation_kernel(icn::AbstractICN)
    :index_base in icn.parameters && return Val(:generic)
    selected = _selected_operation_names(icn)
    names = Tuple(layer.name for layer in icn.layers)
    if names == (:Transformation, :Arithmetic, :Pointwise, :Aggregation, :Comparison) &&
       Set(selected[1]) == Set((:block_local_indices, :cyclic_indirect_values)) &&
       selected[2] == [:difference] && selected[3] == [:absolute] &&
       selected[4] == [:sum] && selected[5] == [:id]
        return Val(:cyclic)
    elseif names == (:SimpleFilter, :Transformation, :Arithmetic, :Pointwise,
            :Aggregation, :Comparison) && selected[1] == [:id] &&
           Set(selected[2]) == Set((:id, :index_indicator)) &&
           selected[3] == [:difference] && selected[4] == [:absolute] &&
           selected[5] == [:sum] && selected[6] == [:id]
        return Val(:indicator)
    elseif names == (:Transformation, :Arithmetic, :Aggregation, :Comparison) &&
           selected[1] == [:predecessor_counts] && selected[2] == [:sum] &&
           selected[3] == [:count_zero] && selected[4] == [:id]
        return Val(:functional_graph_predecessors)
    elseif names == (:Transformation, :Arithmetic, :Aggregation, :Comparison) &&
           selected[1] == [:orbit_exclusion_indicators] &&
           selected[2] == [:sum] && selected[3] == [:sum] && selected[4] == [:id]
        return Val(:functional_graph_orbit)
    elseif names == (:Transformation, :Arithmetic, :Aggregation, :Comparison) &&
           selected[1] == [:nonfixed_indicators] && selected[2] == [:sum] &&
           selected[3] == [:sum] && selected[4] == [:condition_residual]
        return Val(:nonfixed_condition)
    elseif names == (:Transformation, :Arithmetic, :Pointwise, :Transformation,
            :Arithmetic, :Aggregation, :Comparison) &&
           selected[1] == [:disjoint_pair_differences] &&
           length(selected[2]) == 1 && only(selected[2]) in (:sum, :product) &&
           selected[3] == [:absolute] && length(selected[4]) == 1 &&
           only(selected[4]) in (:count_equal_left, :count_equal_right) &&
           length(selected[5]) == 1 && only(selected[5]) in (:sum, :product) &&
           selected[6] == [:count_positive] && selected[7] == [:id]
        return Val(:pair_distance_collisions)
    end
    return Val(:generic)
end

function _compose_index_relation(icn::AbstractICN, kernel;
        name, jlfun, fname, dbg)
    parameters = collect(icn.parameters)
    helper = if kernel isa Val{:cyclic}
        :(CompositionalNetworks._cyclic_index_l1)
    elseif kernel isa Val{:indicator}
        :(CompositionalNetworks._indicator_index_l1)
    elseif kernel isa Val{:functional_graph_predecessors}
        :(CompositionalNetworks._functional_graph_predecessor_penalty)
    elseif kernel isa Val{:functional_graph_orbit}
        :(CompositionalNetworks._functional_graph_orbit_penalty)
    elseif kernel isa Val{:nonfixed_condition}
        :(CompositionalNetworks._nonfixed_condition_penalty)
    elseif kernel isa Val{:pair_distance_collisions}
        :(CompositionalNetworks._pair_distance_collision_penalty)
    else
        error("unknown indexed structural kernel: $kernel")
    end
    f = JLFunction()
    f.name = name
    f.args = [:x]
    # Keep the same caller-owned-workspace call contract as the generic compiled path.
    # Indexed fusions do not need `X`, but accepting it lets solver code switch between
    # compiled compositions without signature-dependent branches.
    f.kwargs = Any[Expr(:kw, :X, :nothing), parameters...]
    call = _keyword_call(helper, Any[:x], parameters)
    f.body = Expr(:block, :(return $call))
    generated = codegen_ast(f)
    dbg && @info "fused indexed relation composition" generated
    if !isempty(fname)
        open(fname, "w") do io
            write(io, sprint_expr(f))
        end
    end
    return eval(generated), jlfun ? f : generated
end

function _supports_inplace_compilation(icn::AbstractICN)
    length(icn.layers) == 4 &&
        Tuple(layer.name for layer in icn.layers) ==
        (:Transformation, :Arithmetic, :Aggregation, :Comparison) || return false
    selected = _selected_operation_names(icn)
    transformations = first(selected)
    :index_base in icn.parameters && return false
    :condition_residuals in transformations && return false
    only(selected[3]) === :at && return false
    any(op -> op in (:disjoint_pair_differences, :nonzero, :first_equal_position), transformations) && return false
    only(selected[3]) in (:first_or_zero, :argmin, :argmax) && transformations != [:id] && return false
    return true
end

function _keyword_call(function_name, arguments, parameters)
    return Expr(:call, function_name, Expr(:parameters, parameters...), arguments...)
end

function _compose_inplace(icn::AbstractICN; name, jlfun, fname, dbg)
    selected = _selected_operation_names(icn)
    transformations = selected[1]
    length(transformations) > 0 || error("an ICN transformation layer cannot be empty")
    arithmetic = only(selected[2])
    aggregation = only(selected[3])
    comparison = only(selected[4])
    parameters = collect(icn.parameters)
    columns = length(transformations)
    identity_input = transformations == [:id]
    fused_aggregation = columns == 1 && CompositionalNetworks._supports_fused_aggregation(
        Val(only(transformations)), Val(aggregation))
    fused_pairwise_sum = arithmetic === :sum && aggregation === :sum &&
                         all(CompositionalNetworks._is_pairwise_count, transformations)
    fused_elementwise = all(CompositionalNetworks._is_elementwise, transformations)
    fused_mixed_sum = arithmetic === :sum && aggregation === :sum &&
                      all(
        operation -> CompositionalNetworks._is_pairwise_count(operation) ||
                     CompositionalNetworks._is_elementwise(operation),
        transformations,
    )
    workspace_free = identity_input || fused_aggregation || fused_pairwise_sum ||
                     fused_elementwise || fused_mixed_sum
    body = Expr[]
    if !workspace_free
        push!(body, :(rows = length(x)))
        push!(body,
            quote
            size(X, 1) >= rows && size(X, 2) >= $columns ||
                throw(DimensionMismatch(
                    "composition workspace must have at least $(rows)x$($columns) elements"))
            end)
        for (column, operation) in enumerate(transformations)
            output = :(CompositionalNetworks._WorkspaceColumn(X, $column, rows))
            push!(body,
                _keyword_call(
                    :(CompositionalNetworks._transform!),
                    Any[:(Val($(QuoteNode(operation)))), output, :x],
                    parameters
                ))
        end
    end
    # Sum and product are both the identity over a single selected transformation.
    # Keep its output in column one and avoid a redundant full pass over the input.
    if !workspace_free && columns > 1
        push!(body, :(
            CompositionalNetworks._combine_rows!(
            Val($(QuoteNode(arithmetic))), X, rows, $columns)
        ))
    end
    aggregate = if identity_input
        _keyword_call(
            :(CompositionalNetworks._aggregate_input),
            Any[:(Val($(QuoteNode(aggregation)))), :x],
            parameters,
        )
    elseif fused_aggregation
        _keyword_call(
            :(CompositionalNetworks._aggregate_transform),
            Any[
                :(Val($(QuoteNode(only(transformations))))),
                :(Val($(QuoteNode(aggregation)))),
                :x,
            ],
            parameters,
        )
    elseif fused_pairwise_sum
        _keyword_call(
            :(CompositionalNetworks._aggregate_pairwise_sum),
            Any[:(Val($(QuoteNode(Tuple(transformations))))), :x],
            parameters,
        )
    elseif fused_elementwise
        _keyword_call(
            :(CompositionalNetworks._aggregate_elementwise),
            Any[
                :(Val($(QuoteNode(Tuple(transformations))))),
                :(Val($(QuoteNode(arithmetic)))),
                :(Val($(QuoteNode(aggregation)))),
                :x,
            ],
            parameters,
        )
    elseif fused_mixed_sum
        _keyword_call(
            :(CompositionalNetworks._aggregate_mixed_sum),
            Any[:(Val($(QuoteNode(Tuple(transformations))))), :x],
            parameters,
        )
    else
        _keyword_call(
            :(CompositionalNetworks._aggregate),
            Any[:(Val($(QuoteNode(aggregation)))), :X, :rows],
            parameters,
        )
    end
    push!(body, :(value = $aggregate))
    comparison_call = comparison === :id ? :value : _keyword_call(
        :(CompositionalNetworks._compare),
        Any[:(Val($(QuoteNode(comparison)))), :value],
        parameters,
    )
    push!(body, :(return $comparison_call))

    f = JLFunction()
    f.name = name
    f.args = [:x]
    workspace_default = workspace_free ? :nothing :
                        :(CompositionalNetworks.composition_workspace($columns, x))
    f.kwargs = Any[Expr(:kw, :X, workspace_default), parameters...]
    f.body = Expr(:block, body...)
    generated = codegen_ast(f)
    dbg && @info "in-place composition" generated
    if !isempty(fname)
        open(fname, "w") do io
            write(io, sprint_expr(f))
        end
    end
    return eval(generated), jlfun ? f : generated
end

"""
    compose_values(icn; kwargs...)

Compile one parameterized ICN and return a callable that combines its output
over `vals`. Common operation graphs use fused kernels; every other graph falls
back to lazy scalar composition without materializing the component outputs.
"""
function compose_values(icn::AbstractICN; kwargs...)
    composition = first(compose(icn; kwargs...))
    kernel = _value_kernel(icn)
    return ValueComposition{typeof(composition), typeof(kernel)}(composition)
end

struct ValueComposition{F, K}
    composition::F
end

"""Compiled ICN branches for a matrix-valued `vals` parameter.

The first matrix column selects the value seen by `filter_val`; the remaining columns
are normalized to the `op`/`val` occurrence condition by `_vals_row_condition`. The row
formula is shared by every row and matrix shape. An optional domain formula contributes
only when `bool=true`.
"""
struct MatrixRowsComposition{R, D}
    row::R
    domain::D
end

"""One learned row ICN reduced over a collection-valued parameter.

The wrapper is structural rather than constraint-specific: the parameter name and its
row-shaped runtime form create the branches, while `reduction` expresses their quantifier.
"""
struct ParameterRowsComposition{R, K, O}
    row::R
    kernel::K
    reduction::Symbol
    reduction_op::O
    reduction_val::Int
end


function compose_parameter_rows(
        row_icn::AbstractICN;
        reduction::Symbol = :exists_min,
        reduction_op::F = (==),
        reduction_val::Integer = 0,
        name::Symbol = gensym(:parameter_rows),
) where {F}
    reduction in (:exists_min, :exists_product, :forall_sum, :forall_max, :count) ||
        throw(ArgumentError("unsupported parameter-row reduction: $reduction"))
    row = composition(row_icn; name = Symbol(name, :_row))
    selected = _selected_operation_names(row_icn)
    kernel = length(selected) == 5 &&
             selected[1] == [:aligned_not_equal] &&
             selected[2] == [:id] &&
             length(selected[3]) == 1 &&
             only(selected[3]) in (:sum, :product) &&
             length(selected[4]) == 1 &&
             only(selected[4]) in (:sum, :count_positive) &&
             selected[5] == [:id] ? Val(:aligned_mismatch) : Val(:generic)
    return ParameterRowsComposition(
        row, kernel, reduction, reduction_op, Int(reduction_val),
    )
end


@inline _parameter_row_count(rows::AbstractMatrix) = size(rows, 1)
@inline _parameter_row_count(rows) = length(rows)
@inline _parameter_row_length(rows::AbstractMatrix, row::Int) = size(rows, 2)
@inline _parameter_row_length(rows, row::Int) = length(@inbounds rows[row])
@inline _parameter_row_value(rows::AbstractMatrix, row::Int, column::Int) =
    @inbounds rows[row, column]
@inline _parameter_row_value(rows, row::Int, column::Int) =
    @inbounds rows[row][column]

@inline function _aligned_mismatch_count(values, rows, row::Int)
    _parameter_row_length(rows, row) == length(values) || return -1
    mismatches = 0
    @inbounds for column in eachindex(values)
        mismatches += values[column] != _parameter_row_value(rows, row, column)
    end
    return mismatches
end

@inline function _parameter_rows_aligned_mismatch(
        compiled::ParameterRowsComposition,
        values,
        pair_vars,
)
    rows = _parameter_row_count(pair_vars)
    if compiled.reduction === :exists_min
        minimum_distance = typemax(Int)
        @inbounds for row in 1:rows
            distance = _aligned_mismatch_count(values, pair_vars, row)
            distance < 0 && continue
            minimum_distance = min(minimum_distance, distance)
            iszero(minimum_distance) && return 0.0
        end
        return minimum_distance == typemax(Int) ? 1.0 : Float64(minimum_distance)
    elseif compiled.reduction === :exists_product
        product = 1.0
        found = false
        @inbounds for row in 1:rows
            distance = _aligned_mismatch_count(values, pair_vars, row)
            distance < 0 && continue
            found = true
            product *= distance
            iszero(product) && return 0.0
        end
        return found ? product : 1.0
    elseif compiled.reduction === :forall_sum
        total = 0.0
        @inbounds for row in 1:rows
            distance = _aligned_mismatch_count(values, pair_vars, row)
            distance < 0 || (total += distance)
        end
        return total
    elseif compiled.reduction === :forall_max
        maximum_distance = 0
        @inbounds for row in 1:rows
            distance = _aligned_mismatch_count(values, pair_vars, row)
            distance < 0 || (maximum_distance = max(maximum_distance, distance))
        end
        return Float64(maximum_distance)
    end

    matches = 0
    @inbounds for row in 1:rows
        matches += iszero(_aligned_mismatch_count(values, pair_vars, row))
    end
    return Float64(_count_violation(
        matches, compiled.reduction_op, compiled.reduction_val,
    ))
end


@inline function (compiled::ParameterRowsComposition{R, Val{:aligned_mismatch}})(
        values;
        pair_vars,
        parameters...,
) where {R}
    Base.require_one_based_indexing(values, pair_vars)
    return _parameter_rows_aligned_mismatch(compiled, values, pair_vars)
end


@inline function (compiled::ParameterRowsComposition{R, Val{:generic}})(
        values;
        pair_vars,
        parameters...,
) where {R}
    Base.require_one_based_indexing(values, pair_vars)
    rows = _parameter_row_count(pair_vars)
    row_value = function (row)
        _parameter_row_length(pair_vars, row) == length(values) || return 1.0
        tuple = pair_vars isa AbstractMatrix ? @view(pair_vars[row, :]) : pair_vars[row]
        return compiled.row(values; pair_vars = tuple, parameters...)
    end
    return reduce_icn_outputs(
        row_value, 1:rows, compiled.reduction;
        op = compiled.reduction_op,
        val = compiled.reduction_val,
    )
end


function canonical_key(compiled::ParameterRowsComposition; simplified::Bool = true)
    row = canonical_key(compiled.row; simplified)
    return "parameter_rows[reduction=$(compiled.reduction);op=$(compiled.reduction_op);" *
           "val=$(compiled.reduction_val);row={$row}]"
end


function code(compiled::ParameterRowsComposition, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    language === :maths || throw(ArgumentError(
        "ParameterRowsComposition currently supports interpretable :maths output",
    ))
    row = _math_expression(compiled.row.ir; simplified)
    return "$(name)(x; pair_vars) = reduce_rows($row; " *
           "reduction=$(compiled.reduction), op=$(compiled.reduction_op), " *
           "val=$(compiled.reduction_val))"
end


@testitem "Parameter rows compose numeric and symbolic tuple relations" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @test, @test_throws

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
        @test CN.check_weights_validity(network, network.weights)
        return network
    end

    layers = [CN.PairedMap, CN.Transformation, CN.Arithmetic,
        CN.Aggregation, CN.Comparison]
    row = CN.ICN(;
        parameters = [:pair_vars],
        parameter_values = (; pair_vars = [1, 2, 3]),
        layers,
        connection = UInt32.(eachindex(layers)),
    )
    select!(row, (
        [:aligned_not_equal], [:id], [:sum], [:sum], [:id],
    ))
    supports = CN.compose_parameter_rows(row; reduction = :exists_min)
    conflicts = CN.compose_parameter_rows(
        row; reduction = :count, reduction_op = (==), reduction_val = 0,
    )

    numeric_rows = [[1, 2, 3], [3, 2, 1]]
    @test supports([1, 7, 3]; pair_vars = numeric_rows) == 1.0
    @test supports([3, 2, 1]; pair_vars = numeric_rows) == 0.0
    @test conflicts([1, 7, 3]; pair_vars = numeric_rows) == 0.0
    @test conflicts([3, 2, 1]; pair_vars = numeric_rows) == 1.0
    @test supports([1, 2]; pair_vars = [[1], [1, 2, 3]]) == 1.0

    # No numeric operation is selected by this learned formula. Any domain that
    # defines equality therefore inherits it without retraining.
    symbolic_rows = [["red", "blue"], ["blue", "red"]]
    @test supports(["red", "blue"]; pair_vars = symbolic_rows) == 0.0
    @test conflicts(["green", "blue"]; pair_vars = symbolic_rows) == 0.0
    @test conflicts(["blue", "red"]; pair_vars = symbolic_rows) == 1.0

    numeric_matrix = [1 2 3; 3 2 1]
    @test supports([3, 2, 1]; pair_vars = numeric_matrix) == 0.0
    @test occursin("aligned_not_equal", CN.code(supports, :maths))
    @test CN.canonical_key(supports) != CN.canonical_key(conflicts)
    @test_throws ArgumentError CN.compose_parameter_rows(row; reduction = :median)

    function allocations(compiled, values, rows)
        compiled(values; pair_vars = rows)
        return @allocated compiled(values; pair_vars = rows)
    end
    @test allocations(supports, [1, 7, 3], numeric_rows) == 0
    @test allocations(conflicts, [1, 7, 3], numeric_rows) == 0

    state = CN.incremental_state(supports, [1, 7, 3]; pair_vars = numeric_rows)
    @test CN.incremental_value(state) == 1.0
    @test CN.incremental_update!(state, 2, 2) == 0.0
    @test state.values == [1, 2, 3]
    symbolic_state = CN.incremental_state(
        conflicts, ["green", "blue"]; pair_vars = symbolic_rows,
    )
    @test CN.incremental_update!(symbolic_state, 1, "red") == 1.0
    function incremental_allocations(state, position, replacement)
        old = state.values[position]
        CN.incremental_update!(state, position, replacement)
        CN.incremental_update!(state, position, old)
        return @allocated begin
            CN.incremental_update!(state, position, replacement)
            CN.incremental_update!(state, position, old)
        end
    end
    @test incremental_allocations(state, 2, 7) == 0
end

function compose_matrix_rows(
        row_icn::AbstractICN,
        domain_icn::Union{Nothing, AbstractICN} = nothing;
        name::Symbol = gensym(:matrix_rows),
)
    _selected_operation_names(row_icn) == [
        [:filter_equal_filter_val], [:id], [:sum],
        [:count_elements], [:condition_residual],
    ] || throw(ArgumentError(
        "matrix-row compilation requires an occurrence-count residual composition",
    ))
    if !isnothing(domain_icn)
        _selected_operation_names(domain_icn) == [
            [:filter_ne_vals], [:id], [:sum], [:count_elements], [:id],
        ] || throw(ArgumentError(
            "matrix-domain compilation requires an outside-membership count composition",
        ))
    end
    row = composition(row_icn; name = Symbol(name, :_row))
    domain = isnothing(domain_icn) ? nothing :
             composition(domain_icn; name = Symbol(name, :_domain))
    return MatrixRowsComposition(row, domain)
end

@inline function (compiled::MatrixRowsComposition)(
        x;
        vals::AbstractMatrix,
        bool::Bool = false,
        dom_size = length(x),
        numvars::Integer = length(x),
)
    Base.require_one_based_indexing(x, vals)
    size(vals, 2) >= 1 || throw(ArgumentError(
        "matrix-valued vals requires at least one column",
    ))
    total = if size(vals, 2) == 1
        _matrix_rows_default_occurrence(compiled, x, vals, dom_size, numvars)
    elseif size(vals, 2) == 2
        _matrix_rows_exact_occurrence(compiled, x, vals, dom_size, numvars)
    else
        _matrix_rows_range_occurrence(compiled, x, vals, dom_size, numvars)
    end
    return total + _matrix_rows_domain_cost(compiled, x, vals, bool)
end

@inline function _matrix_rows_default_occurrence(
        compiled, x, vals, dom_size, numvars)
    total = 0.0
    @inbounds for row in axes(vals, 1)
        total += abs(_matrix_occurrence_count(x, vals[row, 1]) - 1)
    end
    return total
end

@inline function _matrix_rows_exact_occurrence(compiled, x, vals, dom_size, numvars)
    total = 0.0
    @inbounds for row in axes(vals, 1)
        total += abs(_matrix_occurrence_count(x, vals[row, 1]) - vals[row, 2])
    end
    return total
end


@inline function _matrix_rows_range_occurrence(compiled, x, vals, dom_size, numvars)
    total = 0.0
    @inbounds for row in axes(vals, 1)
        lower = vals[row, 2]
        upper = vals[row, 3]
        count = _matrix_occurrence_count(x, vals[row, 1])
        minimum_count = min(lower, upper)
        maximum_count = max(lower, upper)
        total += count < minimum_count ? minimum_count - count :
                 count > maximum_count ? count - maximum_count : zero(count)
    end
    return total
end

@inline function _matrix_occurrence_count(x, target)
    count = 0
    @inbounds for value in x
        count += value == target
    end
    return count
end

@inline function _matrix_rows_domain_cost(compiled, x, vals, bool::Bool)
    bool || return 0.0
    isnothing(compiled.domain) && throw(ArgumentError(
        "bool=true requires a compiled domain-membership branch",
    ))
    outside = 0
    @inbounds for value in x
        found = false
        for row in axes(vals, 1)
            if value == vals[row, 1]
                found = true
                break
            end
        end
        outside += !found
    end
    return Float64(outside)
end

function canonical_key(compiled::MatrixRowsComposition; simplified::Bool = true)
    row = canonical_key(compiled.row; simplified)
    domain = isnothing(compiled.domain) ? "none" :
             canonical_key(compiled.domain; simplified)
    return "matrix_vals_rows[row={$row};domain={$domain}]"
end

function code(compiled::MatrixRowsComposition, language::Symbol = :maths;
        name = "composition", simplified::Bool = true)
    language === :maths || throw(ArgumentError(
        "MatrixRowsComposition currently supports interpretable :maths output",
    ))
    row = _math_expression(compiled.row.ir; simplified)
    domain = isnothing(compiled.domain) ? "0" :
             _math_expression(compiled.domain.ir; simplified)
    return "$(name)(x; vals, bool) = " *
           "sum_rows($row; filter_val=vals[r,1], (op,val)=occurs(vals,r)) + " *
           "bool * ($domain; vals=vals[:,1])"
end

@testitem "Matrix vals rows share one exact compiled composition" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @test, @test_throws

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
        @test CN.check_weights_validity(network, network.weights)
        return network
    end

    layers = [CN.SimpleFilter, CN.Transformation, CN.Arithmetic,
        CN.Aggregation, CN.Comparison]
    row = CN.ICN(;
        parameters = [:dom_size, :numvars, :filter_val, :op, :val],
        parameter_values = (;
            dom_size = 5, numvars = 4, filter_val = 1, op = in, val = 0:2,
        ),
        layers,
        connection = UInt32.(eachindex(layers)),
    )
    select!(row, (
        [:filter_equal_filter_val], [:id], [:sum],
        [:count_elements], [:condition_residual],
    ))
    domain = CN.ICN(;
        parameters = [:vals],
        parameter_values = (; vals = [1, 2]),
        layers,
        connection = UInt32.(eachindex(layers)),
    )
    select!(domain, (
        [:filter_ne_vals], [:id], [:sum], [:count_elements], [:id],
    ))
    compiled = CN.compose_matrix_rows(row, domain; name = :matrix_rows_test)

    function reference(values, parameters, closed)
        constrained = @view parameters[:, 1]
        total = closed ? count(value -> !(value in constrained), values) : 0
        for row_index in axes(parameters, 1)
            occurrences = count(==(parameters[row_index, 1]), values)
            if size(parameters, 2) == 1
                total += abs(occurrences - 1)
            elseif size(parameters, 2) == 2
                total += abs(occurrences - parameters[row_index, 2])
            else
                lower = min(parameters[row_index, 2], parameters[row_index, 3])
                upper = max(parameters[row_index, 2], parameters[row_index, 3])
                total += occurrences < lower ? lower - occurrences :
                         occurrences > upper ? occurrences - upper : 0
            end
        end
        return Float64(total)
    end

    values = [2, 5, 5, 8]
    matrices = (
        reshape([2, 5, 10], 3, 1),
        [2 1; 5 2; 10 0],
        [2 0 1; 5 1 3; 10 2 3],
        [2 1 0 99; 5 3 1 -4; 10 3 2 42],
    )
    for parameters in matrices, closed in (false, true)
        @test compiled(values; vals = parameters, bool = closed) ==
              reference(values, parameters, closed)
    end
    @test_throws ArgumentError CN.compose_matrix_rows(row)(
        values; vals = matrices[1], bool = true,
    )
    @test occursin("sum_rows", CN.code(compiled, :maths; name = "cardinality"))
    @test occursin("filter_equal_filter_val", CN.code(compiled, :maths))

    function allocations(compiled, values, parameters)
        compiled(values; vals = parameters, bool = true)
        return @allocated compiled(values; vals = parameters, bool = true)
    end
    @test allocations(compiled, values, matrices[3]) == 0

    state = CN.incremental_state(
        compiled, values; vals = matrices[3], bool = true,
    )
    @test CN.incremental_value(state) == compiled(
        values; vals = matrices[3], bool = true,
    )
    for (position, replacement) in ((1, 10), (4, 2), (2, 8), (3, 5))
        @test CN.incremental_update!(state, position, replacement) == compiled(
            state.values; vals = matrices[3], bool = true,
        )
    end
    function incremental_allocations(state, position, replacement)
        old_value = state.values[position]
        CN.incremental_update!(state, position, replacement)
        CN.incremental_update!(state, position, old_value)
        return @allocated begin
            CN.incremental_update!(state, position, replacement)
            CN.incremental_update!(state, position, old_value)
        end
    end
    @test incremental_allocations(state, 1, 2) == 0
end

function _value_kernel(icn::AbstractICN)
    _supports_inplace_compilation(icn) || return Val(:generic)
    selected = _selected_operation_names(icn)
    transformations = selected[1]
    if length(transformations) == 2 &&
       Set(transformations) == Set((:var_minus_val, :val_minus_var)) &&
       selected[2] == [:sum] && selected[3] == [:minimum] && selected[4] == [:id]
        return Val(:absolute_minimum)
    end
    return Val(:generic)
end

@testitem "Value-parameter compositions fuse reductions" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @test

    function select!(network, operations)
        fill!(network.weights.parent, false)
        offset = 0
        for (layer, selected) in zip(network.layers, operations)
            names = collect(keys(layer.fn))
            for operation in selected
                index = only(findall(==(operation), names))
                network.weights.parent[offset + index] = true
            end
            offset += length(layer.fn)
        end
        @test CN.check_weights_validity(network, network.weights)
        return network
    end

    network = CN.ICN(parameters = [:val, :dom_size, :numvars])
    select!(network, (
        [:var_minus_val, :val_minus_var], [:sum], [:minimum], [:id],
    ))
    compiled = CN.compose_values(network; name = :absolute_minimum_values_test)
    input = [0, 3, 7]
    vals = (1, 3, 5)
    @test compiled(input; vals, reduction = :exists_min) == 0
    @test compiled(input; vals, reduction = :exists_product) == 0
    @test compiled(input; vals, reduction = :forall_max) == 2
    @test compiled(input; vals, reduction = :forall_sum) == 3
    @test compiled(input; vals, reduction = :mean) == 1
    @test compiled(
        input;
        vals,
        reduction = :count,
        reduction_op = (>=),
        reduction_val = 2,
    ) == 1
    @test CN.icn_zero_set(compiled, input; vals, reduction = :exists_min)
    @test !CN.icn_zero_set(compiled, input; vals, reduction = :forall_sum)
    @test !CN.icn_zero_set(
        compiled,
        input;
        vals,
        reduction = :count,
        reduction_op = (>=),
        reduction_val = 2,
    )

    function allocations(compiled, input, vals)
        compiled(input; vals, reduction = :forall_sum)
        return @allocated compiled(input; vals, reduction = :forall_sum)
    end
    @test allocations(compiled, input, vals) == 0

    fallback_network = CN.ICN(parameters = [:val, :dom_size, :numvars])
    select!(fallback_network, ([:id], [:sum], [:sum], [:id]))
    fallback = CN.compose_values(fallback_network; name = :generic_values_test)
    @test fallback(
        input;
        vals,
        reduction = :forall_sum,
        dom_size = 8,
        numvars = 3,
    ) == 30
end

@inline function (compiled::ValueComposition{F, Val{:generic}})(
        x;
        vals,
        reduction::Symbol = :mean,
        reduction_op::Function = (==),
        reduction_val::Integer = 0,
        reduction_weights = nothing,
        parameters...,
) where {F}
    return reduce_icn_outputs(
        value -> compiled.composition(x; val = value, parameters...),
        vals,
        reduction;
        op = reduction_op,
        val = reduction_val,
        weights = reduction_weights,
    )
end

@inline function (compiled::ValueComposition{F, Val{:absolute_minimum}})(
        x;
        vals,
        reduction::Symbol = :mean,
        reduction_op::Function = (==),
        reduction_val::Integer = 0,
        reduction_weights = nothing,
        parameters...,
) where {F}
    return _reduce_absolute_minimum_values(
        x,
        vals,
        reduction;
        op = reduction_op,
        val = reduction_val,
        weights = reduction_weights,
    )
end

@inline function icn_zero_set(
        compiled::ValueComposition{F, Val{:generic}},
        x;
        vals,
        reduction::Symbol = :mean,
        reduction_op::Function = (==),
        reduction_val::Integer = 0,
        reduction_weights = nothing,
        parameters...,
) where {F}
    return icn_zero_set(
        value -> compiled.composition(x; val = value, parameters...),
        vals,
        reduction;
        op = reduction_op,
        val = reduction_val,
        weights = reduction_weights,
    )
end

@inline function icn_zero_set(
        compiled::ValueComposition{F, Val{:absolute_minimum}},
        x;
        vals,
        reduction::Symbol = :mean,
        reduction_op::Function = (==),
        reduction_val::Integer = 0,
        reduction_weights = nothing,
        parameters...,
) where {F}
    return icn_zero_set(
        value -> _absolute_minimum(x, value),
        vals,
        reduction;
        op = reduction_op,
        val = reduction_val,
        weights = reduction_weights,
    )
end

@testitem "Specialized front layers preserve compiled semantics" default_imports=false begin
    import CompositionalNetworks as CN
    import ConstraintCommons
    import Test: @test, @testset

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
        @test CN.check_weights_validity(network, network.weights)
        return network
    end

    values = [0, 1, 2, 3, 4]
    cases = (
        filter = (
            [CN.SimpleFilter, CN.Transformation, CN.Arithmetic, CN.Aggregation, CN.Comparison],
            ([:filter_ne_vals], [:id], [:sum], [:sum], [:id]),
            [:vals],
            (; vals = (1, 3)),
            6.0,
        ),
        paired = (
            [CN.PairedMap, CN.Transformation, CN.Arithmetic, CN.Aggregation, CN.Comparison],
            ([:sub], [:id], [:sum], [:sum], [:id]),
            [:pair_vars],
            (; pair_vars = [0, 2, 2, 2, 5]),
            3.0,
        ),
        language = (
            [CN.Language, CN.Transformation, CN.Arithmetic, CN.Aggregation, CN.Comparison],
            ([:accept], [:id], [:sum], [:sum], [:id]),
            [:language],
            (;
                language = ConstraintCommons.Automaton(
                    Dict((:zero, 0) => :zero, (:zero, 1) => :one,
                        (:one, 0) => :zero, (:one, 1) => :one),
                    :zero,
                    :zero,
                ),
            ),
            1.0,
        ),
    )

    @testset "$name" for (name, (layers, operations, parameters, runtime, expected)) in
                             pairs(cases)
        network = CN.ICN(; layers, parameters, connection = UInt32.(eachindex(layers)))
        select!(network, operations)
        @test CN._supports_specialized_compilation(network)
        compiled = first(CN.compose(network; name = gensym(name)))
        @test Base.invokelatest(compiled, values; runtime...) == expected
        @test CN.evaluate(network, CN.Solution(values); runtime...) == expected
    end
end

@testitem "Composed pairwise disjunction is exact and allocation free" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @test

    function select!(network, operations)
        fill!(network.weights.parent, false)
        offset = 0
        for (layer, selected) in zip(network.layers, operations)
            names = collect(keys(layer.fn))
            for operation in selected
                index = only(findall(==(operation), names))
                network.weights.parent[offset + index] = true
            end
            offset += length(layer.fn)
        end
        @test CN.check_weights_validity(network, network.weights)
        return network
    end

    layers = [
        CN.PairedMap, CN.PairMask, CN.GroupReduction, CN.Transformation,
        CN.Arithmetic, CN.Aggregation, CN.Comparison,
    ]
    network = CN.ICN(;
        parameters = [:dom_size, :numvars, :pair_vars, :dim, :bool],
        layers = layers,
        connection = UInt32.(eachindex(layers)),
    )
    select!(network, (
        [:pairwise_oriented_affine_margins], [:zero_extent_groups], [:minimum],
        [:positive_part], [:sum], [:sum], [:id],
    ))
    compiled = first(CN.compose(network; name = :learned_pairwise_disjunction))
    origins = [0, 1, 4]
    lengths = [2, 2, 1]
    parameters = (; pair_vars = lengths, dim = 1, bool = true,
        numvars = 3, dom_size = 5)
    @test compiled(origins; parameters...) == 1.0
    @test CN.evaluate(network, CN.Solution(origins); parameters...) == 1.0

    # Grammar operations expose reusable affine/disjunction machinery. No
    # layer operation is allowed to decide the target constraint directly.
    @test all(
        name -> !occursin("overlap", lowercase(String(name))),
        Iterators.flatten(keys(layer.fn) for layer in layers),
    )
    @test all(
        operation -> !occursin("overlap", lowercase(string(CN.codegen_ast(operation)))),
        Iterators.flatten(layer.fnexprs for layer in layers),
    )

    function reference_no_overlap(origins, lengths, dimensions, zero_ignored)
        tasks = length(origins) ÷ dimensions
        for first_task in 1:(tasks - 1), second_task in (first_task + 1):tasks
            first_offset = (first_task - 1) * dimensions
            second_offset = (second_task - 1) * dimensions
            if zero_ignored && any(1:dimensions) do dimension
                    iszero(lengths[first_offset + dimension]) ||
                        iszero(lengths[second_offset + dimension])
                end
                continue
            end
            separated = any(1:dimensions) do dimension
                first_index = first_offset + dimension
                second_index = second_offset + dimension
                origins[first_index] + lengths[first_index] <= origins[second_index] ||
                    origins[second_index] + lengths[second_index] <= origins[first_index]
            end
            separated || return false
        end
        return true
    end

    exact_zero_set = Ref(true)
    for dimensions in 1:4, tasks in 2:3
        coordinates = tasks * dimensions
        # Keep the exhaustive guard compact for the largest multidimensional case.
        domain = coordinates <= 8 ? (0:2) : (0:1)
        length_patterns = (
            ones(Int, coordinates),
            [1 + mod(index, 3) for index in 1:coordinates],
            [isone(index) ? 0 : 2 for index in 1:coordinates],
        )
        for tuple in Iterators.product(ntuple(_ -> domain, coordinates)...)
            candidate = collect(tuple)
            for task_lengths in length_patterns, zero_ignored in (false, true)
                penalty = compiled(
                    candidate;
                    pair_vars = task_lengths,
                    dim = dimensions,
                    bool = zero_ignored,
                    numvars = tasks,
                    dom_size = length(domain),
                )
                exact_zero_set[] &= iszero(penalty) == reference_no_overlap(
                    candidate, task_lengths, dimensions, zero_ignored,
                )
                exact_zero_set[] || break
            end
            exact_zero_set[] || break
        end
        exact_zero_set[] || break
    end
    @test exact_zero_set[]

    function allocations(compiled, origins, lengths)
        compiled(
            origins;
            pair_vars = lengths,
            dim = 1,
            bool = true,
            numvars = 3,
            dom_size = 5,
        )
        return @allocated compiled(
            origins;
            pair_vars = lengths,
            dim = 1,
            bool = true,
            numvars = 3,
            dom_size = 5,
        )
    end
    @test allocations(compiled, origins, lengths) == 0

end

@testitem "Adjacent affine margins stay generic and allocation free" default_imports=false begin
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
        @test CN.check_weights_validity(network, network.weights)
        return network
    end

    layers = [CN.PairedMap, CN.Transformation, CN.Arithmetic,
        CN.Aggregation, CN.Comparison]
    network = CN.ICN(;
        parameters = [:pair_vars, :op],
        parameter_values = (; pair_vars = [2, 1, 3, 99], op = (<=)),
        layers,
        connection = UInt32.(eachindex(layers)),
    )
    select!(network, (
        [:adjacent_left_affine_margins], [:id], [:sum], [:count_positive], [:id],
    ))
    compiled = first(CN.compose(network; name = :generic_adjacent_affine_test))
    values = [1, 4, 5, 9]
    pair_vars = [2, 1, 3, 99]
    penalty = compiled(values; pair_vars, op = (<=))
    expected = all(values[i] + pair_vars[i] <= values[i + 1]
        for i in 1:(length(values) - 1))
    @test iszero(penalty) == expected
    @test CN.code(network, :maths) ==
          "composition(x) = count_positive(adjacent_left_affine_margins(x))"

    function allocations(compiled, values, pair_vars)
        compiled(values; pair_vars, op = (<=))
        return @allocated compiled(values; pair_vars, op = (<=))
    end
    @test allocations(compiled, values, pair_vars) == 0
    @test all(name -> !occursin("ordered", lowercase(String(name))) &&
                     !occursin("slide", lowercase(String(name))),
        Iterators.flatten(keys(layer.fn) for layer in layers))
end

@testitem "Distinct counts compile from atomic ICN operations" default_imports=false begin
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
        @test CN.check_weights_validity(network, network.weights)
        return network
    end

    plain = select!(CN.ICN(
        parameters = [:op, :val],
        parameter_values = (; op = (==), val = 3),
    ), ([:count_equal_right], [:sum], [:count_zero], [:condition_residual]))
    plain_compiled = first(CN.compose(plain; name = :distinct_plain_test))

    layers = [CN.SimpleFilter, CN.Transformation, CN.Arithmetic,
        CN.Aggregation, CN.Comparison]
    except = select!(CN.ICN(;
        parameters = [:vals, :op, :val],
        parameter_values = (; vals = [0], op = (==), val = 2),
        layers,
        connection = UInt32.(eachindex(layers)),
    ), ([:filter_ne_vals], [:count_equal_right], [:sum],
        [:count_zero], [:condition_residual]))
    except_compiled = first(CN.compose(except; name = :distinct_except_test))

    values = [0, 1, 1, 2]
    @test plain_compiled(values; op = (==), val = 3) == 0.0
    @test except_compiled(values; vals = [0], op = (==), val = 2) == 0.0
    @test CN.evaluate(plain, CN.Solution(values); op = (==), val = 3) == 0.0
    @test CN.evaluate(except, CN.Solution(values);
        vals = [0], op = (==), val = 2) == 0.0
    @test CN._supports_specialized_compilation(except)

    function plain_allocations(compiled, input)
        compiled(input; op = (==), val = 3)
        return @allocated compiled(input; op = (==), val = 3)
    end
    function except_allocations(compiled, input, vals)
        compiled(input; vals, op = (==), val = 2)
        return @allocated compiled(input; vals, op = (==), val = 2)
    end
    @test plain_allocations(plain_compiled, values) == 0
    @test except_allocations(except_compiled, values, [0]) == 0
end

function _specialized_compilation_plan(icn::AbstractICN)
    names = Tuple(layer.name for layer in icn.layers)
    selected = _selected_operation_names(icn)
    if names == (:PairedMap, :PairMask, :GroupReduction, :Transformation,
            :Arithmetic, :Aggregation, :Comparison) &&
       selected[1] == [:pairwise_oriented_affine_margins] &&
       selected[2] == [:zero_extent_groups] && selected[3] == [:minimum] &&
       selected[5] == [:sum] &&
       length(selected[4]) == 1 && length(selected[6]) == 1 &&
       selected[7] == [:id]
        transformation = only(selected[4])
        aggregation = only(selected[6])
        (transformation === :positive_part ||
         (transformation === :id && aggregation === :count_positive)) || return nothing
        return (
            front = :PairedMap,
            operation = :pairwise_oriented_affine_margins,
            transformation = :grouped_min_positive,
            aggregation,
            comparison = :id,
        )
    elseif names == (:EventMap, :SegmentMap, :Arithmetic, :Aggregation, :Comparison) &&
           selected[1] == [:weighted_interval_segments]
        if selected[2] == [:loads] && selected[3] == [:sum] &&
           selected[4] == [:maximum] && length(selected[5]) == 1
            return (
                front = :EventMap,
                operation = :weighted_interval_profile,
                transformation = :id,
                aggregation = :maximum,
                comparison = only(selected[5]),
            )
        elseif Set(selected[2]) == Set((:widths, :condition_residuals)) &&
               selected[3] == [:product] && selected[4] == [:sum] &&
               selected[5] == [:id]
            return (
                front = :EventMap,
                operation = :weighted_interval_segments,
                transformation = :segment_condition_area,
                aggregation = :sum,
                comparison = :id,
            )
        end
        return nothing
    elseif names != (:SimpleFilter, :Transformation, :Arithmetic, :Aggregation,
            :Comparison) &&
           names != (:PairedMap, :Transformation, :Arithmetic, :Aggregation,
            :Comparison) &&
           names != (:Language, :Transformation, :Arithmetic, :Aggregation,
            :Comparison)
        return nothing
    end
    front_name = first(names)
    front_name === :PairedMap && selected[1] == [:gather] && return nothing
    length(selected[1]) == 1 || return nothing
    filtered_distinct = front_name === :SimpleFilter &&
                        length(selected[2]) == 1 &&
                        only(selected[2]) in (:count_equal_left, :count_equal_right) &&
                        selected[3] == [:sum] && selected[4] == [:count_zero]
    valid_transformation = selected[2] == [:id] || filtered_distinct ||
                           (front_name === :PairedMap &&
                            selected[1] == [:aligned_difference] &&
                            selected[2] == [:absolute])
    valid_transformation || return nothing
    length(selected[3]) == length(selected[4]) == length(selected[5]) == 1 ||
        return nothing
    only(selected[4]) in (:first_or_zero, :argmin, :argmax, :at) && return nothing
    return (
        front = front_name,
        operation = only(selected[1]),
        transformation = only(selected[2]),
        aggregation = only(selected[4]),
        comparison = only(selected[5]),
    )
end

_supports_specialized_compilation(icn::AbstractICN) =
    !isnothing(_specialized_compilation_plan(icn))

function _compose_specialized(icn::AbstractICN; name, jlfun, fname, dbg)
    plan = something(_specialized_compilation_plan(icn))
    front = plan.front
    front_operation = plan.operation
    transformation = plan.transformation
    aggregation = plan.aggregation
    comparison = plan.comparison
    parameters = collect(icn.parameters)
    aggregate_parameters = if front === :Language
        [parameters; :language_workspace]
    elseif front === :EventMap
        [parameters; :X]
    else
        parameters
    end
    aggregate = _keyword_call(
        :(CompositionalNetworks._aggregate_specialized),
        Any[
            :(Val($(QuoteNode(front)))),
            :(Val($(QuoteNode(front_operation)))),
            :(Val($(QuoteNode(transformation)))),
            :(Val($(QuoteNode(aggregation)))),
            :x,
        ],
        aggregate_parameters,
    )
    comparison_call = comparison === :id ? :value : _keyword_call(
        :(CompositionalNetworks._compare),
        Any[:(Val($(QuoteNode(comparison)))), :value],
        parameters,
    )
    f = JLFunction()
    f.name = name
    f.args = [:x]
    f.kwargs = Any[Expr(:kw, :X, :nothing), parameters...]
    front === :Language && push!(f.kwargs, Expr(:kw, :language_workspace, :nothing))
    f.body = Expr(:block, :(value = $aggregate), :(return $comparison_call))
    generated = codegen_ast(f)
    dbg && @info "specialized composition" generated
    if !isempty(fname)
        open(fname, "w") do io
            write(io, sprint_expr(f))
        end
    end
    return eval(generated), jlfun ? f : generated
end

@testitem "Published ICN penalties are reconstructible" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @test, @testset

    function select!(network, operations)
        fill!(network.weights.parent, false)
        offset = 0
        for (layer, selected) in zip(network.layers, operations)
            names = collect(keys(layer.fn))
            for operation in selected
                index = findfirst(==(operation), names)
                isnothing(index) && error("unknown operation $operation in $(layer.name)")
                network.weights.parent[offset + index] = true
            end
            offset += length(layer.fn)
        end
        @test CN.check_weights_validity(network, network.weights)
        return network
    end

    left_equal(x, i) = count(j -> x[j] == x[i], firstindex(x):(i - 1))
    right_equal(x, i) = count(j -> x[j] == x[i], (i + 1):lastindex(x))
    less_right(x, i) = count(j -> x[j] < x[i], (i + 1):lastindex(x))
    less_than_shift(x, i, parameter) =
        count(j -> j != i && x[j] < x[i] + parameter, eachindex(x))
    greater_than_shift(x, i, parameter) =
        count(j -> j != i && x[j] > x[i] + parameter, eachindex(x))
    bounded_shift(x, i, parameter) =
        count(j -> j != i && x[i] <= x[j] <= x[i] + parameter, eachindex(x))
    forward_gap(x, i) = i == lastindex(x) ? 0 : max(0, x[i] - x[i + 1])
    reverse_gap(x, i) = i == lastindex(x) ? 0 : max(0, x[i + 1] - x[i])

    published = (
        complete_all_different = (
            ([:count_equal_left], [:sum], [:count_positive], [:id]),
            (x, p, n) -> count(i -> left_equal(x, i) > 0, eachindex(x)),
        ),
        complete_linear_sum = (
            ([:id], [:sum], [:sum], [:abs_val]),
            (x, p, n) -> abs(sum(x) - p),
        ),
        complete_linear_less_than = (
            ([:id], [:sum], [:sum], [:var_minus_val]),
            (x, p, n) -> max(0, sum(x) - p),
        ),
        complete_linear_greater_than = (
            ([:id], [:sum], [:sum], [:val_minus_var]),
            (x, p, n) -> max(0, p - sum(x)),
        ),
        complete_minimum = (
            ([:val_minus_var], [:sum], [:count_positive], [:id]),
            (x, p, n) -> count(value -> max(0, p - value) > 0, x),
        ),
        complete_no_overlap_1d = (
            ([:count_equal_left, :count_less_val], [:sum], [:sum],
                [:max_var_minus_numvars]),
            (x, p, n) -> max(0,
                sum(left_equal(x, i) + less_than_shift(x, i, p) for i in eachindex(x)) - n),
        ),
        complete_ordered = (
            ([:contiguous_vars_minus], [:sum], [:sum], [:id]),
            (x, p, n) -> sum(forward_gap(x, i) for i in eachindex(x)),
        ),
        incomplete_all_different = (
            ([:count_equal_left, :count_equal_right], [:sum], [:count_positive], [:id]),
            (x, p, n) -> count(
                i -> left_equal(x, i) + right_equal(x, i) > 0, eachindex(x)),
        ),
        incomplete_linear_sum = (
            ([:id], [:sum], [:sum], [:abs_val]),
            (x, p, n) -> abs(sum(x) - p),
        ),
        incomplete_linear_less_than = (
            ([:id], [:sum], [:sum], [:var_minus_val]),
            (x, p, n) -> max(0, sum(x) - p),
        ),
        incomplete_linear_greater_than = (
            ([:id, :contiguous_vars_minus_rev], [:product], [:sum], [:val_minus_var]),
            (x, p, n) -> max(0,
                p - sum(x[i] * reverse_gap(x, i) for i in eachindex(x))),
        ),
        incomplete_minimum = (
            ([:count_great_val, :val_minus_var], [:sum], [:sum], [:var_minus_val]),
            (x, p, n) -> max(0,
                sum(greater_than_shift(x, i, p) + max(0, p - x[i]) for i in eachindex(x)) - p),
        ),
        incomplete_no_overlap_1d = (
            ([:count_bounding_val, :count_less_val], [:sum], [:sum], [:var_minus_val]),
            (x, p, n) -> max(0,
                sum(bounded_shift(x, i, p) + less_than_shift(x, i, p) for i in eachindex(x)) - p),
        ),
        incomplete_ordered = (
            ([:count_less_right], [:sum], [:sum], [:id]),
            (x, p, n) -> sum(less_right(x, i) for i in eachindex(x)),
        ),
    )

    network = CN.ICN(parameters = [:dom_size, :numvars, :val])
    assignments =
        [collect(values) for values in Iterators.product(ntuple(_ -> 0:2, 4)...)]
    @testset "$name" for (name, (operations, reference)) in pairs(published)
        select!(network, operations)
        compiled = first(CN.compose(network; name = gensym(name)))
        workspace = CN.composition_workspace(network, 4)
        for x in assignments
            parameters = (; val = 2, numvars = length(x), dom_size = 3)
            expected = Float64(reference(x, parameters.val, parameters.numvars))
            @test CN.evaluate(network, CN.Solution(x); parameters...) == expected
            @test Base.invokelatest(compiled, x; X = workspace, parameters...) == expected
        end
    end
end
