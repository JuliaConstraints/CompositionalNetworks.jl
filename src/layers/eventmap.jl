"""One maximal constant-load interval produced by an event sweep."""
struct WeightedIntervalSegment{L, W}
    load::L
    width::W
end

function _check_weighted_interval_arguments(values, pair_vars)
    pair_vars isa AbstractMatrix || throw(ArgumentError(
        "a weighted interval profile requires matrix-valued pair_vars",
    ))
    size(pair_vars, 1) == 2 || throw(DimensionMismatch(
        "a weighted interval profile requires one extent row and one weight row",
    ))
    size(pair_vars, 2) == length(values) || throw(DimensionMismatch(
        "weighted intervals require one parameter column per origin",
    ))
    Base.require_one_based_indexing(values, pair_vars)

    lengths = @view pair_vars[1, :]
    heights = @view pair_vars[2, :]
    all(length -> length >= zero(length), lengths) || throw(ArgumentError(
        "weighted interval extents must be non-negative",
    ))
    return nothing
end

function _weighted_interval_profile(values, pair_vars)
    _check_weighted_interval_arguments(values, pair_vars)

    lengths = @view pair_vars[1, :]
    heights = @view pair_vars[2, :]

    time_type = promote_type(eltype(values), eltype(lengths))
    load_type = promote_type(eltype(heights), Int)
    events = Vector{Tuple{time_type, load_type}}(undef, 2length(values))
    @inbounds for task in eachindex(values)
        event = 2task - 1
        start = convert(time_type, values[task])
        height = convert(load_type, heights[task])
        events[event] = (start, height)
        events[event + 1] = (start + lengths[task], -height)
    end
    sort!(events)

    # One load per maximal positive-width interval, plus the final zero load.
    # Simultaneous starts and ends are grouped before the load is observed.
    profile = Vector{load_type}(undef, max(1, length(events)))
    isempty(events) && (profile[1] = zero(load_type); return profile)
    usage = zero(load_type)
    event = firstindex(events)
    output = 0
    @inbounds while event <= lastindex(events)
        time = events[event][1]
        while event <= lastindex(events) && events[event][1] == time
            usage += events[event][2]
            event += 1
        end
        if event > lastindex(events) || events[event][1] > time
            output += 1
            profile[output] = usage
        end
    end
    resize!(profile, output)
    return profile
end

"""
Build a typed weighted-interval profile.

Finite positive-width segments are followed by one unit-width zero-load witness for the
exterior of the schedule. The witness preserves the zero set for lower-bound and set
conditions over the full time line; it has no effect on ordinary non-negative capacities.
"""
function _weighted_interval_segments(values, pair_vars)
    _check_weighted_interval_arguments(values, pair_vars)

    lengths = @view pair_vars[1, :]
    heights = @view pair_vars[2, :]
    time_type = promote_type(eltype(values), eltype(lengths))
    load_type = promote_type(eltype(heights), Int)
    events = Vector{Tuple{time_type, load_type}}(undef, 2length(values))
    @inbounds for task in eachindex(values)
        event = 2task - 1
        start = convert(time_type, values[task])
        height = convert(load_type, heights[task])
        events[event] = (start, height)
        events[event + 1] = (start + lengths[task], -height)
    end
    sort!(events)

    segments = Vector{WeightedIntervalSegment{load_type, time_type}}(
        undef, max(1, length(events)),
    )
    if isempty(events)
        segments[1] = WeightedIntervalSegment(zero(load_type), one(time_type))
        return segments
    end
    usage = zero(load_type)
    event = firstindex(events)
    output = 0
    @inbounds while event <= lastindex(events)
        time = events[event][1]
        while event <= lastindex(events) && events[event][1] == time
            usage += events[event][2]
            event += 1
        end
        if event <= lastindex(events)
            width = events[event][1] - time
            if width > zero(width)
                output += 1
                segments[output] = WeightedIntervalSegment(usage, width)
            end
        end
    end
    output += 1
    segments[output] = WeightedIntervalSegment(zero(load_type), one(time_type))
    resize!(segments, output)
    return segments
end

"""Combinatorial sweep-line features for matrix-parameterized interval families."""
const EventMap = LayerCore(
    :EventMap,
    true,
    (:(AbstractVector),) => AbstractVector,
    (
        weighted_interval_segments = :(
            (x; pair_vars) -> CompositionalNetworks._weighted_interval_segments(
                x, pair_vars,
            )
        ),
    ),
)

@testitem "Weighted interval segments group events without deciding validity" begin
    import Test: @test, @test_throws

    segments = EventMap.fn[:weighted_interval_segments]
    observed = segments([0, 1, 3]; pair_vars = [2 2 1; 2 1 2])
    @test getproperty.(observed, :load) == [2, 3, 1, 2, 0]
    @test getproperty.(observed, :width) == [1, 1, 1, 1, 1]
    observed = segments([0, 2]; pair_vars = [2 1; 3 4])
    @test getproperty.(observed, :load) == [3, 4, 0]
    @test getproperty.(observed, :width) == [2, 1, 1]
    @test getproperty.(segments([1, 1]; pair_vars = [0 0; 2 3]), :load) == [0]
    @test getproperty.(segments(Int[]; pair_vars = zeros(Int, 2, 0)), :load) == [0]
    @test_throws DimensionMismatch segments([0, 1]; pair_vars = ones(Int, 3, 2))
    @test_throws DimensionMismatch segments([0, 1]; pair_vars = ones(Int, 2, 3))
    @test_throws ArgumentError segments([0, 1]; pair_vars = [-1 1; 1 1])

    @test all(
        name -> !occursin(r"cumulative|overlap", lowercase(String(name))),
        keys(EventMap.fn),
    )
    @test all(
        operation -> !occursin(
            r"cumulative|overlap",
            lowercase(string(CompositionalNetworks.codegen_ast(operation))),
        ),
        EventMap.fnexprs,
    )
end

@testitem "Weighted interval maximum residual is reconstructible" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @test

    network = CN.ICN(
        parameters = [:pair_vars, :op, :val, :numvars, :dom_size],
        layers = [CN.EventMap, CN.SegmentMap, CN.Arithmetic,
            CN.Aggregation, CN.Comparison],
        connection = UInt32.(1:5),
    )
    fill!(network.weights.parent, false)
    operations = (
        (:weighted_interval_segments,), (:loads,), (:sum,), (:maximum,),
        (:var_minus_val,),
    )
    let offset = 0
        for (layer, selected) in zip(network.layers, operations)
            names = collect(keys(layer.fn))
            for operation in selected
                network.weights.parent[offset + only(findall(==(operation), names))] = true
            end
            offset += length(layer.fn)
        end
    end
    @test CN.check_weights_validity(network, network.weights)
    @test CN.code(network, :maths; name = "penalty") ==
          "penalty(x) = var_minus_val(maximum(loads(weighted_interval_segments(x))))"

    compiled = first(CN.compose(network; name = :weighted_interval_residual_test))
    task_sets = (
        [2 2 1; 2 1 2],
        [0 3 1; 5 2 4],
        [1 1 1; 1 3 2],
    )
    for pair_vars in task_sets, capacity in 0:7,
        assignment in Iterators.product(ntuple(_ -> 0:4, 3)...)
        origins = collect(assignment)
        maximum_load = 0
        for time in minimum(origins):(maximum(origins .+ pair_vars[1, :]) - 1)
            load = sum((pair_vars[2, task] for task in eachindex(origins)
                        if origins[task] <= time < origins[task] + pair_vars[1, task]);
                init = 0)
            maximum_load = max(maximum_load, load)
        end
        parameters = (;
            pair_vars, op = (<=), val = capacity, numvars = 3, dom_size = 5,
        )
        @test compiled(origins; parameters...) == max(0, maximum_load - capacity)
    end
end

@testitem "Weighted interval condition areas are reconstructible" default_imports=false begin
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

    parameters = [:pair_vars, :op, :val, :numvars, :dom_size]
    layers = [CN.EventMap, CN.SegmentMap, CN.Arithmetic,
        CN.Aggregation, CN.Comparison]
    linear = select!(CN.ICN(; parameters, layers, connection = UInt32.(1:5)), (
        (:weighted_interval_segments,), (:widths, :condition_residuals),
        (:product,), (:sum,), (:id,),
    ))
    @test CN.check_weights_validity(linear, linear.weights)
    @test occursin("weighted_interval_segments", CN.code(linear, :maths))
    @test occursin("condition_residuals", CN.code(linear, :maths))

    task_sets = ([2 2 1; 2 1 2], [0 3 1; 5 2 4], [1 1 1; 1 3 2])
    for pair_vars in task_sets, capacity in 0:7,
        assignment in Iterators.product(ntuple(_ -> 0:4, 3)...)
        origins = collect(assignment)
        expected_linear = 0.0
        for time in minimum(origins):(maximum(origins .+ pair_vars[1, :]) - 1)
            load = sum((pair_vars[2, task] for task in eachindex(origins)
                        if origins[task] <= time < origins[task] + pair_vars[1, task]);
                init = 0)
            residual = max(0, load - capacity)
            expected_linear += residual
        end
        runtime = (; pair_vars, op = (<=), val = capacity, numvars = 3, dom_size = 5)
        configuration = CN.Solution(origins)
        @test CN.evaluate(linear, configuration; runtime...) == expected_linear
    end

    exterior_runtime = (;
        pair_vars = [1 1; 1 1], op = (>=), val = 1, numvars = 2, dom_size = 1)
    exterior_configuration = CN.Solution([0, 0])
    @test CN.evaluate(linear, exterior_configuration; exterior_runtime...) > 0

    not_in = (load, forbidden) -> load ∉ forbidden
    grouped_runtime = (;
        pair_vars = [1 1; 1 1], op = not_in, val = 1:1, numvars = 2, dom_size = 1)
    @test CN.evaluate(linear, exterior_configuration; grouped_runtime...) == 0
end
