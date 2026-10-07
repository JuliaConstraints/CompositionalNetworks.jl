function _segment_loads(segments)
    return map(segment -> segment.load, segments)
end

function _segment_widths(segments)
    return map(segment -> segment.width, segments)
end

function _segment_condition_residuals(segments, op, val)
    return map(segment -> _condition_residual(segment.load, val, op), segments)
end

"""Field projections and scalar maps over typed event-sweep segments."""
const SegmentMap = LayerCore(
    :SegmentMap,
    false,
    (:(AbstractVector{<:CompositionalNetworks.WeightedIntervalSegment}),) => AbstractVector,
    (
        loads = :((x) -> CompositionalNetworks._segment_loads(x)),
        widths = :((x) -> CompositionalNetworks._segment_widths(x)),
        condition_residuals = :(
            (x; op, val) -> CompositionalNetworks._segment_condition_residuals(x, op, val)
        ),
    ),
)

@testitem "Segment fields remain independently composable" begin
    using Test

    segments = [WeightedIntervalSegment(3, 2), WeightedIntervalSegment(6, 1)]
    @test SegmentMap.fn[:loads](segments) == [3, 6]
    @test SegmentMap.fn[:widths](segments) == [2, 1]
    @test SegmentMap.fn[:condition_residuals](segments; op = (<=), val = 4) == [0.0, 2.0]
end
