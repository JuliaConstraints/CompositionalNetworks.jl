@testitem "Compiled directional counts preserve the atomic pair sum" default_imports=false begin
    import CompositionalNetworks as CN
    import Test: @test

    struct CartesianVector{T} <: AbstractVector{T}
        values::Vector{T}
    end
    Base.size(x::CartesianVector) = size(x.values)
    Base.IndexStyle(::Type{<:CartesianVector}) = IndexCartesian()
    Base.getindex(x::CartesianVector, index::Int) = x.values[index]

    # Independent full-square reference, including every zero-valued pair.
    function reference(x, operations)
        total = 0
        for i in eachindex(x), j in eachindex(x), operation in operations
            left = endswith(String(operation), "left")
            active = left ? j < i : j > i
            active || continue
            total += startswith(String(operation), "count_equal") ? x[j] == x[i] :
                     startswith(String(operation), "count_less") ? x[j] < x[i] :
                     x[j] > x[i]
        end
        return total
    end

    operation_sets = (
        (:count_equal_left,), (:count_less_left,), (:count_great_left,),
        (:count_equal_right,), (:count_less_right,), (:count_great_right,),
        (:count_less_left, :count_great_left),
        (:count_equal_left, :count_great_left),
        (:count_less_right, :count_great_right),
        (:count_equal_right, :count_great_right),
        (:count_equal_left, :count_equal_right),
        (:count_less_left, :count_great_right),
    )
    for operations in operation_sets
        for n in 0:5, assignment in Iterators.product(ntuple(_ -> -1:1, n)...)
            x = collect(assignment)
            @test CN._aggregate_pairwise_sum(Val(operations), x) == reference(x, operations)
        end
        x = [NaN, Inf, -Inf, 0.0, -0.0, 1.0, 1.0]
        @test CN._aggregate_pairwise_sum(Val(operations), x) == reference(x, operations)
        @test CN._aggregate_pairwise_sum(Val(operations), view(x, 2:6)) ==
              reference(view(x, 2:6), operations)
        cartesian = CartesianVector([2, 1, 2, 3])
        @test CN._aggregate_pairwise_sum(Val(operations), cartesian) ==
              reference(cartesian, operations)
        @test CN._aggregate_pairwise_sum(Val(operations), (2, 1, 2, 3)) ==
              reference((2, 1, 2, 3), operations)
    end

    # These are the selected atomic operations in exact learned witnesses 2 and 52.
    for operations in operation_sets[7:8]
        network = CN.ICN()
        fill!(network.weights.parent, false)
        offset = 0
        for (layer, selected) in zip(network.layers,
                (operations, (:sum,), (:sum,), (:id,)))
            names = collect(keys(layer.fn))
            for name in selected
                network.weights.parent[offset + findfirst(==(name), names)] = true
            end
            offset += length(layer.fn)
        end
        @test CN.check_weights_validity(network, network.weights)
        compiled = first(CN.compose(network))
        values = mod.(collect(1:1000), 23)
        @test Base.invokelatest(compiled, values) == reference(values, operations)
        @test CN.evaluate(network, CN.Solution(values)) == reference(values, operations)
        function allocations(compiled, values)
            compiled(values)
            return @allocated compiled(values)
        end
        @test Base.invokelatest(allocations, compiled, values) == 0
    end
end
