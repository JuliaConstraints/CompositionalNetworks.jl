using Test, Random
import CompositionalNetworks as CN

# Materialized evaluator retained as an independent regression reference. In
# particular, compact positions are not indices into the full operation catalog.
function materialized_evaluation(network, x; parameters...)
    input = x
    offset = 1
    function_offset = 0
    for (i, layer) in enumerate(network.layers)
        positions = offset:(offset + network.weightlen[i] - 1)
        selected = parentindices(network.weights)[1][positions] .- function_offset
        selected = selected[findall(network.weights[positions])]
        functions = [layer.fn[j] for j in selected]
        input = if layer.name === :Arithmetic && layer.mutex
            operation = collect(keys(layer.fn))[only(selected)]
            operation === :sum ? sum(input) : operation === :product ?
                reduce((t...) -> broadcast(*, t...), input) : functions[1](input; parameters...)
        elseif layer.mutex
            functions[1](input; parameters...)
        else
            output_type = last(layer.argtype)
            outputs = Vector{output_type}(undef, length(functions))
            for j in eachindex(functions)
                outputs[j] = functions[j](input; parameters...)
            end
            outputs
        end
        offset += network.weightlen[i]
        function_offset += length(layer.fn)
    end
    Float64(input)
end

function materialized_names(network, weights)
    selected=Vector{Vector{Symbol}}(); offset=1; function_offset=0
    for (i,layer) in enumerate(network.layers)
        positions=offset:(offset+network.weightlen[i]-1)
        push!(selected,[collect(keys(layer.fn))[parentindices(network.weights)[1][j]-function_offset]
            for j in positions if weights[j]])
        offset+=network.weightlen[i];function_offset+=length(layer.fn)
    end
    selected
end

function materialized_validity(network, weights)
    offset=1
    for (i,layer) in enumerate(network.layers)
        n=sum(weights[offset:(offset+network.weightlen[i]-1)])
        (layer.mutex ? n==1 : n>=1) || return false
        offset+=network.weightlen[i]
    end
    selected=materialized_names(network,weights)
    for (i,layer) in enumerate(network.layers)
        operations=selected[i]
        if layer.name===:Transformation && any(op->op in (:disjoint_pair_differences,:nonzero,:first_equal_position),operations) && length(operations)!=1
            return false
        end
        if layer.name===:PairMask && :zero_extent_groups in operations
            i>1 && network.layers[i-1].name===:PairedMap && selected[i-1]==[:pairwise_oriented_affine_margins] || return false
        end
        if layer.name===:Arithmetic && :difference in operations
            i>1 && !network.layers[i-1].mutex && length(selected[i-1])==2 || return false
        end
    end
    true
end

@testset "Weight validation and regularization preserve their contracts" begin
    Random.seed!(914)
    for layers in ([CN.Transformation,CN.Arithmetic,CN.Aggregation,CN.Comparison],
            [CN.PairedMap,CN.PairMask,CN.GroupReduction,CN.Transformation,CN.Arithmetic,CN.Aggregation,CN.Comparison])
        network=CN.ICN(;layers,parameters=[:dom_size,:numvars,:pair_vars],connection=UInt32.(eachindex(layers)))
        for trial in 1:150
            weights=rand(Bool,length(network.weights))
            network.weights .= weights
            @test CN._selected_operation_names(network,weights)==materialized_names(network,weights)
            @test CN.check_weights_validity(network,weights)==materialized_validity(network,weights)
            selected=0; maximum=0; offset=1
            for (i,layer) in enumerate(layers)
                if !layer.mutex
                    selected+=length(findall(weights[offset:(offset+network.weightlen[i]-1)]))
                    maximum+=network.weightlen[i]
                end
                offset+=network.weightlen[i]
            end
            @test isequal(CN.regularization(network),selected/(maximum+1))
        end
    end
end

@testset "Concurrent ICN selections own their changing weights" begin
    tasks = map(1:4) do worker
        Threads.@spawn begin
            Random.seed!(300 + worker)
            network = CN.ICN(parameters=[:dom_size,:numvars])
            checks = Bool[]
            for generation in 1:20
                CN.generate_new_valid_weights!(network)
                for x in ([1,1,1], [1,2,3], [3,1,2], [2,1,2])
                    expected = materialized_evaluation(network, x;dom_size=3,numvars=3)
                    push!(checks, isequal(CN.evaluate(network,CN.Solution(x);dom_size=3,numvars=3),expected))
                end
            end
            checks
        end
    end
    for task in tasks, check in fetch(task)
        @test check
    end
end

@testset "Direct ICN operation selection preserves evaluation" begin
    Random.seed!(781)
    for params in ((;), (;dom_size=3, numvars=3))
        network = CN.ICN(parameters=collect(keys(params)),
            layers=[CN.Transformation, CN.Arithmetic, CN.Aggregation, CN.Comparison],
            connection=UInt32[1,2,3,4])
        # Parameter filtering leaves holes in the full catalog.
        @test parentindices(network.weights)[1] != collect(1:length(network.weights))
        for generation in 1:80
            CN.generate_new_valid_weights!(network)
            @test CN.check_weights_validity(network, collect(network.weights))
            for x in ([1,1,1], [1,2,3], [3,1,2], [2,1,2])
                expected = materialized_evaluation(network, x; params...)
                @test isequal(CN.evaluate(network, CN.Solution(x); params...), expected)
                @test CN.evaluate(network, CN.Solution(x); weights_validity=false, params...) == Inf
            end
        end
    end
    # The non-mutex output keeps its declared element type, including BigInt
    # values; no Float64 scratch buffer or shared mutable cache is introduced.
    layer = CN.Transformation
    id_index = findfirst(==(:id), keys(layer.fn))
    indices = [id_index, id_index]
    weights = Bool[true, true]
    input = BigInt[big(2)^70, 1, 2]
    output = CN._evaluate_selected_layer(layer, weights, indices, 1:2, 0, input)
    @test eltype(output) == last(layer.argtype)
    @test output == [input, input]
    @test output[1][1] == big(2)^70
    weights[2] = false
    @test length(CN._evaluate_selected_layer(layer, weights, indices, 1:2, 0, input)) == 1
    @test_throws BoundsError CN._evaluate_selected_layer(CN.Arithmetic, falses(1), [1], 1:1, 0, [input])
end
