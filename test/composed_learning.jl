@testitem "Composed ICNs expose operation wiring and quantifier weights to learning" default_imports=false begin
    using Test
    import CompositionalNetworks as CN
    import Serialization
    import Dictionaries

    # Bounded subgrammar of existing operations, not constraint-specific functions.
    function subset(layer, names)
        expressions = map(names) do name
            expression = deepcopy(layer.fnexprs[findfirst(==(name), keys(layer.fn))])
            filter!(kw -> !(kw isa Expr && kw.head === :(...)), expression.kwargs)
            CN.codegen_ast(expression)
        end
        return CN.LayerCore(layer.name, layer.mutex, layer.argtype, NamedTuple{names}(expressions))
    end
    function small(p)
        layers = CN.AbstractLayer[]
        haskey(p, :pair_vars) && push!(layers, subset(CN.PairedMap, (:prod, :sum)))
        append!(layers, [subset(CN.Transformation, (:id,)), subset(CN.Arithmetic, (:sum,)),
            subset(CN.Aggregation, (:sum, :count_elements)), subset(CN.Comparison, (:condition_residual, :id))])
        return CN.ICN(; layers, parameters = collect(keys(p)), parameter_values = p,
            connection = UInt32.(eachindex(layers)))
    end
    function labels(concept, n, domain)
        return Set{CN.Configuration}(concept(x) ? CN.Solution(x) : CN.NonSolution(x)
            for x in (collect(tuple) for tuple in Iterators.product(ntuple(_ -> domain, n)...)))
    end
    function choose!(network, component, role, index)
        block = only(b for b in CN.composition_weight_blocks(network) if b.component == component && b.role == role)
        network.weights[block.range] .= false
        network.weights[first(block.range) + index - 1] = true
    end

    @testset "Parameter-induced row network searches weights and learns conjunction" begin
        params = (; pair_vars = [1 2 3; 3 2 1], op = ((<=), (>=)), val = [5, 7])
        network = CN.learnable_composition(params; factory = small, include_direct = false, max_depth = 1)
        @test network isa CN.ComposedICN
        @test length(network.components) == 1
        @test only(network.components).repetitions == (:pair_vars_rows,)
        configurations = labels(x -> sum(x .* [1, 2, 3]) <= 5 && sum(x .* [3, 2, 1]) >= 7, 3, 0:2)
        optimizer = CN.StructuralEnumerationOptimizer(; max_candidates = 1000, time_limit = 30.0)
        result = CN.optimize!(network, configurations, CN.zero_set_loss, optimizer; params...)
        @test last(result)
        @test optimizer.telemetry[:candidates_evaluated] > 0
        @test CN.zero_set_loss(network, configurations; params...) == 0
        saved_weights = copy(network.weights)
        @test count(identity, saved_weights) > 0
        # Recreate the untrained search space, then install JUST its genotype.
        rebuilt = CN.learnable_composition(params; factory = small, include_direct = false, max_depth = 1)
        @test CN.apply!(rebuilt, saved_weights)
        decoded = CN.composition(rebuilt)
        for n in (3, 5), x in (collect(t) for t in Iterators.product(ntuple(_ -> 0:1, n)...))
            coeffs = vcat(permutedims(collect(1:n)), permutedims(collect(n:-1:1)))
            targets = [n, n + 1]
            p = (; pair_vars = coeffs, op = ((<=), (>=)), val = targets)
            expected = sum(x .* coeffs[1, :]) <= targets[1] && sum(x .* coeffs[2, :]) >= targets[2]
            @test iszero(decoded(x; p...)) == expected
            @test decoded(x; p...) == CN.evaluate(rebuilt, CN.Solution(x); p...)
        end
        # The quantifier is a part of the genotype, not an external configuration.
        choose!(rebuilt, 1, :reduction, 3) # existential minimum
        @test CN.check_weights_validity(rebuilt, rebuilt.weights)
        @test CN.zero_set_loss(rebuilt, configurations; params...) > 0
        @test decoded([0, 0, 0]; params...) > 0 # snapshot unaffected by mutations
        @test occursin("pair_vars_rows", CN.code(decoded, :maths))
        @test CN.canonical_key(decoded) != CN.canonical_key(rebuilt)
        # Weights are plain portable data; no serialized anonymous witness is needed.
        io = IOBuffer()
        Serialization.serialize(io, saved_weights)
        seekstart(io)
        @test CN.apply!(rebuilt, Serialization.deserialize(io))
        @test CN.zero_set_loss(rebuilt, configurations; params...) == 0
        dict_weights = Dictionaries.Dictionary(collect(eachindex(saved_weights)), collect(saved_weights))
        @test CN.apply!(rebuilt, dict_weights)
        @test_throws DimensionMismatch CN.apply!(rebuilt, saved_weights[1:end-1])
    end

    @testset "Per-value quantifier is learned and independent of value-list length" begin
        p = (; vals = [1, 3], op = (==))
        network = CN.learnable_composition(p; factory = small, include_direct = false, max_depth = 1)
        configurations = labels(x -> sum(x) in p.vals, 3, 0:1)
        _, valid = CN.optimize!(network, configurations, CN.zero_set_loss,
            CN.StructuralEnumerationOptimizer(; max_candidates = 1000, time_limit = 30.0); p...)
        @test valid
        decoded = CN.composition(network)
        for vals in ([0], [1, 2, 4], Int[]), x in ([0, 0, 0], [1, 1, 0], [1, 1, 2])
            @test iszero(decoded(x; vals, op = (==))) == (sum(x) in vals)
        end
    end

    @testset "Nested parameter-induced quantifiers have an encodable witness" begin
        p = (; vals = [1, 3], op = (==))
        network = CN.learnable_composition(p; factory = small, include_direct = false, max_depth = 2)
        @test length(network.components) == 3
        for b in CN.composition_weight_blocks(network)
            network.weights[b.range] .= false
            network.weights[first(b.range)] = true
        end
        inner = network.components[2].network
        @test inner isa CN.ComposedICN
        # Choose normal scalar residual operations and existential inner quantifier.
        for b in CN.composition_weight_blocks(inner)
            inner.weights[b.range] .= false
            inner.weights[first(b.range)] = true
        end
        choose!(inner, 1, :reduction, 3) # exists_min over vals
        # Install the child genotype in the corresponding part of the parent genotype.
        blocks = filter(b -> b.component == 2 && b.role == :operation,
            CN.composition_weight_blocks(network))
        network.weights[first(first(blocks).range):last(last(blocks).range)] .= inner.weights
        choose!(network, 2, :repetition, 1) # one child ICN per singleton
        choose!(network, 2, :reduction, 1) # forall_sum over variables
        choose!(network, 0, :outputs, 2)
        # Unused first component still has a valid operation assignment.
        blocks1 = filter(b -> b.component == 1 && b.role == :operation,
            CN.composition_weight_blocks(network))
        for b in blocks1
            network.weights[b.range] .= false
            network.weights[first(b.range)] = true
        end
        @test CN.apply!(network, copy(network.weights))
        weights = copy(network.weights)
        rebuilt = CN.learnable_composition(p; factory = small, include_direct = false, max_depth = 2)
        @test CN.apply!(rebuilt, weights)
        decoded = CN.composition(rebuilt)
        for vals in ([1, 3], [0, 2, 4]), n in (3, 5), tuple in Iterators.product(ntuple(_ -> 0:3, n)...)
            x = collect(tuple)
            @test iszero(decoded(x; vals, op = (==))) == all(v -> v in vals, x)
        end
        @test decoded([1, 1, 3]; vals = Int[], op = (==)) > 0
        @test decoded(Int[]; vals = Int[], op = (==)) == 0
    end

    @testset "Edges output selection bindings and repeated scopes are genotype choices" begin
        p = (; op = (==), val = 0)
        child = small(p)
        first_node = CN.ICNComponent(child; bindings = ((; op = (==), val = 0),),
            repetitions = (:once, :singletons, :prefixes))
        second = CN.ICNComponent(child;
            inputs = (CN.InputReference(), CN.ComponentReference(1; branches = true)),
            bindings = ((; op = (==), val = 0), (; op = (==), val = 3)))
        network = CN.ComposedICN([first_node, second]; reductions = (:forall_sum, :exists_min))
        # Restrict each operation to its first alternative: id/sum/sum/condition.
        for b in CN.composition_weight_blocks(network)
            network.weights[b.range] .= false
            network.weights[firstindex(network.weights) + first(b.range) - 1] = true
        end
        choose!(network, 1, :repetition, 3) # prefixes produce [1,3,6]
        choose!(network, 2, :input, 2)
        choose!(network, 0, :outputs, 2)
        @test CN.apply!(network, copy(network.weights))
        @test CN.evaluate(network, CN.Solution([1, 2, 3])) == 10
        choose!(network, 2, :input, 1)
        @test CN.evaluate(network, CN.Solution([1, 2, 3])) == 6
        choose!(network, 2, :binding, 2)
        @test CN.evaluate(network, CN.Solution([1, 2, 3])) == 3
        @test_throws ArgumentError CN.ComposedICN([CN.ICNComponent(child; inputs = (CN.ComponentReference(1),))])
        @test_throws ArgumentError CN.ComposedICN([CN.ICNComponent(child; inputs = (identity,))])
        block = only(b for b in CN.composition_weight_blocks(network) if b.component == 2 && b.role == :input)
        network.weights[block.range] .= true
        @test !CN.check_weights_validity(network, network.weights)
        @test CN.evaluate(network, CN.Solution([1, 2, 3])) == Inf
        @test_throws ArgumentError CN.composition(network)
    end

    @testset "Matrix vals and Boolean policy are parameter-induced encodable weights" begin
        p = (; vals = [1 2; 3 1], bool = false)
        network = CN.learnable_composition(p)
        @test length(network.components) == 2
        @test network.parameters == Set(keys(p))
        function install_operations!(network, component, operations)
            child = network.components[component].network
            @test child isa CN.ICN
            fill!(parent(child.weights), false)
            offset = 0
            for (layer, op) in zip(child.layers, operations)
                parent(child.weights)[offset + findfirst(==(op), keys(layer.fn))] = true
                offset += length(layer.fn)
            end
            @test CN.check_weights_validity(child, child.weights)
            blocks = filter(b -> b.component == component && b.role == :operation,
                CN.composition_weight_blocks(network))
            network.weights[first(first(blocks).range):last(last(blocks).range)] .= child.weights
        end
        install_operations!(network, 1, (:filter_equal_filter_val, :id, :sum, :count_elements, :condition_residual))
        install_operations!(network, 2, (:filter_ne_vals, :id, :sum, :count_elements, :id))
        for b in CN.composition_weight_blocks(network)
            b.role === :operation && continue
            network.weights[b.range] .= false
            network.weights[first(b.range)] = true
        end
        block = only(b for b in CN.composition_weight_blocks(network) if b.component == 0 && b.role == :outputs)
        network.weights[block.range] .= true
        @test CN.apply!(network, copy(network.weights))
        decoded = CN.composition(network)
        for n in (3, 4), tuple in Iterators.product(ntuple(_ -> 0:3, n)...), closed in (false, true)
            x = collect(tuple)
            expected = count(==(1), x) == 2 && count(==(3), x) == 1 && (!closed || all(v -> v in (1, 3), x))
            @test iszero(decoded(x; vals = p.vals, bool = closed)) == expected
        end
        rebuilt = CN.learnable_composition((; vals = [0 1; 2 2], bool = true))
        @test CN.apply!(rebuilt, copy(network.weights))
        @test CN.evaluate(rebuilt, CN.Solution([0, 2, 2]); vals = [0 1; 2 2], bool = true) == 0
        @test !isempty(CN.symbols(decoded))
        # No-parameter / scalar-val signatures still use the original four layers.
        @test CN.learnable_composition((;)) isa CN.ICN
        scalar = CN.learnable_composition((; val = 2))
        @test [layer.name for layer in scalar.layers] == [:Transformation, :Arithmetic, :Aggregation, :Comparison]
    end
end

@testitem "Generic input bindings preserve aliases and runtime shapes" default_imports=false begin
    using Test
    import CompositionalNetworks as CN
    z=collect(1:10)
    refs=(;pair_vars=CN.ReshapedReference(CN.InputReference([1,2,1,4]),(2,2)),val=CN.InputReference(1))
    resolved=CN.resolve_parameters(refs,z,(;))
    @test resolved.pair_vars==[1 1;2 4]
    z[1]=7
    @test CN.resolve_parameters(refs,z,(;)).pair_vars==[7 7;2 4]
    @test CN.resolve_parameters(refs,z,(;)).val==7
    mixed=CN.ReshapedReference((0,1,((<=),CN.InputReference(1)),((==),CN.InputReference(2))),(2,2))
    @test CN.resolve_parameters((;vals=mixed),z,(;)).vals == Any[0 ((<=),7);1 ((==),2)]
    p=(;pair_vars=CN.InputReference(4:6),op=(==),val=CN.InputReference(1))
    net=CN.learnable_binding(p,z;input=CN.InputReference(1:3))
    @test net isa CN.ComposedICN
    @test isempty(net.parameters)
    @test isempty(net.constants)
    @test_throws DimensionMismatch CN.resolve_parameters(
        (;val=CN.ReshapedReference(CN.InputReference(1:3),(2,2))),z,(;))
    @test_throws BoundsError CN.learnable_binding((;val=CN.InputReference(99)),z)
    @test CN.learnable_composition((;id=nothing,op=(==),val=nothing)) isa CN.ComposedICN
end

@testitem "LocalSearchSolvers consumes composed ICN genotype" tags=[:extension] default_imports=false begin
    using Test
    import CompositionalNetworks as CN
    import LocalSearchSolvers as LS
    params = (; vals = [1, 2], op = (==))
    network = CN.learnable_composition(params; include_direct = false, max_depth = 1)
    configurations = Set{CN.Configuration}([CN.Solution([1, 0, 0]), CN.NonSolution([0, 0, 0])])
    observations = Int[]
    candidates = Any[]
    function metric(icn, configs, solutions; weights_validity, kwargs...)
        push!(observations, length(icn.weights))
        @test icn isa CN.ComposedICN
        return weights_validity ? CN.zero_set_loss(icn, configs, solutions; kwargs...) : Inf
    end
    optimizer = CN.LocalSearchOptimizer(; options = LS.Options(
        iteration = (false, 200), time_limit = 30.0, print_level = :silent,
        log_to_file = false, use_progress_meter = false,
        process_threads_map = Dict(1 => 1)))
    result = CN.optimize!(network, configurations, metric, optimizer;
        candidate_observer = candidate -> push!(candidates, candidate), params...)
    @test first(result) === network
    @test !isempty(observations)
    @test all(==(length(network.weights)), observations)
    @test !isempty(candidates) # Actual objective calls, not only post-run validation.
    @test all(c -> length(c.weights) == length(network.weights), candidates)
    @test CN.generate_new_valid_weights!(network)
    @test CN.check_weights_validity(network, network.weights)
    # A bounded interface test is NOT evidence of learning success.
end
