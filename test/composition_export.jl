@testitem "All composition paths preserve exported source" default_imports=false begin
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
        network
    end

    inplace = select!(CN.ICN(), ((:id,), (:sum,), (:sum,), (:id,)))
    indexed = select!(CN.ICN(),
        ((:predecessor_counts,), (:sum,), (:count_zero,), (:id,)))
    layers = [CN.SimpleFilter, CN.Transformation, CN.Arithmetic,
        CN.Aggregation, CN.Comparison]
    specialized = select!(CN.ICN(; layers, parameters = [:vals],
        connection = UInt32.(eachindex(layers))),
        ((:filter_ne_vals,), (:id,), (:sum,), (:sum,), (:id,)))
    custom = CN.LayerCore(:CustomInput, false,
        (:(AbstractVector),) => AbstractVector,
        (shifted = :((x) -> x .+ 1),))
    layers = [custom, CN.Arithmetic, CN.Aggregation, CN.Comparison]
    generic = select!(CN.ICN(; layers, connection = UInt32.(eachindex(layers))),
        ((:shifted,), (:sum,), (:sum,), (:id,)))

    @test CN._supports_inplace_compilation(inplace)
    @test !(CN._index_relation_kernel(indexed) isa Val{:generic})
    @test CN._supports_specialized_compilation(specialized)
    @test !CN._supports_inplace_compilation(generic)
    @test !CN._supports_specialized_compilation(generic)
    @test CN._index_relation_kernel(generic) isa Val{:generic}

    cases = ((inplace, [0, 1, 2, 3, 4], (;)),
        (indexed, [2, 3, 1], (;)),
        (specialized, [0, 1, 2, 3, 4], (; vals = (1, 3))),
        (generic, [0, 1, 2, 3, 4], (;)))
    mktempdir() do directory
        for (index, (network, input, parameters)) in enumerate(cases)
            saved_weights = copy(network.weights.parent)
            saved_input = copy(input)
            expected = CN.evaluate(network, CN.Solution(input); parameters...)
            for jlfun in (true, false)
                name = Symbol(:export_path_, index, :_, jlfun)
                path = joinpath(directory, string(name) * ".jl")
                # An AbstractString subtype exercises the original path contract.
                compiled, representation = CN.compose(network;
                    name, jlfun, fname = SubString(path))
                _, definition = CN.compose(network; name)
                @test read(path, String) == CN.sprint_expr(definition)
                @test jlfun ? representation isa CN.JLFunction : representation isa Expr
                @test Base.invokelatest(compiled, input; parameters...) == expected
                owner = Module(gensym(:ExportedComposition))
                Core.eval(owner, :(import CompositionalNetworks))
                exported = Base.include(owner, path)
                @test Base.invokelatest(exported, input; parameters...) == expected
                @test network.weights.parent == saved_weights
                @test input == saved_input
            end
            @test_throws SystemError CN.compose(network;
                fname = joinpath(directory, "missing", "composition.jl"))
        end
    end
end
