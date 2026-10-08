@testitem "Parameter-name availability preserves intersection semantics" default_imports=false begin
    import CompositionalNetworks as CN
    import Random: Xoshiro, rand
    import Test: @test

    reference(names, parameters) = intersect(names, parameters) == names
    function outcome(fn, names, parameters)
        try
            fn(names, parameters)
        catch exception
            exception
        end
    end
    rng = Xoshiro(681)
    symbols = [:op, :val, :vals, :numvars, :dom_size, :filter_val, :op_filter, :X]
    for _ in 1:2000
        names = rand(rng, symbols, rand(rng, 0:9))
        parameters = rand(rng, symbols, rand(rng, 0:9))
        saved_names, saved_parameters = copy(names), copy(parameters)
        @test CN._parameter_names_match(names, parameters) == reference(names, parameters)
        @test names == saved_names && parameters == saved_parameters
    end

    # Both native scan limits and the original wide-list fallback are exercised.
    for n in (0, 1, 2, 15, 16, 17, 63, 64, 65, 1000)
        values = [Symbol(:p_, i) for i in 1:n]
        for names in (values, reverse(values), vcat(values, values))
            for parameters in (values, reverse(values), Symbol[], vcat(values, :other))
                @test CN._parameter_names_match(names, parameters) == reference(names, parameters)
            end
        end
    end

    for names in ([:op, :val], Any[:op, 1, :val], Set([:op, :val]), (:op, :val),
            @view([:op, :val, :X][1:2])),
        parameters in ([:op, :val], Any[:op, 1, :val], Set([:op, :val]), (:op, :val),
            @view([:op, :val, :X][1:2]), collect(()), Vector{Union{}}(undef, 1))
        actual = outcome(CN._parameter_names_match, names, parameters)
        expected = outcome(reference, names, parameters)
        @test expected isa Exception ? typeof(actual) === typeof(expected) :
              isequal(actual, expected)
    end
end

@testitem "ICN metadata extractors receive fresh mutable parameter hints" default_imports=false begin
    import CompositionalNetworks as CN
    import ConstraintCommons as CC
    import Test: @test

    struct HintProbe
        inputs::Vector{Vector{Symbol}}
        snapshots::Vector{Vector{Symbol}}
    end
    struct HintLayer{F} <: CN.AbstractLayer
        name::Symbol
        mutex::Bool
        fn::F
    end
    function CC.extract_parameters(probe::HintProbe; parameters)
        push!(probe.inputs, parameters)
        push!(probe.snapshots, copy(parameters))
        parameters[1] = :mutated_hint
        push!(parameters, :added_hint)
        return Vector{Symbol}[]
    end

    usual = copy(CC.USUAL_CONSTRAINT_PARAMETERS)
    expected = vcat(usual, [:numvars, :dom_size, :op_filter, :filter_val])
    probe = HintProbe(Vector{Symbol}[], Vector{Symbol}[])
    layer = HintLayer(:HintProbe, false, (; first = probe, second = probe, third = probe))
    for _ in 1:2
        network = CN.ICN(; weights = trues(3), layers = [layer], connection = UInt32[1])
        @test collect(network.weights) == trues(3)
        @test network.weightlen == [3]
    end
    @test length(probe.inputs) == 6
    @test all(==(expected), probe.snapshots)
    @test all(input -> input isa Vector{Symbol}, probe.inputs)
    @test all(input -> first(input) === :mutated_hint && last(input) === :added_hint,
        probe.inputs)
    @test all(probe.inputs[i] !== probe.inputs[j] for i in 1:6 for j in (i + 1):6)
    @test all(input !== CC.USUAL_CONSTRAINT_PARAMETERS for input in probe.inputs)
    @test CC.USUAL_CONSTRAINT_PARAMETERS == usual
end
