using Test, Random, CompositionalNetworks

const _arithmetic_reference_product = x -> reduce((t...) -> broadcast(*, t...), x)

function check_arithmetic_preserved(inputs)
    saved = deepcopy(inputs)
    for (actual_fn, reference_fn) in (
        (CompositionalNetworks._arithmetic_sum, sum),
        (CompositionalNetworks._arithmetic_product, _arithmetic_reference_product),
    )
        reference = try reference_fn(inputs) catch e; e end
        actual = try actual_fn(inputs) catch e; e end
        @test typeof(actual) === typeof(reference)
        if reference isa Exception
            @test actual isa Exception
        else
            @test isequal(actual, reference)
            if actual isa Vector{<:Union{Float16,Float32,Float64}}
                @test reinterpret(UInt8, actual) == reinterpret(UInt8, reference)
            end
            if length(inputs) == 1
                @test actual === first(inputs)
            elseif length(inputs) >= 2
                @test all(v -> actual !== v, inputs)
            end
        end
        @test isequal(inputs, saved)
    end
end

@testset "Generated arithmetic compositions preserve the reference" begin
    CN = CompositionalNetworks
    inputs_layer = CN.LayerCore(:ArithmeticInputs, false,
        (:(AbstractVector),) => AbstractVector,
        (original=:((x) -> x), shifted=:((x) -> x .+ one(eltype(x))),
         doubled=:((x) -> x .* 2)))
    for (operation, reference) in ((:sum, sum), (:product, _arithmetic_reference_product))
        network = CN.ICN(layers=[inputs_layer, CN.Arithmetic, CN.Aggregation],
            parameters=Symbol[], connection=UInt32[1, 2, 3])
        fill!(network.weights.parent, false)
        offset = 0
        for (layer, names) in zip(network.layers, ((:original, :shifted, :doubled), (operation,), (:sum,)))
            for name in names
                position = findfirst(==(name), collect(keys(layer.fn)))
                network.weights.parent[offset + position] = true
            end
            offset += length(layer.fn)
        end
        @test CN.check_weights_validity(network, collect(network.weights))
        compiled = first(CN.compose(network; name=gensym(:arithmetic_regression)))
        for values in ([1.0, -2.0, 3.0], Float16[1, -2, 3], Int8[1, -2, 3], BigInt[1, -2, 3])
            expected = sum(reference([values, values .+ one(eltype(values)), values .* 2]))
            actual = Base.invokelatest(compiled, values)
            @test typeof(actual) === typeof(expected)
            @test isequal(actual, expected)
            @test isequal(CN.evaluate(network, CN.Solution(values)), Float64(expected))
        end
    end
end

@testset "Owned arithmetic preserves Base reduction semantics" begin
    rng = MersenneTwister(71039)
    for T in (Bool, Int8, Int16, Int32, Int64, Int128,
              UInt8, UInt16, UInt32, UInt64, UInt128, Float16, Float32, Float64)
        values = T <: AbstractFloat ? T[0, -0.0, 1, -1, Inf, -Inf, NaN,
            floatmin(T), nextfloat(zero(T)), floatmax(T)] :
            T[typemin(T), typemax(T), zero(T), one(T)]
        for n in (0, 1, 2, 3, 4, 7, 15, 16, 17, 1024, 1025)
            check_arithmetic_preserved([rand(rng, values, 11) for _ in 1:n])
        end
        check_arithmetic_preserved([T[] for _ in 1:4])
        repeated = rand(rng, values, 9)
        check_arithmetic_preserved(fill(repeated, 7))
        check_arithmetic_preserved(AbstractVector[rand(rng, values, 11) for _ in 1:7])
    end

    # Promotions must preserve the array element type as well as the values.
    for types in ((Bool, Bool, Int8, Bool), (Int8, Int8, Bool, Int8),
                  (UInt8, Bool, UInt8, Bool), (Float16, Float16, Float32, Float64),
                  (Int64, UInt64, Int128, Bool), (Bool, Bool, Bool, Bool))
        check_arithmetic_preserved(AbstractVector[T[0, 1, 1] for T in types])
    end
    for T in (Float16, Float32, Float64)
        large = T === Float16 ? T(4096) : T === Float32 ? T(1e8) : T(1e16)
        check_arithmetic_preserved([T[large, -zero(T), floatmax(T)],
            T[-large, zero(T), floatmin(T)], T[1, -zero(T), 2]])
        for n in 3:15
            check_arithmetic_preserved([T.(randn(rng, 9)) for _ in 1:n])
        end
    end
    for T in (BigInt, BigFloat, Rational{Int}, ComplexF64)
        check_arithmetic_preserved([T[1, 2, 3] for _ in 1:5])
    end
    check_arithmetic_preserved([trues(4) for _ in 1:4])
    check_arithmetic_preserved([Union{Int,Float64}[1, 2.0] for _ in 1:4])
    check_arithmetic_preserved([view([1.0, 2.0, 3.0], 1:2) for _ in 1:4])
    check_arithmetic_preserved(([1, 2], [3, 4], [5, 6]))
    check_arithmetic_preserved([[1], [2, 3], [4, 5]]) # broadcast compatibility
    check_arithmetic_preserved([[1, 2], [2, 3, 4], [4, 5]]) # incompatible axes
    check_arithmetic_preserved([[1, 2], [2, 3], [4, 5, 6]]) # late mismatch

    # Reuse is private to one call, including when callers share their inputs.
    inputs = [rand(rng, 13) for _ in 1:7]
    saved = deepcopy(inputs)
    expected = (sum(inputs), _arithmetic_reference_product(inputs))
    tasks = [Threads.@spawn begin
        [(CompositionalNetworks._arithmetic_sum(inputs),
          CompositionalNetworks._arithmetic_product(inputs)) for _ in 1:40]
    end for _ in 1:4]
    outputs = reduce(vcat, fetch.(tasks))
    @test all(pair -> isequal(pair, expected), outputs)
    @test isequal(inputs, saved)
    @test length(Set(objectid(pair[1]) for pair in outputs)) == length(outputs)
    @test length(Set(objectid(pair[2]) for pair in outputs)) == length(outputs)

    @test isequal(CompositionalNetworks.Arithmetic.fn[:sum](inputs), expected[1])
    @test isequal(CompositionalNetworks.Arithmetic.fn[:product](inputs), expected[2])
end
