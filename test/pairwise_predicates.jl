@testitem "Native pair predicates preserve learned counts" default_imports=false begin
    import CompositionalNetworks as CN
    using Test, Random

    operations = ((:count_less_left, :count_great_left),
                  (:count_equal_left, :count_great_left))
    function oracle(x, names)
        total = 0
        for i in eachindex(x), j in eachindex(x)
            j < i || continue
            total += names[1] === :count_less_left ?
                     ((x[j] < x[i]) + (x[j] > x[i])) :
                     ((x[j] == x[i]) + (x[j] > x[i]))
        end
        return total
    end
    function compare(x, names)
        before = isbitstype(eltype(x)) ? copy(reinterpret(UInt8, x)) : copy(x)
        expected = oracle(x, names)
        original = invoke(CN._aggregate_pairwise_sum, Tuple{Val{names},Any}, Val(names), x)
        actual = CN._aggregate_pairwise_sum(Val(names), x)
        @test actual == original == expected
        @test typeof(actual) === typeof(original) === Int
        @test (isbitstype(eltype(x)) ? reinterpret(UInt8, x) : x) == before
    end

    rng = MersenneTwister(451)
    for T in (Bool, Int8, Int16, Int32, Int64, Int128,
              UInt8, UInt16, UInt32, UInt64, UInt128)
        values = T === Bool ? Bool[false, true] :
                 T[typemin(T), typemin(T) + one(T), zero(T), one(T),
                   typemax(T) - one(T), typemax(T)]
        for names in operations
            compare(T[], names)
            for a in values
                compare(T[a], names)
                for b in values
                    compare(T[a, b], names)
                    for c in values
                        compare(T[a, b, c], names)
                    end
                end
            end
            compare(rand(rng, T, 32), names)
        end
    end
    for (T, U) in ((Float16, UInt16), (Float32, UInt32), (Float64, UInt64))
        values = T[-Inf, -floatmax(T), -1, -0.0, 0.0, nextfloat(T(0)),
                   1, floatmax(T), Inf, NaN]
        # Include signaling and quiet NaN bit patterns of both signs.
        exponent = T === Float16 ? U(0x7c00) :
                   T === Float32 ? U(0x7f800000) : U(0x7ff0000000000000)
        sign = one(U) << (8sizeof(T) - 1)
        append!(values, reinterpret(T, U[exponent | one(U), exponent | sign | one(U),
                                          typemax(U)]))
        for names in operations
            compare(T[], names)
            for a in values
                compare(T[a], names)
                for b in values
                    compare(T[a, b], names)
                    compare(T[a, b, T(0), T(NaN)], names)
                end
            end
            for _ in 1:64
                compare(collect(reinterpret(T, rand(rng, U, 17))), names)
            end
        end
    end
end

@testitem "Pair predicate fallback preserves custom reads" default_imports=false begin
    import CompositionalNetworks as CN
    using Test

    struct ObservedPairVector <: AbstractVector{Float64}
        values::Vector{Float64}
        reads::Vector{Int}
        throw_at::Int
        mutate::Bool
    end
    Base.size(x::ObservedPairVector) = size(x.values)
    Base.IndexStyle(::Type{ObservedPairVector}) = IndexLinear()
    function Base.getindex(x::ObservedPairVector, index::Int)
        push!(x.reads, index)
        index == x.throw_at && error("late pair read")
        x.mutate && index == 1 && (x.values[end] = 7.0)
        return x.values[index]
    end
    function capture(f)
        try
            (:result, f())
        catch error
            (:error, typeof(error), error isa ErrorException ? error.msg : nothing)
        end
    end
    for names in ((:count_less_left, :count_great_left),
                  (:count_equal_left, :count_great_left)),
            count in (0, 1, 2, 3, 5), throw_at in (0, 1, 3), mutate in (false, true)
        a = ObservedPairVector(Float64.(1:count), Int[], throw_at, mutate)
        b = ObservedPairVector(Float64.(1:count), Int[], throw_at, mutate)
        actual = capture(() -> CN._aggregate_pairwise_sum(Val(names), a))
        original = capture(() -> invoke(CN._aggregate_pairwise_sum,
            Tuple{Val{names},Any}, Val(names), b))
        @test isequal(actual, original)
        @test a.reads == b.reads
        @test a.values == b.values
    end
    for names in ((:count_less_left, :count_great_left),
                  (:count_equal_left, :count_great_left)),
            x in (BigInt[1, 1, 2], Rational{Int}[1, 1, 2], Real[1, 1.0, 2],
                  view([1.0, NaN, 2.0], :), reshape([1, 1, 2], 3))
        @test CN._aggregate_pairwise_sum(Val(names), x) ==
              invoke(CN._aggregate_pairwise_sum, Tuple{Val{names},Any}, Val(names), x)
    end

    struct ObservedPairNumber <: Real
        value::Int
        events::Vector{Tuple{Symbol,Int,Int}}
        throws::Bool
    end
    for (operator, name) in ((:(<), :less), (:(>), :greater), (:(==), :equal))
        @eval function Base.$operator(a::ObservedPairNumber, b::ObservedPairNumber)
            push!(a.events, ($(QuoteNode(name)), a.value, b.value))
            a.throws && a.value == 7 && error("late pair comparison")
            return $operator(a.value, b.value)
        end
    end
    for names in ((:count_less_left, :count_great_left),
                  (:count_equal_left, :count_great_left)), throws in (false, true)
        events_a = Tuple{Symbol,Int,Int}[]
        events_b = Tuple{Symbol,Int,Int}[]
        a = [ObservedPairNumber(value, events_a, throws) for value in (1, 7, 2)]
        b = [ObservedPairNumber(value, events_b, throws) for value in (1, 7, 2)]
        actual = capture(() -> CN._aggregate_pairwise_sum(Val(names), a))
        original = capture(() -> invoke(CN._aggregate_pairwise_sum,
            Tuple{Val{names},Any}, Val(names), b))
        @test isequal(actual, original)
        @test events_a == events_b
    end
end
