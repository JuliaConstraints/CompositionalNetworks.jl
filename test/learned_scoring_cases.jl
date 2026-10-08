@testitem "Learned scoring oracle avoids captured accumulator allocation" default_imports=false begin
    using Test
    include(joinpath(@__DIR__, "..", "perf", "learned_scoring.jl"))
    function reference(x, witness)
        total = 0
        for i in eachindex(x), j in eachindex(x)
            if j < i
                if witness == 2
                    total += x[j] < x[i]
                    total += x[j] > x[i]
                else
                    total += x[j] == x[i]
                    total += x[j] > x[i]
                end
            end
        end
        total
    end
    for witness in (2, 52)
        for n in 0:5, assignment in Iterators.product(ntuple(_ -> -1:1, n)...)
            values = collect(assignment)
            @test LearnedScoringCases.expected_counts(values, witness) ==
                  reference(values, witness)
        end
        values = [NaN, Inf, -Inf, -0.0, 0.0, 1.0, 1.0]
        @test LearnedScoringCases.expected_counts(values, witness) == reference(values, witness)
        @test LearnedScoringCases.expected_counts(view(values, 2:6), witness) ==
              reference(view(values, 2:6), witness)
        values = mod.(collect(1:1000), 23)
        function allocations(values, witness)
            LearnedScoringCases.expected_counts(values, witness)
            @allocated LearnedScoringCases.expected_counts(values, witness)
        end
        @test allocations(values, witness) == 0
    end
end
