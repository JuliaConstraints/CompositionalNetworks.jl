abstract type AbstractOptimizer end

function optimize!(icn, configurations, metric_function, optimizer_config; parameters...)
    error("No backend loaded")
end

struct LocalSearchOptimizer <: AbstractOptimizer
    options
end

# SECTION - GeneticOptimizer Extension
struct GeneticOptimizer <: AbstractOptimizer
    global_iter::Int
    local_iter::Int
    memoize::Bool
    pop_size::Int
    sampler::Union{Nothing,Function}
end


@testitem "GeneticOptimizer" tags = [:extension] default_imports = false begin
    import CompositionalNetworks: Transformation, Arithmetic, Aggregation, Comparison, ICN
    import CompositionalNetworks: GeneticOptimizer, explore_learn
    import ConstraintDomains: domain
    import Evolutionary
    import Test: @test

    test_icn = ICN(;
        parameters=[:val],
        layers=[Transformation, Arithmetic, Aggregation, Comparison],
        connection=[1, 2, 3, 4],
    )

    function allunique_val(x; val)
        for i in 1:(length(x)-1)
            for j in (i+1):length(x)
                if x[i] == x[j]
                    if x[i] != val
                        return false
                    end
                end
            end
        end
        return true
    end

    function gele_vals(x; vals)
        for i in x
            if i > vals[2] && i < vals[1]
                return false
            end
        end
        return true
    end

    @test explore_learn([domain([1, 2, 3, 4]) for i in 1:4], allunique_val, GeneticOptimizer(), icn=test_icn, val=3)[2]

    new_test_icn = ICN(;
        parameters=[:vals],
        layers=[Transformation, Arithmetic, Aggregation, Comparison],
        connection=[1, 2, 3, 4],
    )

    x, y = explore_learn([domain([1, 2, 3, 4]) for i in 1:4], gele_vals, GeneticOptimizer(), icn=new_test_icn, vals=[2, 5])
    @test x[2] == y[2] == true
end

# SECTION - CBLSOptimizer Extension
