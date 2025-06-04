abstract type AbstractOptimizer end

function optimize!(icn, configurations, metric_function, optimizer_config; parameters...)
    error("No backend loaded")
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
    import CompositionalNetworks: Transformation, Arithmetic, Aggregation, Comparison, ICN, SimpleFilter
    import CompositionalNetworks: GeneticOptimizer, explore_learn
    import ConstraintDomains: domain
    import Evolutionary
    import Test: @test

    test_icn = ICN(;
        parameters=[:dom_size, :numvars, :val],
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

    function allunique_vals(x; vals)
        for i in 1:(length(x)-1)
            for j in (i+1):length(x)
                if x[i] == x[j]
                    if !(x[i] in vals)
                        return false
                    end
                end
            end
        end
        return true
    end

    @test explore_learn([domain([1, 2, 3, 4]) for i in 1:4], allunique_val, GeneticOptimizer(), icn=test_icn, val=3)[2]

    new_test_icn = ICN(;
        parameters=[:dom_size, :numvars, :vals],
        layers=[SimpleFilter, Transformation, Arithmetic, Aggregation, Comparison],
        connection=[1, 2, 3, 4, 5],
    )

    @test explore_learn([domain([1, 2, 3, 4]) for i in 1:4], allunique_vals, GeneticOptimizer(), icn=new_test_icn, vals=[3, 4])[2]
end

# SECTION - CBLSOptimizer Extension
struct LocalSearchOptimizer <: AbstractOptimizer
    options
end

@testitem "LocalSearchOptimizer" tags = [:extension] default_imports = false begin
    import CompositionalNetworks: Transformation, Arithmetic, Aggregation, SimpleFilter
    import CompositionalNetworks: LocalSearchOptimizer, explore_learn, Comparison, ICN
    import ConstraintDomains: domain
    import LocalSearchSolvers
    import Test: @test

    test_icn = ICN(;
        parameters=[:dom_size, :numvars, :val],
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

    function allunique_vals(x; vals)
        for i in 1:(length(x)-1)
            for j in (i+1):length(x)
                if x[i] == x[j]
                    if !(x[i] in vals)
                        return false
                    end
                end
            end
        end
        return true
    end

    @test explore_learn([domain([1, 2, 3, 4]) for i in 1:4], allunique_val, LocalSearchOptimizer(), icn=test_icn, val=3)[2] broken = true

    new_test_icn = ICN(;
        parameters=[:dom_size, :numvars, :vals],
        layers=[SimpleFilter, Transformation, Arithmetic, Aggregation, Comparison],
        connection=[1, 2, 3, 4, 5],
    )

    @test explore_learn([domain([1, 2, 3, 4]) for i in 1:4], allunique_vals, LocalSearchOptimizer(), icn=new_test_icn, vals=[3, 4])[2] broken = true
end


struct JuMPOptimizer <: AbstractOptimizer

end
