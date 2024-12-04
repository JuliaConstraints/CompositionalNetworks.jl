@testset "ICNs" begin
    using CompositionalNetworks
    using ConstraintDomains
    using Dictionaries
    using Evolutionary
    using Memoization
    using Test
    using ThreadPools

    import CompositionalNetworks: ICN, hamming, Transformation, Arithmetic, Aggregation, Comparison, generate_configurations, AbstractICN, Configurations, Configuration, Solution, NonSolution, solutions, apply!, evaluate

    include("genetic.jl")

    #=

    # # Test with manually weighted ICN
    icn = ICN(param = [:val])
    @test max_icn_length() == 18
    show_layers(icn)

    icn.weights = vcat(trues(18), falses(6))
    @test CompositionalNetworks.is_viable(icn)
    @test length(icn) == 31

    compo = compose(icn)
    @test code(compo; name = "test_composition") ==
          "test_composition = identity ∘ sum ∘ sum ∘ [val_minus_var, var_minus_val" *
          ", count_bounding_val, count_g_val, count_l_val, count_eq_val," *
          " contiguous_vars_minus_rev, contiguous_vars_minus, count_l_right, count_g_right" *
          ", count_l_left, count_g_left, count_lesser, count_greater, count_eq_right, " *
          "count_eq_left, count_eq, identity]"

    v = [1, 2, 4, 3]
    @test composition(compo)(v; val = 2, dom_size = 4) == 67

    CompositionalNetworks.generate_weights(icn)

    ## Test GA and exploration
    domains = [domain([1, 2, 3, 4]) for i = 1:4]
    compo, _ = explore_learn_compose(domains, allunique; optimizer = GeneticOptimizer())
    @test composition(compo)([1, 2, 3, 3], dom_size = 4) > 0.0

    ## Test export to file
    composition_to_file!(compo, "test_dummy.jl", "all_different")
    rm("test_dummy.jl"; force = true)
    =#

    test_icn = ICN(;
        parameters=[:val],
        layers=[Transformation, Arithmetic, Aggregation, Comparison],
        connection=[1, 2, 3, 4],
    )

    config_test = generate_configurations(allunique, [domain([1, 2, 3]), domain([9, 10, 11, 12]), domain([4, 5, 6, 7])])
    
    optimize(test_icn, config_test, hamming, 64, 64)
end
