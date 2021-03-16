# # Test with manually weighted ICN
icn = ICN(param=true)
show_layers(icn)
icn.weigths = vcat(trues(18), falses(6))
@test CN.is_viable(icn)
@test length(icn) == 31

f = compose(icn)
@test show_composition(icn) == "identity ∘ sum ∘ sum ∘ [param_minus_val, val_minus_param" *
    ", count_bounding_param, count_g_param, count_l_param, count_eq_param," *
    " contiguous_vals_minus_rev, contiguous_vals_minus, count_l_right, count_g_right" * ", count_l_left, count_g_left, count_lesser, count_greater, count_eq_right, " * "count_eq_left, count_eq, identity]"

v = [1,2,4,3]
@test f(v; param=2, dom_size=4) == 67

CN.generate_weights(icn)

## Test GA and exploration
doms = [domain([1,2,3,4]) for i in 1:4]
err = explore_learn_compose(allunique, domains=doms)
@test err([1,2,3,3], dom_size = 4) > 0.0

## Test export to file
compose_to_file!(icn, "all_different", "test_dummy.jl")
