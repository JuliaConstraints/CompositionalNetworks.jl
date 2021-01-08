# # Test with manually weighted ICN
icn = ICN(nvars=4, dom_size=4, param=2)
show_layers(icn)
icn.weigths = vcat(trues(18), falses(6))
@test CN._is_viable(icn)

f = compose(icn)
@test show_composition(icn) == "identity ∘ sum ∘ sum ∘ [param_minus_val, val_minus_param" *
    ", count_bounding_param, count_g_param, count_l_param, count_eq_param," *
    " contiguous_vals_minus_rev, contiguous_vals_minus, count_l_right, count_g_right" * ", count_l_left, count_g_left, count_lesser, count_greater, count_eq_right, " * "count_eq_left, count_eq, identity]"

v = [1,2,4,3]
@test f(v) == 67

## Test GA
X_sols = csv2space("../data/csv/complete_ad-4-4.csv"; filter=:solutions)
@test hamming([1,2,3,3], X_sols) == 1
X = csv2space("../data/csv/complete_ad-4-4.csv")
err = optimize_and_compose(nvars=4, dom_size=4, global_iter = 10, X = X, X_sols = X_sols)
@test err([1,2,3,3]) > 0.0