# # Test with manually weighted ICN

icn = ICN(nvars=4, dom_size=4, param=2)
show_layers(icn)
icn.weigths = vcat(trues(18), falses(6))

f = compose(icn)
@test show_composition(icn) == "identity ∘ sum ∘ sum ∘ [param_minus_val, val_minus_param" *
    ", count_bounding_param, count_g_param, count_l_param, count_eq_param," *
    " contiguous_vals_minus_rev, contiguous_vals_minus, count_l_right, count_g_right" * ", count_l_left, count_g_left, count_lesser, count_greater, count_eq_right, " * "count_eq_left, count_eq, identity]"

v = [1,2,4,3]
@test f(v) == 67

## Test GA

X_sol = csv2space("../data/csv/complete_ad-4-4.csv"; filter=:solutions)
@test hamming([1,2,3,3], X_sol) == 1

X = csv2space("../data/csv/complete_ad-4-4.csv")
icn = ICN(nvars=4, dom_size=4)

CN._optimize!(icn, X, X_sol)
@test CN._is_viable(icn)
err = compose(icn)
@test err([1,2,3,3]) > 0.0
@info show_composition(icn)

CN.optimize!(icn, X, X_sol, 10, 100)