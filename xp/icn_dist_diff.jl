function icn_dist_diff(x; X = zeros(length(x), 3), param=nothing, dom_size)
    CompositionalNetworks.tr_in(Tuple([CompositionalNetworks.tr_contiguous_vals_minus_rev, CompositionalNetworks.tr_contiguous_vals_minus, CompositionalNetworks.tr_count_eq_left]), X, x, param)
    for i in 1:length(x)
        X[i,1] = CompositionalNetworks.ar_sum(@view X[i,:])
    end
    return CompositionalNetworks.ag_sum(@view X[:, 1]) |> (y -> CompositionalNetworks.co_vars_minus_val(y; param, dom_size, nvars=length(x)))
end
