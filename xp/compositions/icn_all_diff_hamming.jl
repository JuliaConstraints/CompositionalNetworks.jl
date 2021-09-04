function icn_all_diff_hamming(x; X = zeros(length(x), 1), param=nothing, dom_size)
    CompositionalNetworks.tr_in(Tuple([CompositionalNetworks.tr_count_eq_left]), X, x, param)
    for i in 1:length(x)
        X[i,1] = CompositionalNetworks.ar_sum(@view X[i,:])
    end
    return CompositionalNetworks.ag_count_positive(@view X[:, 1]) |> (y -> CompositionalNetworks.co_identity(y; param, dom_size, nvars=length(x)))
end
#Generation:
 #8.348 s (53084802 allocations: 2.95 GiB)
#Use:
# 1.052 μs (6 allocations: 352.0 bytes)