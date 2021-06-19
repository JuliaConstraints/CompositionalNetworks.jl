# Transformation layer
data = [
    [1, 5, 2, 4, 3] => 2,
    [1, 2, 3, 2, 1] => 2,
]

# Test transformations without parameters
funcs = Dict(
    CN.tr_identity => [
        data[1].first,
        data[2].first,
    ],
    CN.tr_count_eq => [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
    ],
    CN.tr_count_eq_right => [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ],
    CN.tr_count_eq_left => [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
    ],
    CN.tr_count_greater => [
        [4, 0, 3, 1, 2],
        [3, 1, 0, 1, 3],
    ],
    CN.tr_count_lesser => [
        [0, 4, 1, 3, 2],
        [0, 2, 4, 2, 0],
    ],
    CN.tr_count_g_left => [
        [0, 0, 1, 1, 2],
        [0, 0, 0, 1, 3],
    ],
    CN.tr_count_l_left => [
        [0, 1, 1, 2, 2],
        [0, 1, 2, 1, 0],
    ],
    CN.tr_count_g_right => [
        [4, 0, 2, 0, 0],
        [3, 1, 0, 0, 0],
    ],
    CN.tr_count_l_right => [
        [0, 3, 0, 1, 0],
        [0, 1, 2, 1, 0],
    ],
    CN.tr_contiguous_vals_minus => [
        [0, 3, 0, 1, 0],
        [0, 0, 1, 1, 0],
    ],
    CN.tr_contiguous_vals_minus_rev => [
        [4, 0, 2, 0, 0],
        [1, 1, 0, 0, 0],
    ],
)

for (f, results) in funcs
    @info f
    for (key, vals) in enumerate(data)
        @test f(vals.first) == results[key]
        foreach(i -> f(i, vals.first), vals.first)
    end
end

# Test transformations with parameter
funcs_param = Dict(
    CN.tr_count_eq_param => [
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
    ],
    CN.tr_count_l_param => [
        [2, 5, 3, 5, 4],
        [4, 5, 5, 5, 4],
    ],
    CN.tr_count_g_param => [
        [2, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    CN.tr_count_bounding_param => [
        [3, 1, 3, 2, 3],
        [5, 3, 1, 3, 5],
    ],
    CN.tr_val_minus_param => [
        [0, 3, 0, 2, 1],
        [0, 0, 1, 0, 0],
    ],
    CN.tr_param_minus_val => [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
    ],
)

for (f, results) in funcs_param
    @info f
    for (key, vals) in enumerate(data)
        @test f(vals.first; param=vals.second) == results[key]
        foreach(i -> f(i, vals.first; param=vals.second), vals.first)
    end
end

# arithmetic layer
@info CN.ar_sum
@test CN.ar_sum(map(p -> p.first, data)) == [2, 7, 5, 6, 4]
@info CN.ar_prod
@test CN.ar_prod(map(p -> p.first, data)) == [1, 10, 6, 8, 3]

# aggregation layer
@info CN.ag_sum
@test CN.ag_sum(data[1].first) == 15
@test CN.ag_sum(data[2].first) == 9

@info CN.ag_count_positive
@test CN.ag_count_positive(data[1].first) == 5
@test CN.ag_count_positive(data[2].first) == 5
@test CN.ag_count_positive([1, 0, 1, 0, 1]) == 3

# Comparison layer
data = [3 => (1, 5), 5 => (10, 5)]

funcs = [
    CN.co_identity => [3, 5],
]

# test no param/vars
for (f, results) in funcs
    @info f
    for (key, vals) in enumerate(data)
        @test f(vals.first) == results[key]
    end
end

funcs_param = [
    CN.co_abs_diff_val_param => [2, 5],
    CN.co_val_minus_param => [2, 0],
    CN.co_param_minus_val => [0, 5],
]

for (f, results) in funcs_param
    @info f
    for (key, vals) in enumerate(data)
        @test f(vals.first; param=vals.second[1]) == results[key]
    end
end

funcs_vars = [
    CN.co_abs_diff_val_vars => [2, 0],
    CN.co_val_minus_vars => [0, 0],
    CN.co_vars_minus_val => [2, 0],
]

for (f, results) in funcs_vars
    @info f
    for (key, vals) in enumerate(data)
        @test f(vals.first, nvars=vals.second[2]) == results[key]
    end
end

funcs_param_dom = [
    CN.co_euclidian_param => [1.4, 2.0],
]

for (f, results) in funcs_param_dom
    @info f
    for (key, vals) in enumerate(data)
        @info "Updated" f(vals.first, param=vals.second[1], dom_size=vals.second[2]) results key
        @test f(vals.first, param=vals.second[1], dom_size=vals.second[2]) ≈ results[key]
    end
end

funcs_dom = [
    CN.co_euclidian => [1.6, 2.0],
]

for (f, results) in funcs_dom
    @info f
    for (key, vals) in enumerate(data)
        @info "Updated" f(vals.first, dom_size=vals.second[2]) results key
        @test f(vals.first, dom_size=vals.second[2]) ≈ results[key]
    end
end
