# Transformation layer
data = [
    [1, 5, 2, 4, 3] => 2,
    [1, 2, 3, 2, 1] => 2,
]

# Test transformations without parameters
funcs = Dict(
    CN._tr_identity => [
        data[1].first,
        data[2].first,
    ],
    CN._tr_count_eq => [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
    ],
    CN._tr_count_eq_right => [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ],
    CN._tr_count_eq_left => [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
    ],
    CN._tr_count_greater => [
        [4, 0, 3, 1, 2],
        [3, 1, 0, 1, 3],
    ],
    CN._tr_count_lesser => [
        [0, 4, 1, 3, 2],
        [0, 2, 4, 2, 0],
    ],
    CN._tr_count_g_left => [
        [0, 0, 1, 1, 2],
        [0, 0, 0, 1, 3],
    ],
    CN._tr_count_l_left => [
        [0, 1, 1, 2, 2],
        [0, 1, 2, 1, 0],
    ],
    CN._tr_count_g_right => [
        [4, 0, 2, 0, 0],
        [3, 1, 0, 0, 0],
    ],
    CN._tr_count_l_right => [
        [0, 3, 0, 1, 0],
        [0, 1, 2, 1, 0],
    ],
    CN._tr_contiguous_vals_minus => [
        [0, 3, 0, 1, 0],
        [0, 0, 1, 1, 0],
    ],
    CN._tr_contiguous_vals_minus_rev => [
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
    CN._tr_count_eq_param => [
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
    ],
    CN._tr_count_l_param => [
        [2, 5, 3, 5, 4],
        [4, 5, 5, 5, 4],
    ],
    CN._tr_count_g_param => [
        [2, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    CN._tr_count_bounding_param => [
        [3, 1, 3, 2, 3],
        [5, 3, 1, 3, 5],
    ],
    CN._tr_val_minus_param => [
        [0, 3, 0, 2, 1],
        [0, 0, 1, 0, 0],
    ],
    CN._tr_param_minus_val => [
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
@info CN._ar_sum
@test CN._ar_sum(map(p -> p.first, data)) == [2, 7, 5, 6, 4]
@info CN._ar_prod
@test CN._ar_prod(map(p -> p.first, data)) == [1, 10, 6, 8, 3]

# aggregation layer
@info CN._ag_sum
@test CN._ag_sum(data[1].first) == 15
@test CN._ag_sum(data[2].first) == 9

@info CN._ag_count_positive
@test CN._ag_count_positive(data[1].first) == 5
@test CN._ag_count_positive(data[2].first) == 5
@test CN._ag_count_positive([1, 0, 1, 0, 1]) == 3

# Comparison layer
data = [3 => (1, 5), 5 => (10, 5)]

funcs = [
    CN._co_identity => [3, 5],
]

# test no param/vars
for (f, results) in funcs
    @info f
    for (key, vals) in enumerate(data)
        @test f(vals.first) == results[key]
    end
end

funcs_param = [
    CN._co_abs_diff_val_param => [2, 5],
    CN._co_val_minus_param => [2, 0],
    CN._co_param_minus_val => [0, 5],
]

for (f, results) in funcs_param
    @info f
    for (key, vals) in enumerate(data)
        @test f(vals.first; param=vals.second[1]) == results[key]
    end
end

funcs_vars = [
    CN._co_abs_diff_val_vars => [2, 0],
    CN._co_val_minus_vars => [0, 0],
    CN._co_vars_minus_val => [2, 0],
]

for (f, results) in funcs_vars
    @info f
    for (key, vals) in enumerate(data)
        @test f(vals.first, nvars=vals.second[2]) == results[key]
    end
end

funcs_param_dom = [
    CN._co_euclidian_param => [3.5, 2.0],
]

for (f, results) in funcs_param_dom
    @info f
    for (key, vals) in enumerate(data)
        @test f(vals.first, param=vals.second[1], dom_size=vals.second[2]) ≈ results[key]
    end
end

funcs_dom = [
    CN._co_euclidian => [8 / 3, 2.0],
]

for (f, results) in funcs_dom
    @info f
    for (key, vals) in enumerate(data)
        @test f(vals.first, dom_size=vals.second[2]) ≈ results[key]
    end
end
