# Transformation layer
data = [
    [1, 5, 2, 4, 3] => 2,
    [1, 2, 3, 2, 1] => 2,
]

# Test transformations without parameters
funcs = Dict(
    CN._identity => [
        data[1].first,
        data[2].first,
    ],
    CN._count_eq => [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
    ],
    CN._count_eq_right => [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ],
    CN._count_eq_left => [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
    ],
    CN._count_greater => [
        [4, 0, 3, 1, 2],
        [3, 1, 0, 1, 3],
    ],
    CN._count_lesser => [
        [0, 4, 1, 3, 2],
        [0, 2, 4, 2, 0],
    ],
    CN._count_g_left => [
        [0, 0, 1, 1, 2],
        [0, 0, 0, 1, 3],
    ],
    CN._count_l_left => [
        [0, 1, 1, 2, 2],
        [0, 1, 2, 1, 0],
    ],
    CN._count_g_right => [
        [4, 0, 2, 0, 0],
        [3, 1, 0, 0, 0],
    ],
    CN._count_l_right => [
        [0, 3, 0, 1, 0],
        [0, 1, 2, 1, 0],
    ],
    CN._contiguous_vals_minus => [
        [0, 3, 0, 1, 0],
        [0, 0, 1, 1, 0],
    ],
    CN._contiguous_vals_minus_rev => [
        [4, 0, 2, 0, 0],
        [1, 1, 0, 0, 0],
    ],
)

for (f, results) in funcs, (key, vals) in enumerate(data)
    @test f(vals.first) == results[key]
    foreach(i -> f(i, vals.first), vals.first)
    @info f
end

# Test transformations with parameter
funcs_param = Dict(
    CN._count_eq_param => [
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
    ],
    CN._count_l_param => [
        [2, 5, 3, 5, 4],
        [4, 5, 5, 5, 4],
    ],
    CN._count_g_param => [
        [2, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    CN._count_bounding_param => [
        [3, 1, 3, 2, 3],
        [5, 3, 1, 3, 5],
    ],
    CN._val_minus_param => [
        [0, 3, 0, 2, 1],
        [0, 0, 1, 0, 0],
    ],
    CN._param_minus_val => [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
    ],
)

for (f, results) in funcs_param, (key, vals) in enumerate(data)
    @test f(vals.first, vals.second) == results[key]
    foreach(i -> f(i, vals.first, vals.second), vals.first)
    @info f
end

# arithmetic layer
@test CN._ar_sum(map(p -> p.first, data)) == [2, 7, 5, 6, 4]
@test CN._ar_prod(map(p -> p.first, data)) == [1, 10, 6, 8, 3]

# aggregation layer
@test CN._ag_sum(data[1].first) == 15
@test CN._ag_sum(data[2].first) == 9

@test CN._count_positive(data[1].first) == 5
@test CN._count_positive(data[2].first) == 5
@test CN._count_positive([1, 0, 1, 0, 1]) == 3

# Comparison layer
data = [3 => (1, 5), 5 => (10, 5)]

funcs = [
    CN._identity => [3, 5],
]

# test no param/vars
for (f, results) in funcs, (key, vals) in enumerate(data)
    @test f(vals.first) == results[key]
end

funcs_param = [
    CN._abs_diff_val_param => [2, 5],
    CN._co_val_minus_param => [2, 0],
    CN._co_param_minus_val => [0, 5],
]

for (f, results) in funcs_param, (key, vals) in enumerate(data)
    @test f(vals.first, vals.second[1]) == results[key]
end

funcs_vars = [
    CN._abs_diff_val_vars => [2, 0],
    CN._val_minus_vars => [0, 0],
    CN._vars_minus_val => [2, 0],
]

for (f, results) in funcs_vars, (key, vals) in enumerate(data)
    @test f(vals.first, vals.second[2]) == results[key]
end


funcs_param_dom = [
    CN._euclidian_param => [3.5, 2.0],
]

for (f, results) in funcs_param_dom, (key, vals) in enumerate(data)
    @test f(vals.first, vals.second[1], vals.second[2]) ≈ results[key]
end

funcs_dom = [
    CN._euclidian => [8/3, 2.0],
]

for (f, results) in funcs_dom, (key, vals) in enumerate(data)
    @test f(vals.first, vals.second[2]) ≈ results[key]
end
