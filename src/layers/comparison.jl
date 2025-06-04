const Comparison = LayerCore(
    :Comparison,
    true,
    (:(Real),) => Real,
    (
        id = :((x) -> identity(x)),
        abs_val = :((x; val) -> abs(x - val)),
        val_minus_var = :((x; val) -> maximum((0, val - x))),
        var_minus_val = :((x; val) -> maximum((0, x - val))),
        euclidean_val = :(
            (x; val, dom_size) -> x == val ? 0 : (1 + (abs(x - val) / dom_size))
        ),
        euclidean_val_op = :(
            (x; op, val, dom_size) -> op(x, val) ? 0 : (1 + (abs(x - val) / dom_size))
        ),
        euclidean = :((x; dom_size) -> x == 0 ? 0 : (1 + (x / dom_size))),
        euclidean_op = :((x; op, dom_size) -> op(x, 0) ? 0 : (1 + (x / dom_size))),
        var_minus_numvars = :((x; numvars) -> abs(x - numvars)),
        max_numvars_minus_var = :((x; numvars) -> maximum((0, numvars - x))),
        max_var_minus_numvars = :((x; numvars) -> maximum((x - numvars, 0))),
        vals_minus_var_gele = :(
            (x;
        vals) -> length(vals) != 2 ? typemax(eltype(x)) :
                 (
            vals[1] <= x <= vals[2] ? 0 :
            minimum((abs(x - vals[1]), abs(x - vals[2])))
        )
        ),
        vals_minus_var_gl = :(
            (x;
        vals) -> length(vals) != 2 ? typemax(eltype(x)) :
                 (vals[1] < x < vals[2] ? 0 :
                  minimum((abs(x - vals[1]), abs(x - vals[2]))))
        )        #        var_minus_val=:((x; vals) -> maximum((0, (x .- vals)...))),        #        euclidean_val=:((x; vals, dom_size) -> x in vals ? 0 : (1 + (abs((length(vals) * x) - sum(vals)) / dom_size))),
    )
)

# TODO: Add more operations in comparison

# SECTION - Docstrings to put back/update
"""
    co_identity(x)
Identity function. Already defined in Julia as `identity`, specialized for scalars in the `comparison` layer.
"""

"""
    co_abs_diff_var_val(x; val)
Return the absolute difference between `x` and `val`.
"""

"""
    co_var_minus_val(x; val)
Return the difference `x - val` if positive, `0.0` otherwise.
"""

"""
    co_val_minus_var(x; val)
Return the difference `val - x` if positive, `0.0` otherwise.
"""

"""
    co_euclidean_val(x; val, dom_size)
Compute an euclidean norm with domain size `dom_size`, weighted by `val`, of a scalar.
"""

"""
    co_euclidean(x; dom_size)
Compute an euclidean norm with domain size `dom_size` of a scalar.
"""

"""
    co_abs_diff_var_vars(x; nvars)
Return the absolute difference between `x` and the number of variables `nvars`.
"""

"""
    co_var_minus_vars(x; nvars)
Return the difference `x - nvars` if positive, `0.0` otherwise, where `nvars` denotes the numbers of variables.
"""

"""
    co_vars_minus_var(x; nvars)
Return the difference `nvars - x` if positive, `0.0` otherwise, where `nvars` denotes the numbers of variables.
"""

"""
    make_comparisons(param::Symbol)

Generate the comparison functions for the given parameter.
"""

"""
    comparison_layer(param = false)
Generate the layer of transformations functions of the ICN. Iff `param` value is set, also includes all the parametric comparison with that value. The operations are mutually exclusive, that is only one will be selected.
"""

## SECTION - Test Items
# @testitem "Comparison Layer" tags = [:comparison, :layer] begin
#     CN = CompositionalNetworks

#     data = [3 => (1, 5), 5 => (10, 5)]

#     funcs = [CN.co_identity => [3, 5]]

#     # test no param/vars
#     for (f, results) in funcs
#         for (key, vals) in enumerate(data)
#             @test f(vals.first) == results[key]
#         end
#     end

#     funcs_param = [
#         CN.co_abs_diff_var_val => [2, 5],
#         CN.co_var_minus_val => [2, 0],
#         CN.co_val_minus_var => [0, 5],
#     ]

#     for (f, results) in funcs_param
#         for (key, vals) in enumerate(data)
#             @test f(vals.first; val = vals.second[1]) == results[key]
#         end
#     end

#     funcs_vars = [
#         CN.co_abs_diff_var_vars => [2, 0],
#         CN.co_var_minus_vars => [0, 0],
#         CN.co_vars_minus_var => [2, 0],
#     ]

#     for (f, results) in funcs_vars
#         for (key, vals) in enumerate(data)
#             @test f(vals.first, nvars = vals.second[2]) == results[key]
#         end
#     end

#     funcs_val_dom = [CN.co_euclidean_val => [1.4, 2.0]]

#     for (f, results) in funcs_val_dom
#         for (key, vals) in enumerate(data)
#             @test f(vals.first, val = vals.second[1], dom_size = vals.second[2]) ≈
#                   results[key]
#         end
#     end

#     funcs_dom = [CN.co_euclidean => [1.6, 2.0]]

#     for (f, results) in funcs_dom
#         for (key, vals) in enumerate(data)
#             @test f(vals.first, dom_size = vals.second[2]) ≈ results[key]
#         end
#     end

# end
