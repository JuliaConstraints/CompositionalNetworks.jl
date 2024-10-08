"""
    co_identity(x)
Identity function. Already defined in Julia as `identity`, specialized for scalars in the `comparison` layer.
"""
co_identity(x; params...) = identity(x)

"""
    co_abs_diff_var_val(x; val)
Return the absolute difference between `x` and `val`.
"""
co_abs_diff_var_val(x; val, params...) = abs(x - val)

"""
    co_var_minus_val(x; val)
Return the difference `x - val` if positive, `0.0` otherwise.
"""
co_var_minus_val(x; val, params...) = max(0.0, x - val)

"""
    co_val_minus_var(x; val)
Return the difference `val - x` if positive, `0.0` otherwise.
"""
co_val_minus_var(x; val, params...) = max(0.0, val - x)

"""
    co_euclidean_val(x; val, dom_size)
Compute an euclidean norm with domain size `dom_size`, weighted by `val`, of a scalar.
"""
function co_euclidean_val(x; val, dom_size, params...)
    return x == val ? 0.0 : (1.0 + abs(x - val) / dom_size)
end

"""
    co_euclidean(x; dom_size)
Compute an euclidean norm with domain size `dom_size` of a scalar.
"""
function co_euclidean(x; dom_size, params...)
    return co_euclidean_val(x; val = 0.0, dom_size)
end

"""
    co_abs_diff_var_vars(x; nvars)
Return the absolute difference between `x` and the number of variables `nvars`.
"""
co_abs_diff_var_vars(x; nvars, params...) = abs(x - nvars)

"""
    co_var_minus_vars(x; nvars)
Return the difference `x - nvars` if positive, `0.0` otherwise, where `nvars` denotes the numbers of variables.
"""
co_var_minus_vars(x; nvars, params...) = co_var_minus_val(x; val = nvars)

"""
    co_vars_minus_var(x; nvars)
Return the difference `nvars - x` if positive, `0.0` otherwise, where `nvars` denotes the numbers of variables.
"""
co_vars_minus_var(x; nvars, params...) = co_val_minus_var(x; val = nvars)


# Parametric layers
"""
    make_comparisons(param::Symbol)

Generate the comparison functions for the given parameter.
"""
make_comparisons(param::Symbol) = make_comparisons(Val(param))

function make_comparisons(::Val{:none})
    return LittleDict{Symbol,Function}(
        :identity => co_identity,
        :euclidean => co_euclidean,
        :abs_diff_var_vars => co_abs_diff_var_vars,
        :var_minus_vars => co_var_minus_vars,
        :vars_minus_var => co_vars_minus_var,
    )
end

function make_comparisons(::Val{:val})
    return LittleDict{Symbol,Function}(
        :abs_diff_var_val => co_abs_diff_var_val,
        :var_minus_val => co_var_minus_val,
        :val_minus_var => co_val_minus_var,
        :euclidean_val => co_euclidean_val,
    )
end


"""
    comparison_layer(param = false)
Generate the layer of transformations functions of the ICN. Iff `param` value is set, also includes all the parametric comparison with that value. The operations are mutually exclusive, that is only one will be selected.
"""
function comparison_layer(parameters = Vector{Symbol}())
    comparisons = make_comparisons(:none)

    for p in parameters
        comparisons_param = make_comparisons(p)
        comparisons = LittleDict{Symbol,Function}(union(comparisons, comparisons_param))
    end

    return Layer(true, comparisons, parameters)
end

## SECTION - Test Items
@testitem "Comparison Layer" tags = [:comparison, :layer] begin
    CN = CompositionalNetworks

    data = [3 => (1, 5), 5 => (10, 5)]

    funcs = [CN.co_identity => [3, 5]]

    # test no param/vars
    for (f, results) in funcs
        for (key, vals) in enumerate(data)
            @test f(vals.first) == results[key]
        end
    end

    funcs_param = [
        CN.co_abs_diff_var_val => [2, 5],
        CN.co_var_minus_val => [2, 0],
        CN.co_val_minus_var => [0, 5],
    ]

    for (f, results) in funcs_param
        for (key, vals) in enumerate(data)
            @test f(vals.first; val = vals.second[1]) == results[key]
        end
    end

    funcs_vars = [
        CN.co_abs_diff_var_vars => [2, 0],
        CN.co_var_minus_vars => [0, 0],
        CN.co_vars_minus_var => [2, 0],
    ]

    for (f, results) in funcs_vars
        for (key, vals) in enumerate(data)
            @test f(vals.first, nvars = vals.second[2]) == results[key]
        end
    end

    funcs_val_dom = [CN.co_euclidean_val => [1.4, 2.0]]

    for (f, results) in funcs_val_dom
        for (key, vals) in enumerate(data)
            @test f(vals.first, val = vals.second[1], dom_size = vals.second[2]) ≈
                  results[key]
        end
    end

    funcs_dom = [CN.co_euclidean => [1.6, 2.0]]

    for (f, results) in funcs_dom
        for (key, vals) in enumerate(data)
            @test f(vals.first, dom_size = vals.second[2]) ≈ results[key]
        end
    end

end
