export AbstractLayer, Transformation, Aggregation, LayerCore, Arithmetic, Comparison

abstract type AbstractLayer end

struct LayerCore <: AbstractLayer
    name::Symbol
    mutex::Bool
    argtype::Pair
    fnexprs::NamedTuple{names,T} where {names,T<:Tuple{Vararg{<:Union{Symbol,JLFunction}}}}
    fn::NamedTuple{names,T} where {names,T<:Tuple{Vararg{Function}}}
    function LayerCore(name::Symbol, mutex::Bool, Q::Pair, fnexprs)
        fnexprs = map(x -> JLFunction(x), fnexprs)
        for jlexp in fnexprs
            #=
            if isnothing(jlexp.rettype)
            	jlexp.rettype = Q[2]
            end
            =#
            for (i, arg) in enumerate(jlexp.args)
                if arg isa Symbol
                    jlexp.args[i] = Expr(:(::), arg, Q[1][i])
                end
            end
            if isnothing(jlexp.kwargs)
                jlexp.kwargs = [:(params...)]
            else
                push!(jlexp.kwargs, :(params...))
            end
        end
        new(name, mutex, Q, fnexprs, map(x -> eval(codegen_ast(x)), fnexprs))
    end
end

const Transformation = LayerCore(
    :Transformation,
    false,
    (:(AbstractVector{<:Real}),) => AbstractVector{<:Real},
    (
        id=:((x) -> identity(x)),
        count_equal_right=:((x) -> map(i -> count(t -> t == x[i], @view(x[i+1:end])), eachindex(x))),
        count_less_right=:((x) -> map(i -> count(t -> t < x[i], @view(x[i+1:end])), eachindex(x))),
        count_great_right=:((x) -> map(i -> count(t -> t > x[i], @view(x[i+1:end])), eachindex(x))),
        count_equal_left=:((x) -> map(i -> count(t -> t == x[i], @view(x[1:i-1])), eachindex(x))),
        count_less_left=:((x) -> map(i -> count(t -> t < x[i], @view(x[1:i-1])), eachindex(x))),
        count_great_left=:((x) -> map(i -> count(t -> t > x[i], @view(x[1:i-1])), eachindex(x))),
        count_equal_val=:((x; val = 0) -> map(i -> count(t -> t == (i + val), x), x)),
        count_less_val=:((x; val = 0) -> map(i -> count(t -> t < (i + val), x), x)),
        count_great_val=:((x; val = 0) -> map(i -> count(t -> t > (i + val), x), x)),
        var_minus_val=:((x; val = 0) -> map(i -> max(0, i - val), x)),
        val_minus_var=:((x; val = 0) -> map(i -> max(0, val - i), x)),
        contiguous_vars_minus=:((x) -> map(i -> i == length(x) ? 0 : max(0, x[i] - x[i+1]), eachindex(x[1:end]))),
        contiguous_vars_minus_rev=:((x) -> map(i -> i == length(x) ? 0 : max(0, x[i+1] - x[i]), eachindex(x[1:end]))),
        count_equal=:((x) -> map(i -> count(t -> t == i, x), x)),
        count_less=:((x) -> map(i -> count(t -> t < i, x), x)),
        count_great=:((x) -> map(i -> count(t -> t > i, x), x)),
        count_bounding_val=:((x; val = 0) -> map(i -> count(t -> t >= i && t <= i + val, x), x))
    )
)

const Arithmetic = LayerCore(
    :Arithmetic,
    true,
    (:(AbstractVector{<:AbstractVector{<:Real}}),) => AbstractVector{Int},
    (
        sum=:((x) -> sum(x)),
        product=:((x) -> reduce(.*, x))
    )
)

const Aggregation = LayerCore(
    :Aggregation,
    true,
    (:(AbstractVector{<:Real}),) => T where {T<:Real},
    (
        sum=:((x) -> sum(x)),
        count_positive=:((x) -> count(i -> i > 0, x))
    )
)

const Comparison = LayerCore(
    :Comparison,
    true,
    (:(Real),) => Real,
    (
        id=:((x) -> identity(x)),
        abs_val=:((x; val) -> abs(x - val)),
        val_minus_var=:((x; val) -> maximum((0, val - x))),
        var_minus_val=:((x; val) -> maximum((0, x - val))),
        euclidean_val=:((x; val, dom_size) -> x == val ? 0 : (1 + (abs(x - val) / dom_size))),
        euclidean=:((x; dom_size) -> x == 0 ? 0 : (1 + (x / dom_size))),
        var_minus_numvars=:((x; numvars) -> abs(x - numvars)),
        max_numvars_minus_var=:((x; numvars) -> maximum((0, numvars - x))),
        max_var_minus_numvars=:((x; numvars) -> maximum((numvars - x, 0)))
    )
)

# TODO: Add more operations in comparison
