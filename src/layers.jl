export AbstractLayer, Transformation, Aggregation, LayerCore, Arithmetic, Comparison

abstract type AbstractLayer end

struct LayerCore <: AbstractLayer
	name::Symbol
	mutex::Bool
	argtype::Pair
    fnexprs::NamedTuple{names,T} where {names, T <: Tuple{Vararg{<:Union{Symbol, JLFunction}}}}
	fn::NamedTuple{names,T} where {names, T <: Tuple{Vararg{Function}}}
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
		id = :((x) -> identity(x)),
		count_e_r = :((x) -> map(i -> count(t -> t == x[i], @view(x[i+1:end])), eachindex(x))),
		count_l_r = :((x) -> map(i -> count(t -> t < x[i], @view(x[i+1:end])), eachindex(x))),
		count_g_r = :((x) -> map(i -> count(t -> t > x[i], @view(x[i+1:end])), eachindex(x))),
		count_e_l = :((x) -> map(i -> count(t -> t == x[i], @view(x[1:i-1])), eachindex(x))),
		count_l_l = :((x) -> map(i -> count(t -> t < x[i], @view(x[1:i-1])), eachindex(x))),
		count_g_l = :((x) -> map(i -> count(t -> t > x[i], @view(x[1:i-1])), eachindex(x))),
		count_e_val = :((x; val = 0) -> map(i -> count(t -> t == i + val, x), x)),
		count_l_val = :((x; val = 0) -> map(i -> count(t -> t < i + val, x), x)),
		count_g_val = :((x; val = 0) -> map(i -> count(t -> t > i + val, x))),
		max_val_pos = :((x; val = 0) -> map(i -> max(0, i - val), x)),
		max_val_neg = :((x; val = 0) -> map(i -> max(0, val - i), x)),
		max_pos = :((x) -> map(i -> max(0, x[i] - x[i+1]), eachindex(x))),
		max_neg = :((x) -> map(i -> max(0, x[i+1] - x[i]), eachindex(x))),
		count_e = :((x) -> map(i -> count(t -> t == i, x), x)),
		count_l = :((x) -> map(i -> count(t -> t < i, x), x)),
		count_g = :((x) -> map(i -> count(t -> t > i, x), x)),
		count_ge_le_val = :((x; val = 0) -> map(i -> count(t -> t >= i && t <= i + val, x), x))
	)
)

const Arithmetic = LayerCore(
	:Arithmetic,
	true,
	(:(AbstractVector{<:AbstractVector{<:Real}}),) => AbstractVector{Int},
	(
		sum = :((x) -> sum(x)),
		product = :((x) -> reduce(.*, x))
	)
)

const Aggregation = LayerCore(
	:Aggregation,
	true,
	(:(AbstractVector{<:Real}),) => T where T <: Real,
	(
		sum = :((x) -> sum(x)),
		count_0 = :((x) -> count(i -> i > 0, x))
	)
)

const Comparison = LayerCore(
	:Comparison,
	true,
	(:(Real),) => Real,
	(
		id = :((x) -> identity(x)),
		abs_param = :((x; param) -> abs(x - param)),
		max_param_g = :((x; param) -> maximum((0, param - x))),
		max_param_l = :((x; param) -> maximum((0, x - param))),
	)
)

# TODO: Add more operations in comparison
