abstract type AbstractLayer end

struct LayerCore{N, Q} <: AbstractLayer
	mutex::Bool
    fnexprs::NamedTuple{names,T} where {names, T <: Tuple{Vararg{<:Union{Symbol, JLFunction}}}}
	fn::NamedTuple{names,T} where {names, T <: Tuple{Vararg{Q}}}
	function LayerCore(name::Symbol, mutex::Bool, Q, fnexprs)
		@assert Q <: FunctionWrapper
		fnexprs = map(x -> JLFunction(x), fnexprs)
		for jlexp in fnexprs
			if isnothing(jlexp.kwargs)
				jlexp.kwargs = [:(params...)]
			else
				push!(jlexp.kwargs, :(params...))
			end
		end
		new{name, Q}(mutex, fnexprs, map(x -> Q(eval(codegen_ast(x))), fnexprs))
	end
end

const Transformation = LayerCore(
	:Transformation,
	false,
	FW{AbstractVector{<:Real}, Tuple{AbstractVector{<:Real}}},
	(
		id = :identity,
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
	FW{AbstractVector{<:Real}, Tuple{AbstractVector{<:AbstractVector{<:Real}}}},
	(
		
	)
)

const Aggregation = LayerCore(
	:Aggregation,
	true,
	FW{<:Real, Tuple{AbstractVector{<:Real}}},
	(
		
	)
)

const Comparison = LayerCore(
	:Comparison,
	true,
	FW{<:Real, Tuple{<:Real}},
	(
		
	)
)


