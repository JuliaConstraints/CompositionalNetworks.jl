abstract type AbstractLayer end

struct LayerCore <: AbstractLayer
    name::Symbol
    mutex::Bool
    argtype::Pair
    fnexprs::NamedTuple{
        names, T} where {names, T <: Tuple{Vararg{<:Union{Symbol, JLFunction}}}}
    fn::NamedTuple{names, T} where {names, T <: Tuple{Vararg{Function}}}
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
