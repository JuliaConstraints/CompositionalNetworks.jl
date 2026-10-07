abstract type AbstractLayer end

# const AbstractLayerInput{T, N} = Union{AbstractVector{T}, NTuple{T, N}} # consider this in the future

struct LayerCore{Q, E, F} <: AbstractLayer
    name::Symbol
    mutex::Bool
    argtype::Q
    fnexprs::E
    fn::F
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
        functions = map(x -> eval(codegen_ast(x)), fnexprs)
        new{typeof(Q), typeof(fnexprs), typeof(functions)}(
            name, mutex, Q, fnexprs, functions
        )
    end
end
