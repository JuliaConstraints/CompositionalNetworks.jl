"""
Generate a julia function for a given ICN

Example usage:
```julia
compose(ICN(), name = :hopefullyworkingfunction)
```
"""
function compose(icn::AbstractICN; name::Symbol=gensym(), jlfun=true, fname="")
    f = JLFunction()
    f.name = name
    f.args = [:x]
    f.kwargs = append!(collect(icn.parameters), collect(keys(icn.constants)))

    fns = []
    _start = 1
    weights = icn.weights.parent
    for (i, layer) in enumerate(icn.layers)
        j = findall(weights[_start:(_start-1+length(layer.fn))])
        if layer.mutex
            push!(fns, :($(layer.name) = x = $(layer.fnexprs[j[1]].body)))
        else
            temp = xtuple([layer.fnexprs[k].body for k in j]...)
            push!(fns, :($(layer.name) = x = $(temp) |> collect))
        end
        _start += length(layer.fn)
    end
    f.body = Expr(:block, push!(fns, :(return x))...)
    if !isempty(fname)
        open(fname, "w") do fio
            write(fio, sprint_expr(f))
        end
    end
    return if jlfun
        f
    else
        codegen_ast(f)
    end
end
