struct Composition{F<:Function}
    code::Dict{Symbol,String}
    f::F
    symbols::Vector{Vector{Symbol}}
end

function Composition(f::F, symbols) where {F<:Function}
    code = Dict{Symbol,String}()
    return Composition{F}(code, f, symbols)
end

function code(c::Composition, lang=:maths; name="composition")
    return get!(c.code, lang, generate(c, name, Val(lang)))
end
composition(c::Composition) = c.f
symbols(c::Composition) = c.symbols

"""
    compose(icn, weights=nothing)
Return a function composed by some of the operations of a given ICN. Can be applied to any vector of variables. If `weights` are given, will assign to `icn`.
"""
function compose(icn::ICN, weigths::BitVector=BitVector())
    !isempty(weigths) && weights!(icn, weigths)
    composition, symbols = _compose(icn)
    return Composition(composition, symbols)
end

function generate(c::Composition, name, ::Val{:maths})
    aux = map(s -> reduce_symbols(s, ", ", length(s) > 1), symbols(c))
    def = reduce_symbols(aux, " ∘ ", false)
    return "$name = $def"
end

function generate(c::Composition, name, ::Val{:Julia})
    symbs = symbols(c)
    @assert length(symbs) == 4 "Length of the decomposition ≠ 4"
    tr_length = length(symbs[1])

    CN = "CompositionalNetworks."
    tr = reduce_symbols(symbs[1], ", "; prefix=CN * "tr_")
    ar = reduce_symbols(symbs[2], ", ", false; prefix=CN * "ar_")
    ag = reduce_symbols(symbs[3], ", ", false; prefix=CN * "ag_")
    co = reduce_symbols(symbs[4], ", ", false; prefix=CN * "co_")

    documentation = """\"\"\"
        $name(x; X = zeros(length(x), $tr_length), param=nothing, dom_size)

    Composition `$name` generated by CompositionalNetworks.jl.
    ```
    $(code(c; name))
    ```
    \"\"\"
    """

    output = """
    function $name(x; X = zeros(length(x), $tr_length), param=nothing, dom_size)
        $(CN)tr_in(Tuple($tr), X, x, param)
        X[:, 1] .= 1:length(x) .|> (i -> $ar(@view X[i, 1:$tr_length]))
        return $ag(@view X[:, 1]) |> (y -> $co(y; param, dom_size, nvars=length(x)))
    end
    """
    return documentation * format_text(output, BlueStyle(); pipe_to_function_call=false)
end

function composition_to_file!(c::Composition, path, name, language=:Julia)
    output = code(c, language; name)
    write(path, output)
    return nothing
end