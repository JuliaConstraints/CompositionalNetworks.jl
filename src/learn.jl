function _partial_search_space(domains, concept, param=nothing; sol_number=100)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    while length(solutions) < 100 || length(non_sltns) < 100
        config = map(_draw, domains)
        c = concept(config; param = param)
        c && length(solutions) < 100 && push!(solutions, config)
        !c && length(non_sltns) < 100 && push!(non_sltns, config)
    end
    return solutions, non_sltns
end

function _complete_search_space(domains, concept, param=nothing)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    message = "Space size for complete search"
    space_size = prod(length, domains)

    if space_size < 10^6
        @info message space_size
    else
        @warn message space_size
    end

    f = isnothing(param) ? ((x; param = p) -> concept(x)) : concept

    configurations = Base.Iterators.product(map(d -> get_domain(d), domains)...)
    foreach(c -> (cv = collect(c); push!(f(cv; param=param) ? solutions : non_sltns, cv)), configurations)

    return solutions, non_sltns
end

"""
    learn_compose(;
        nvars, dom_size, param=nothing, icn=ICN(nvars, dom_size, param),
        X, X_sols, global_iter=100, local_iter=100, metric=hamming, popSize=200
    )
Create an ICN, optimize it, and return its composition.
"""
function learn_compose(X, X_sols, dom_size, param=nothing;
    global_iter=10, local_iter=100, metric=hamming, popSize=200,
    action=:composition
)
    icn = ICN(param=!isnothing(param))
    optimize!(icn, X, X_sols, global_iter, local_iter, dom_size, param; metric=metric, popSize=200)
    @info show_composition(icn)

    return compose(icn, action=action)
end

function explore_learn_compose(concept; domains, param=nothing,
    search=:complete, global_iter=10, local_iter=100, metric=hamming, popSize=200,
    action=:composition,
)
    dom_size = maximum(length, domains)
    if search == :complete
        X_sols, X = _complete_search_space(domains, concept, param)
        union!(X, X_sols)
        return learn_compose(X, X_sols, dom_size, param;
            local_iter=local_iter, global_iter=global_iter, action=action)
    end
end

function _compose_to_string(symbols, name)
    @assert length(symbols) == 4 "Length of the decomposition ≠ 4"
    tr_length = length(symbols[1])

    CN = "CompositionalNetworks."
    tr = _reduce_symbols(symbols[1], ", "; prefix=CN * "_tr_")
    ar = _reduce_symbols(symbols[2], ", ", false; prefix=CN * "_ar_")
    ag = _reduce_symbols(symbols[3], ", ", false; prefix=CN * "_ag_")
    co = _reduce_symbols(symbols[4], ", ", false; prefix=CN * "_co_")

    julia_string = """
    function $name(x; param=nothing, dom_size)
        fill(x, $tr_length) .|> map(f -> (y -> f(y; param=param)), $tr) |> $ar |> $ag |> (y -> $co(y; param=param, dom_size=dom_size, nvars=length(x)))
    end
    """

    return julia_string
end

function compose_to_file!(icn::ICN, name, path, language=:Julia)
    language == :Julia # TODO: handle other languages
    file = open(path, "w")
    write(file, _compose_to_string(compose(icn, action=:symbols), name))
    close(file)
end

function compose_to_file!(concept, name, path;
    domains, param=nothing, language=:Julia,
    search=:complete, global_iter=10, local_iter=100, metric=hamming, popSize=200
)
    language == :Julia # TODO: handle other languages

    symbols = explore_learn_compose(concept, domains=domains, param=param, search=search,
        global_iter=global_iter, local_iter=local_iter , metric=metric, popSize=popSize,
        action=:symbols)

    file = open(path, "w")
    write(file, _compose_to_string(symbols, name))
    close(file)
end
