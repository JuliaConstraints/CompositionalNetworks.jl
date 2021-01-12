function _partial_search_space(domains, concept; sol_number=100)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    while length(solutions) < 100 || length(non_sltns) < 100 
        config = map(_draw, domains)
        c = concept(config)
        c && length(solutions) < 100 && push!(solutions, config)
        !c && length(non_sltns) < 100 && push!(non_sltns, config)
    end
    return solutions, non_sltns
end

function _complete_search_space(domains, concept)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    message = "Space size for complete search"
    space_size = prod(_length, domains)

    if space_size < 10^6
        @info message space_size
    else
        @warn message space_size
    end

    configurations = product(map(d -> _get_domain(d), domains)...)
    foreach(c -> (cv = collect(c); push!(concept(cv) ? solutions : non_sltns, cv)), configurations)
        
    return solutions, non_sltns
end

"""
    learn_compose(;
        nvars, dom_size, param=nothing, icn=ICN(nvars, dom_size, param),
        X, X_sols, global_iter=100, local_iter=100, metric=hamming, popSize=200
    )
Create an ICN, optimize it, and return its composition.
"""
function learn_compose(X, X_sols; nvars, dom_size, param=nothing,
    global_iter=10, local_iter=100, metric=hamming, popSize=200,
    action=:composition
)
    icn = ICN(nvars=nvars, dom_size=dom_size, param=param)
    optimize!(icn, X, X_sols, global_iter, local_iter; metric=metric, popSize=200)
    @info show_composition(icn)

    return compose(icn, action=action)
end

function explore_learn_compose(concept; domains, param=nothing,
    search=:complete, global_iter=10, local_iter=100, metric=hamming, popSize=200,
    action=:composition,
)
    if search == :complete
        X_sols, X = _complete_search_space(domains, concept)
        union!(X, X_sols)
        return learn_compose(X, X_sols;
            nvars=length(domains), dom_size=maximum(_length, domains),
            local_iter=local_iter, global_iter=global_iter, param=param,
            action=action)
    end
end

function _compose_to_string(symbols, name)
    @assert length(symbols) == 4 "Length of the decomposition ≠ 4"
    tr_length = length(symbols[1])
    
    const CN = "CompositionalNetworks"
    tr = _reduce_symbols(symbols[1], ", "; prefix = CN * "_tr_")
    ar = _reduce_symbols(symbols[2], ", ", false; prefix = CN * "_ar_")
    ag = _reduce_symbols(symbols[3], ", ", false; prefix = CN * "_ag_")
    co = _reduce_symbols(symbols[4], ", ", false; prefix = CN * "_co_")

    julia_string = """
    $name = x -> fill(x, $tr_length) .|> $tr |> $ar |> $ag |> $co
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