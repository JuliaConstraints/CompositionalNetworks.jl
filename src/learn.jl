"""
    partial_search_space(domains, concept, param = nothing; sol_number = 100)

Search a part of a search space and returns a pair of vector of configurations: `(solutions, non_solutions)`.

# Arguments:
- `domains`: a collection of domains
- `concept`: the concept of the targeted constraint
- `param`: an optional parameter of the constraint
- `sol_number`: the required number of solutions (half of the number of configurations), default to `100`
"""
function partial_search_space(domains, concept, param=nothing; sol_number=100)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    while length(solutions) < 100 || length(non_sltns) < 100
        config = map(rand, domains)
        c = concept(config; param = param)
        c && length(solutions) < 100 && push!(solutions, config)
        !c && length(non_sltns) < 100 && push!(non_sltns, config)
    end
    return solutions, non_sltns
end

"""
    complete_search_space(domains, concept, param = nothing)

Search a whole search space and returns a pair of vector of configurations: `(solutions, non_solutions)`. Can be expensive on large space.

# Arguments:
- `domains`: a collection of domains
- `concept`: the concept of the targeted constraint
- `param`: an optional parameter of the constraint
"""
function complete_search_space(domains, concept, param=nothing)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    message = "Space size for complete search"
    space_size = prod(length, domains)

    space_size < 10^6 ? @info(message, space_size) : @warn(message, space_size)

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

"""
    explore_learn_compose(concept; domains, param = nothing, search = :complete, global_iter = 10, local_iter = 100, metric = hamming, popSize = 200, action = :composition)

Explore a search space, learn a composition from an ICN, and compose an error function.

# Arguments:
- `concept`: the concept of the targeted constraint
- `domains`: domains of the variables that define the training space
- `param`: an optional parameter of the constraint
- `search`: either `:partial` or `:complete` search
- `global_iter`: number of learning iteration
- `local_iter`: number of generation in the genetic algorithm
- `metric`: the metric to measure the distance between a configuration and known solutions
- `popSize`: size of the population in the genetic algorithm
- `action`: either `:symbols` to have a description of the composition or `:composition` to have the composed function itself
"""
function explore_learn_compose(concept; domains, param=nothing,
    search=:complete, global_iter=10, local_iter=100, metric=hamming, popSize=200,
    action=:composition,
)
    dom_size = maximum(length, domains)
    if search == :complete
        X_sols, X = complete_search_space(domains, concept, param)
        union!(X, X_sols)
        return learn_compose(X, X_sols, dom_size, param;
            local_iter=local_iter, global_iter=global_iter, action=action)
    end
end

"""
    compose_to_string(symbols, name)

Return a string that describes mathematically the composition of an ICN.
"""
function compose_to_string(symbols, name)
    @assert length(symbols) == 4 "Length of the decomposition ≠ 4"
    tr_length = length(symbols[1])

    CN = "CompositionalNetworks."
    tr = reduce_symbols(symbols[1], ", "; prefix=CN * "_tr_")
    ar = reduce_symbols(symbols[2], ", ", false; prefix=CN * "_ar_")
    ag = reduce_symbols(symbols[3], ", ", false; prefix=CN * "_ag_")
    co = reduce_symbols(symbols[4], ", ", false; prefix=CN * "_co_")

    julia_string = """
    function $name(x; param=nothing, dom_size)
        fill(x, $tr_length) .|> map(f -> (y -> f(y; param=param)), $tr) |> $ar |> $ag |> (y -> $co(y; param=param, dom_size=dom_size, nvars=length(x)))
    end
    """

    return julia_string
end

"""
    compose_to_file!(icn::ICN, name, path, language = :Julia)

Compose a string that describes mathematically the composition of an ICN and write it to a file.

# Arguments:
- `icn`: a given compositional network with a learned composition
- `name`: name of the composition
- `path`: path of the output file
- `language`: targeted programming language
"""
function compose_to_file!(icn::ICN, name, path, language=:Julia)
    language == :Julia # TODO: handle other languages
    file = open(path, "w")
    write(file, compose_to_string(compose(icn, action=:symbols), name))
    close(file)
end

"""
    compose_to_file!(concept, name, path; domains, param = nothing, language = :Julia, search = :complete, global_iter = 10, local_iter = 100, metric = hamming, popSize = 200)

Explore, learn and compose a function and write it to a file.

# Arguments:
- `concept`: the concept to learn
- `name`: the name to give to the constraint
- `path`: path of the output file
# Keywords arguments:
- `domains`: domains that defines the search space
- `param`: an optional paramater of the constraint
- `language`: the language to export to, default to `:julia`
- `search`: either `:partial` or `:complete` search
- `global_iter`: number of learning iteration
- `local_iter`: number of generation in the genetic algorithm
- `metric`: the metric to measure the distance between a configuration and known solutions
- `popSize`: size of the population in the genetic algorithm
"""
function compose_to_file!(concept, name, path;
    domains, param=nothing, language=:Julia,
    search=:complete, global_iter=10, local_iter=100, metric=hamming, popSize=200
)
    language == :Julia # TODO: handle other languages

    symbols = explore_learn_compose(concept, domains=domains, param=param, search=search,
        global_iter=global_iter, local_iter=local_iter , metric=metric, popSize=popSize,
        action=:symbols)

    file = open(path, "w")
    write(file, compose_to_string(symbols, name))
    close(file)
end
