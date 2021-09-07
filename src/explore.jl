"""
    explore(domains, concept, param = nothing; search_limit = 1000, solutions_limit = 100)

Search (a part of) a search space and returns a pair of vector of configurations: `(solutions, non_solutions)`. If the search space size is over `search_limit`, then both `solutions` and `non_solutions` are limited to `solutions_limit`.

Beware that if the density of the solutions in the search space is low, `solutions_limit` needs to be reduced. This process will be automatic in the future (simple reinforcement learning).

# Arguments:
- `domains`: a collection of domains
- `concept`: the concept of the targeted constraint
- `param`: an optional parameter of the constraint
- `sol_number`: the required number of solutions (half of the number of configurations), default to `100`
"""
function explore(domains, concept, param=nothing, search=:flexible; search_limit=1000, solutions_limit=100)
    σ = sum(domain_size, domains) < search_limit ? :complete : :partial
    return explore(Val(σ), domains, concept, param, solutions_limit)
end

function explore(::Val{:partial}, domains, concept, param, limit)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    while length(solutions) < limit || length(non_sltns) < limit
        if concept(map(rand, domains); param)
            length(solutions) < 100 && push!(solutions, config)
        else
            length(non_sltns) < 100 && push!(non_sltns, config)
        end
    end
    return solutions, non_sltns
end

function explore(::Val{:complete}, domains, concept, param, ::Int)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    f = isnothing(param) ? ((x; param = p) -> concept(x)) : concept

    configurations = Base.Iterators.product(map(d -> get_domain(d), domains)...)
    foreach(
        c -> (cv = collect(c); push!(f(cv; param) ? solutions : non_sltns, cv)),
        configurations,
    )

    return solutions, non_sltns
end
