abstract type AbstractOptimizer end

function optimize!(args...)
    return nothing
end

"""
    learn_compose(;
        nvars, dom_size, param=nothing, icn=ICN(nvars, dom_size, param),
        X, X_sols, global_iter=100, local_iter=100, metric=hamming, popSize=200
    )
Create an ICN, optimize it, and return its composition.
"""
function learn_compose(
    solutions,
    non_sltns,
    dom_size,
    param=nothing;
    metric=:hamming,
    optimizer,
    X_test = nothing,
)
    icn = ICN(; param=!isnothing(param))
    _, weigths = optimize!(
        icn,
        solutions,
        non_sltns,
        dom_size,
        param,
        metric,
        optimizer,
    )
    compositions = Dictionary{Composition,Int}()
    for (bv, occurences) in pairs(weigths)
        set!(compositions, compose(deepcopy(icn), bv), occurences)
    end

    return compose(icn), icn, compositions
end

"""
    explore_learn_compose(concept; domains, param = nothing, search = :complete, global_iter = 10, local_iter = 100, metric = hamming, popSize = 200, action = :composition)

Explore a search space, learn a composition from an ICN, and compose an error function.

# Arguments:
- `concept`: the concept of the targeted constraint
- `domains`: domains of the variables that define the training space
- `param`: an optional parameter of the constraint
- `search`: either `flexible`,`:partial` or `:complete` search. Flexible search will use `search_limit` and `solutions_limit` to determine if the search space needs to be partially or completely explored
- `global_iter`: number of learning iteration
- `local_iter`: number of generation in the genetic algorithm
- `metric`: the metric to measure the distance between a configuration and known solutions
- `popSize`: size of the population in the genetic algorithm
- `action`: either `:symbols` to have a description of the composition or `:composition` to have the composed function itself
"""
function explore_learn_compose(
    domains,
    concept;
    param=nothing,
    configurations=explore(domains, concept; param),
    metric=:hamming,
    optimizer,
    X_test = nothing,
)
    dom_size = maximum(length, domains)
    solutions, non_sltns = configurations
    return learn_compose(
        solutions,
        non_sltns,
        dom_size,
        param;
        metric,
        optimizer,
        X_test,
    )
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
function compose_to_file!(
    concept,
    name,
    path;
    param=nothing,
    configurations=explore(domains, concept; param),
    domains,
    language=:Julia,
    metric=:hamming,
    optimizer,
    X_test=nothing,
)
    compo, icn, _ = explore_learn_compose(
        domains,
        concept;
        configurations,
        metric,
        optimizer,
        param,
        X_test,
    )
    composition_to_file!(compo, path, name, language)
    return icn
end
