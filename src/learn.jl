function _partial_search_space(nvars, domain, concept; sol_number=100)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    while length(solutions) < 100 || length(non_sltns) < 100 
        config = rand(1:domain, nvars)
        c = concept(config)
        c && length(solutions) < 100 && push!(solutions, config)
        !c && length(non_sltns) < 100 && push!(non_sltns, config)
    end
    return solutions, non_sltns
end

function _complete_search_space(nvars, domain, concept)
    solutions = Set{Vector{Int}}()
    non_sltns = Set{Vector{Int}}()

    message = "Space size for complete search"
    space_size = domain^nvars

    if space_size < 10^6
        @info message space_size
    else
        @warn message space_size
    end

    configurations = product(ntuple(i -> 1:domain, nvars)...)
    foreach(c -> (cv = collect(c); push!(concept(cv) ? solutions : non_sltns, cv)), configurations)
        
    return solutions, non_sltns
end

"""
    optimize_and_compose(;
        nvars, dom_size, param=nothing, icn=ICN(nvars, dom_size, param),
        X, X_sols, global_iter=100, local_iter=100, metric=hamming, popSize=200
    )
Create an ICN, optimize it, and return its composition.
"""
function opt_compose(X, X_sols; nvars, dom_size, param=nothing,
    global_iter=10, local_iter=100, metric=hamming, popSize=200)
    icn = ICN(nvars=nvars, dom_size=dom_size, param=param)
    optimize!(icn, X, X_sols, global_iter, local_iter; metric=metric, popSize=200)
    @info show_composition(icn)
    return compose(icn)
end

function learn_opt_compose(concept; nvars, dom_size, param=nothing,
    search=:complete, global_iter=10, local_iter=100, metric=hamming, popSize=200
)
    if search == :complete
        X_sols, X = _complete_search_space(nvars, dom_size, concept)
        union!(X, X_sols)
        return opt_compose(X, X_sols; nvars=nvars, dom_size = dom_size)
    end
end