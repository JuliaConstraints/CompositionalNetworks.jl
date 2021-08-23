using ConstraintDomains
using CompositionalNetworks
using BenchmarkTools

#using Distributed
#addprocs()

include("domains.jl")
include("concepts.jl")
include("params.jl")
include("../src/learn.jl")


if (isdir("xp"))
    cd("xp")
elseif (contains(pwd()[end-20:end], "CompositionalNetworks"))
    mkdir("xp")
end


# for concept in concept_list
#     for metric in metrics
concept = concept_list[1]
metric = metrics[1]
        println("$concept-$metric")
        func_name = "icn$(String(Symbol(concept))[8:end])"
        param = length(iterate(methods(concept))[1].sig.parameters) == 2 ? nothing : rand(dom)
        icn = compose_to_file!(concept, "$(func_name)_$(String(Symbol(metric)))", "$(func_name)_$(String(Symbol(metric))).jl",
                         domains=domains, param=param, global_iter=1, metric=metric)
        X_sols, X = complete_search_space(domains, concept, param)
        #fitness = w -> loss(X, X_sols, icn, w, metric, dom_size, param)
#     end
# end



#w is weights -> weights(icn)