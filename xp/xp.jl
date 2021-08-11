using ConstraintDomains
using CompositionalNetworks

include("domains.jl")
include("concepts.jl")

for concept in concept_list

    func_name = "icn$(String(Symbol(concept))[8:end])"
    param = length(iterate(methods(concept))[1].sig.parameters) == 2 ? nothing : rand(dom)
    compose_to_file!(concept, func_name, "$(func_name).jl", domains=domains,
                 param=param)
    
end

