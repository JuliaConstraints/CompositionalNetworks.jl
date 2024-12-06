export AbstractOptimizer, GeneticOptimizer, optimize!

abstract type AbstractOptimizer end

struct GeneticOptimizer <: AbstractOptimizer
    global_iter::Int
    local_iter::Int
    memoize::Bool
    pop_size::Int
    sampler::Union{Nothing,Function}
end

function optimize! end
