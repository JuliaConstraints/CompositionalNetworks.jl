abstract type AbstractOptimizer end

struct GeneticOptimizer <: AbstractOptimizer
    global_iter::Int
    local_iter::Int
    memoize::Bool
    pop_size::Int
    sampler::Union{Nothing,Function}
end

function optimize!(icn, configurations, metric_function, optimizer_config; parameters...)
    error("No backend loaded")
end
