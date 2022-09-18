abstract type AbstractOptimizer end

struct GeneticOptimizer <: AbstractOptimizer
    global_iter::Int
    local_iter::Int
    memoize::Bool
    pop_size::Int
    sampler::Union{Nothing, Function}
end

function GeneticOptimizer(;
    global_iter=Threads.nthreads(),
    local_iter=64,
    memoize=false,
    pop_size=64,
    sampler=nothing,
)
    return GeneticOptimizer(global_iter, local_iter, memoize, pop_size, sampler)
end

function optimize!(
    icn, solutions, non_sltns, dom_size, param, metric, optimizer::GeneticOptimizer
)
    return optimize!(
        icn,
        solutions,
        non_sltns,
        optimizer.global_iter,
        optimizer.local_iter,
        dom_size,
        param,
        metric,
        optimizer.pop_size;
        optimizer.sampler,
        optimizer.memoize,
    )
end
