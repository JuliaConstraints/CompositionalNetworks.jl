using Unrolled
using ConstraintDomains
using CompositionalNetworks
using BenchmarkTools

using Distributed

# if nworkers == 1
#     addprocs()
# end

# println("N-workers: $(nworkers())")
n_threads = Threads.nthreads()
println("N-THREADS: $n_threads")

include("domains.jl")
include("concepts.jl")
include("params.jl")
include("helpers.jl")



if (isdir("xp"))
    cd("xp")
    cd("compositions")
elseif (contains(pwd()[end-20:end], "CompositionalNetworks"))
    mkdir("xp")
    cd("xp")
    mkdir("compositions")
    cd("compositions")
end


function write_benchmarks(path, data)
    file = open(path, "a")
    write(file, data)
    close(file)
end

C = 640
global_iters = n_threads
local_iters = C ÷ global_iters

#for concept in concept_list
#    for metric in metrics

concept = concept_list[1]
metric = metrics[2]
        m = metric
        println("$concept-$metric")
        func_name = "icn$(String(Symbol(concept))[8:end])_$(String(Symbol(metric)))"
        # TODO: Fix param value
        param = length(iterate(methods(concept))[1].sig.parameters) == 2 ? nothing : rand(dom)
        path = "$(func_name)_$n_threads.jl"
        icn = ICN()
        g_b = @benchmark ($icn = compose_to_file!($concept, "$($func_name)", $path,
                         domains=domains, param=$param, global_iter=global_iters,local_iter=local_iters ,  metric=$metric)) samples = 2

        g_b_time = BenchmarkTools.prettytime(time(g_b))
        g_b_memory = BenchmarkTools.prettymemory(memory(g_b))
        g_b_allocs = allocs(g_b)
    
        write_benchmarks(path, "#Finding composition:\n#$g_b_time ($g_b_allocs allocation$(g_b_allocs == 1 ? "" : "s"): $g_b_memory)")

        X_sols, X = complete_search_space(domains, concept, param)
        #loss_value = fitness(icn.weigths)
        loss_value = fitness(weigths(icn), X, X_sols, icn, metric, param)
        include("compositions/$path")
        #ef = getfield(Main, Symbol(path[1:end-3])) #function name from generated composition file
        ef = getfield(Main, Symbol(path[1:end-5])) # ... minus n_threads in name
        
        benchmarks = []
        for var in union(X_sols, X)
            push!(benchmarks, @benchmark $ef($var, dom_size=length(domains[1])) samples = 2)
        end
        b_time = BenchmarkTools.prettytime(sum(time, benchmarks)/length(benchmarks))
        b_memory = BenchmarkTools.prettymemory(sum(memory, benchmarks)/length(benchmarks))
        b_allocs = sum(allocs, benchmarks)÷length(benchmarks)
        write_benchmarks(path, "\n#Use:\n#$b_time ($b_allocs allocation$(b_allocs == 1 ? "" : "s"): $b_memory)")
        write_benchmarks(path, "\n#Loss: $loss_value")

#    end
#end


#w is weights -> weights(icn)


# save learned weights aswell