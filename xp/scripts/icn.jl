using Pkg
Pkg.add("DrWatson")

# Load DrWatson (scientific project manager)
using DrWatson

# Activate the ICNBenchmarks project
@quickactivate "ICNBenchmarks"

# Pkg.instantiate()
# Pkg.update()

# Load common code to all script in ICNBenchmarks
using ICNBenchmarks

# Load other packages
using BenchmarkTools
using CompositionalNetworks
using ConstraintDomains
using Constraints

# NOTE - Is write_benchmarks useful?
# function write_benchmarks(path, data)
#     file = open(path, "a")
#     write(file, data)
#     return close(file)
# end

function icn_benchmark_unit(params)
    @info "Running a benchmark unit with" params

    search_space_size = params[:domains_size]^params[:domains_size]
    if params[:search] == :complete
        if search_space_size > params[:complete_search_limit]
            @warn "Unit benchmark aborted (complete) search space is too large" search_space_size params[:complete_search_limit]
            return nothing
        end
    end
    if params[:search] == :partial
        if search_space_size < params[:partial_search_limit]
            @warn "Unit benchmark aborted (partial) search space is too small" search_space_size params[:partial_search_limit]
            return nothing
        end
    end

    # Generate an appropriate parameter for the concept if relevant
    param = if isnothing(params[:concept][2])
        nothing
    elseif params[:concept][2] == 1
        rand(1:params[:domains_size])
    else
        rand(1:params[:domains_size], params[:concept][2])
    end

    # assign parameters
    constraint_concept = concept(BENCHED_CONSTRAINTS[params[:concept][1]])
    metric = params[:metric]
    domain_size = params[:domains_size]
    domains = fill(domain(1:domain_size), domain_size)
    func_name = "icn" * string(constraint_concept)[8:end] * "_" * string(metric)
    func_path = datadir("compositions", func_name * ".jl")

    # Time the data retrieval/generation
    t = @timed search_space(
        domain_size,
        constraint_concept,
        param;
        search=params[:search],
        complete_search_limit=params[:complete_search_limit],
        solutions_limit=params[:sampling],
    )
    solutions, non_sltns, has_data = t.value

    bench = @timed explore_learn_compose(
        domains,
        constraint_concept,
        param;
        global_iter=params[:icn_iterations],
        local_iter=params[:generations],
        metric=metric,
        pop_size=params[:population],
        configurations=(solutions, non_sltns),
    )
    compo, icn, all_compos = bench.value

    results = Dict{Symbol,Any}()
    # Code composition
    for lang in (params[:language], :maths)
        push!(results, lang => CompositionalNetworks.code(compo, lang))
    end
    push!(results, :symbols => CompositionalNetworks.symbols(compo))

    @info "Temp results" results has_data t.time bench.time all_compos
end

function icn_benchmark(params=ALL_PARAMETERS)
    # Ensure the folders for data output exist
    mkpath(datadir("compositions"))

    # Run all the benchmarks for all the unit configuration from params
    configs = dict_list(params)
    @warn "Number of benchmark units is $(length(configs))"
    for (u, c) in enumerate(configs)
        @info "Starting the $u/$(length(configs)) benchmark unit"
        icn_benchmark_unit(c)
    end
    return
end

icn_benchmark()
