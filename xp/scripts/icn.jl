using Pkg
#Pkg.add("DrWatson")

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
using JSON

function icn_benchmark_unit(params)
    @info "Running a benchmark unit with" params

    search_space_size = params[:domains_size]^params[:domains_size]
    if params[:search] == :complete
        if search_space_size > params[:complete_search_limit]
            @warn "Unit benchmark aborted (complete) search space is too large" search_space_size params[:complete_search_limit]
            return nothing
        end
        if search_space_size > params[:loss_sampling_threshold]
            if isnothing(params[:loss_sampler])
                @warn "Unit benchmark aborted (complete) search space is too large, and loss function is deterministically evaluated" search_space_size params[:loss_sampling_threshold] params[:loss_sampler]
                return nothing
            end
        end
    end
    if params[:search] == :partial
        if search_space_size < params[:partial_search_limit]
            @warn "Unit benchmark aborted (partial) search space is too small" search_space_size params[:partial_search_limit]
            return nothing
        end
        if search_space_size ≤ params[:loss_sampling_threshold]
            if !isnothing(params[:loss_sampler])
                @warn "Unit benchmark aborted (complete) search space is too small, and loss function is stochastically evaluated" search_space_size params[:loss_sampling_threshold] params[:loss_sampler]
                return nothing
            end
        end
    end
    json_name = savename(
        (
            con=string(params[:concept][1]),
            par=params[:concept][2],
            csl=params[:complete_search_limit],
            dom=params[:domains_size],
            gen=params[:generations],
            iter=params[:icn_iterations],
            lang=string(params[:language]),
            lst=params[:loss_sampling_threshold],
            ls=string(params[:loss_sampler]),
            metric=string(params[:metric]),
            psl=params[:partial_search_limit],
            pop=params[:population],
            sampling=params[:sampling],
            search=string(params[:search]),
        ),
        "json",
    )
    save_results = joinpath(datadir("compositions"), json_name)
    if isfile(save_results)
        @warn "The result file already exist" save_results
    else

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
            metric,
            pop_size=params[:population],
            configurations=(solutions, non_sltns),
            sampler=params[:loss_sampler],
        )
        _, _, all_compos = bench.value

        results = Dict{Any,Any}()

        # Global results
        push!(results, :data => has_data ? :loaded : :explored)
        push!(results, :data_time => t.time)
        push!(results, :icn_time => bench.time)
        push!(results, :total_time => t.time + bench.time)
        push!(results, :nthreads => Threads.nthreads())

        for (id, (compo, occurence)) in enumerate(pairs(all_compos))
            local_results = Dict{Symbol,Any}()

            # selection rate
            push!(local_results, :selection_rate => occurence / params[:icn_iterations])

            # Code composition
            for lang in (params[:language], :maths)
                push!(local_results, lang => CompositionalNetworks.code(compo, lang))
            end
            push!(local_results, :symbols => CompositionalNetworks.symbols(compo))

            push!(results, :params => params)
            push!(results, id => local_results)
        end
        write(save_results, json(results, 2))
        @info "Temp results" results json_name
    end
    return nothing
end

function icn_benchmark(params=ALL_PARAMETERS; clear_results=false)
    # Ensure the folders for data output exist
    clear_results && rm(datadir("compositions"); recursive=true, force=true)
    mkpath(datadir("compositions"))

    # Run all the benchmarks for all the unit configuration from params
    configs = dict_list(params)
    @warn "Number of benchmark units is $(length(configs))"
    for (u, c) in enumerate(configs)
        @info "Starting the $u/$(length(configs)) benchmark unit"
        icn_benchmark_unit(c)
    end
    return nothing
end

# NOTE - Please use clear_results=false for the real experiments
icn_benchmark(; clear_results=true)
