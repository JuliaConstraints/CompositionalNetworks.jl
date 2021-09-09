# Load DrWatson (scientific project manager)
using DrWatson

# Activate the ICNBenchmarks project
@quickactivate "ICNBenchmarks"

# Load common code to all script in ICNBenchmarks
using ICNBenchmarks

# Load other packages
using Constraints

# Ensure the folders for data output exist
mkpath(datadir("compositions"))

# NOTE - Is write_benchmarks useful?
# function write_benchmarks(path, data)
#     file = open(path, "a")
#     write(file, data)
#     return close(file)
# end

function icn_benchmark_unit(params)
    @info "Running a benchmark unit with" params

    if params[:search] == :complete
        params[:domains_size]^params[:domains_size] > params[:complete_search_limit] && return nothing
    end
    if params[:search] == :partial
        params[:domains_size]^params[:domains_size] < params[:partial_search_limit] && return nothing
    end

    # Time the data retrieval/generation
    t = @timed search_space(
        params[:domains_size],
        concept(BENCHED_CONSTRAINTS[params[:concept][1]]),
        params[:concept][2];
        search=params[:search],
        complete_search_limit=params[:complete_search_limit],
        solutions_limit=params[:sampling],
    )
    solutions, non_sltns, has_data = t.value



    @info "Temp results" solutions has_data t.time
end

# Run all the benchmarks for all the unit configuration from ALL_PARAMETERS
icn_benchmark(params=ALL_PARAMETERS) = foreach(icn_benchmark_unit, dict_list(params))

icn_benchmark()
