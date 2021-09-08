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

    # Retrieve the search space configurations
    ds = params[:domains_size]
    c = concept(BENCHED_CONSTRAINTS[params[:concept][1]])
    p = params[:concept][2]

    solutions, non_sltns = search_space(ds, c, p)

    @info "Temp results" solutions
end

# Run all the benchmarks for all the unit configuration from ALL_PARAMETERS
icn_benchmark(params = ALL_PARAMETERS) = foreach(icn_benchmark_unit, dict_list(params))

icn_benchmark()
