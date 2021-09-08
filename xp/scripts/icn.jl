using BenchmarkTools
using CompositionalNetworks
using ConstraintDomains
using Constraints
using CSV
using Tables

const DEFAULT_CONCEPTS = [
    :all_different,
    :all_equal,
    :all_equal_param,
    :dist_different,
    :eq,
    :less_than_param,
    :minus_equal_param,
    :ordered,
    :sequential_tasks,
    :sum_equal_param,
]

const DEFAULT_DOMAINS_SIZE = [2^i for i in 2:4]

const DEFAULT_LANGUAGES = [:Julia, :C, :CPP]

const DEFAULT_METRICS = [:hamming, :manhattan]

const ALL_PARAMETERS = Dict(
    :complete_search_limit => 1000,
    :concepts => DEFAULT_CONCEPTS,
    :domains_order => DEFAULT_DOMAINS_SIZE,
    :global_iterations => Threads.nthreads(),
    :languages => [:Julia],
    :local_iterations => 100,
    :metrics => DEFAULT_METRICS,
    :population => 400,
)

function learn_search_space(
    dom_size,
    concept,
    param=nothing;
    search=:flexible,
    search_limit=1000,
    solutions_limit=100,
)
    # Define the domains by on the domain size
    domains = fill(domain(1:dom_size), dom_size)

    # # Determine if the search is partial or complete
    if search == :flexible
        search = sum(domain_size, domains) < search_limit ? :complete : :partial
    end

    # Define the output folder and make the related path if necessary
    output_folder = joinpath(datadir("search_spaces"))
    mkpath(output_folder)

    # Define the file names TODO: make better metaprogramming
    name_solutions = ""
    name_non_sltns = ""
    if search == :complete
        configurations = :solutions
        name_solutions = savename(@dict concept param dom_size search configurations)
        configurations = :non_sltns
        name_non_sltns = savename(@dict concept param dom_size search configurations)
    else
        configurations = :solutions
        name_solutions = savename(
            @dict concept param dom_size search search_limit solutions_limit configurations
        )
        configurations = :non_sltns
        name_non_sltns = savename(
            @dict concept param dom_size search search_limit solutions_limit configurations
        )
    end
    file_solutions = joinpath(output_folder, "$name_solutions.csv")
    file_non_sltns = joinpath(output_folder, "$name_non_sltns.csv")

    # Check if existing data are present
    has_data = isfile(file_solutions) && isfile(file_non_sltns)

    function read_csv_as_set(file)
        configs = Set{Vector{Int}}()
        for r in CSV.Rows(file; header=false, type=Int)
            push!(configs, collect(Int, r))
        end
        return configs
    end

    # Load or compute the exploration of the search space
    solutions, non_sltns = if has_data
        read_csv_as_set(file_solutions), read_csv_as_set(file_non_sltns)
    else
        explore(domains, concept, param; search, search_limit, solutions_limit)
    end

    # Save the data locally if generated on this run
    if !has_data
        files = [file_solutions, file_non_sltns]
        configs = [solutions, non_sltns]
        for (file, config) in zip(files, configs), x in config
            CSV.write(file, Tables.table(reshape(x, (1, length(x)))); append=true)
        end
    end

    return solutions, non_sltns
end
