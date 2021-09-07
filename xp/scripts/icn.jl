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
    :domains_order =>DEFAULT_DOMAINS_SIZE,
    :global_iterations => Threads.nthreads(),
    :languages => [:Julia],
    :local_iterations => 100,
    :metrics => DEFAULT_METRICS,
    :population => 400,
)

function learn_search_space(domain_size, concept, param=nothing; search=:flexible, search_limit=1000, solutions_limit=100)
    search = if search == :flexible
        sum(domain_size, domains) < search_limit ? :complete : :partial
    end
    domains = fill(domain(1:domain_size))
    solutions, non_sltns = explore(domains, concept, param; search, search_limit, solutions_limit)

    name = if search == :complete
        savename(@dict domain_size concept param search)
    else
        savename(@dict domain_size concept param search search_limit solutions_limit)
    end

    output_folder = joinpath(datadir("search_spaces"), name)
    mkpath(output_folder)

    CSV.write(joinpath(output_folder, "solutions.csv"), Tables.table(solutions))
    CSV.write(joinpath(output_folder, "non_sltns.csv"), Tables.table(non_sltns))

    return solutions, non_sltns
end
