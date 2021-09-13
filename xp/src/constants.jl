# Constraints parameters
const BENCHED_CONSTRAINTS = deepcopy(usual_constraints)

const DEFAULT_CONCEPTS = [(:all_different, nothing)]
const DEFAULT_LANGUAGES = [:Julia] # [:Julia, :C, :CPP]
const DEFAULT_METRICS = [:hamming, :manhattan]
const DEFAULT_COMPLETE_SEARCH_LIMIT = 6^6 # could be improved with multithreading
const DEFAULT_PARTIAL_SEARCH_LIMIT = 4^4
const DEFAULT_SAMPLINGS = [100, 1000]

const MINIMUM_DOMAIN_SIZE = 3
const MAXIMUM_DOMAIN_SIZE = 8 # NOTE - Check if it is too high

function domains_sizes(;
    min_domain_size=MINIMUM_DOMAIN_SIZE, max_domain_size=MAXIMUM_DOMAIN_SIZE
)
    return collect(min_domain_size:max_domain_size)
end

# Genetic Algorithms default parameters
const MAXIMUM_TOTAL_ITERATIONS = 12 # order => 2^15 = 32768
const MINIMUM_ICN_ITERATIONS = 4 # order => 2^5= 32
const MINIMUM_GENERATIONS = 4 # order => 2^5 = 32

function icn_iterations(;
    min_icn_iterations=MINIMUM_ICN_ITERATIONS,
    min_generations=MINIMUM_GENERATIONS,
    max_total_iterations=MAXIMUM_TOTAL_ITERATIONS,
)
    maximum_icn_iterations = max_total_iterations - min_generations
    return [2^i for i in min_icn_iterations:2:maximum_icn_iterations]
end

function generations(;
    min_icn_iterations=MINIMUM_ICN_ITERATIONS,
    min_generations=MINIMUM_GENERATIONS,
    max_total_iterations=MAXIMUM_TOTAL_ITERATIONS,
)
    maximum_generations = max_total_iterations - min_icn_iterations
    return [2^i for i in min_generations:2:maximum_generations]
end

const DEFAULT_POPULATIONS = [2^i for i in 6:2:10] # 2^5=32 to 2^10 = 1024

const DEFAULT_LOSS_SAMPLING_THRESHOLD = 2^10

root(n) = x -> round(Int, x^(1 / n))
constant(n) = _ -> n

const DEFAULT_LOSS_SAMPLER = [
    nothing, root(2), root(3), constant(DEFAULT_LOSS_SAMPLING_THRESHOLD)
]

const DEFAULT_MEMOIZATION = [true, false]

# All parameters
const ALL_PARAMETERS = Dict(
    # Search parameters
    :concept => DEFAULT_CONCEPTS,
    :complete_search_limit => DEFAULT_COMPLETE_SEARCH_LIMIT,
    :domains_size => domains_sizes(),
    :partial_search_limit => DEFAULT_PARTIAL_SEARCH_LIMIT,
    :sampling => DEFAULT_SAMPLINGS,
    :search => [:partial, :complete], # :flexible is also an option

    # # Learning parameter
    :metric => DEFAULT_METRICS,
    :generations => generations(),
    :icn_iterations => icn_iterations(),
    :language => DEFAULT_LANGUAGES,
    :population => DEFAULT_POPULATIONS,
    :loss_sampler => DEFAULT_LOSS_SAMPLER,
    :loss_sampling_threshold => DEFAULT_LOSS_SAMPLING_THRESHOLD,
    :memoize => DEFAULT_MEMOIZATION,
)

# ALL_PARAMETERS[:search] = [:flexible]
# ALL_PARAMETERS[:domains_size] = 8
