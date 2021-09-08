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
