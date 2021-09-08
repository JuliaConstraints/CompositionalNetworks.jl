const BENCHED_CONSTRAINTS = deepcopy(usual_constraints)

const DEFAULT_CONCEPTS = [
    (:all_different, nothing),
]

const DEFAULT_DOMAINS_SIZE = [2^i for i in 2:4]

const DEFAULT_LANGUAGES = [:Julia, :C, :CPP]

const DEFAULT_METRICS = [:hamming, :manhattan]

const ALL_PARAMETERS = Dict(
    :complete_search_limit => 1000,
    :concept => DEFAULT_CONCEPTS,
    :domains_size => DEFAULT_DOMAINS_SIZE,
    :global_iterations => Threads.nthreads(),
    :language => [:Julia],
    :local_iterations => 100,
    :metric => DEFAULT_METRICS,
    :population => 400,
)
