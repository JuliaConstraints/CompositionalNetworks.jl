module CompositionalNetworks

# Usings
using ConstraintDomains
using Dictionaries
using Evolutionary
using OrderedCollections
using Random
using ThreadPools

# Exports utilities
export csv2space
export lazy, lazy_param
export hamming
export regularization

# Export ICN
export ICN
export optimize!
export compose, show_composition
export compose_to_file!, explore_learn_compose, learn_compose
export transformation_layer, arithmetic_layer, aggregation_layer, comparison_layer
export show_layers

# Include utils
include("utils.jl")
include("metrics.jl")

# Includes layers
include("layer.jl")
include("transformation.jl")
include("arithmetic.jl")
include("aggregation.jl")
include("comparison.jl")

# Include ICN
include("icn.jl")

# Genetic Algorithm and learning
include("genetic.jl")
include("learn.jl")

end
