module CompositionalNetworks

# Usings
using ConstraintDomains
using Dictionaries
using Evolutionary
using OrderedCollections
using Random
using ThreadPools
using Unrolled

# Exports utilities
export hamming
export lazy
export lazy_param
export manhattan
export max_icn_length
export minkowski
export regularization

# Export ICN
export ICN
export aggregation_layer
export arithmetic_layer
export comparison_layer
export compose
export compose_to_file!
export explore_learn_compose
export learn_compose
export optimize!
export show_composition
export show_layers
export transformation_layer

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
