module CompositionalNetworks

# Usings
using ConstraintDomains
using Dictionaries
using Evolutionary
using JuliaFormatter
using OrderedCollections
using Random
using ThreadPools
using Unrolled

export Composition
export ICN
export aggregation_layer
export arithmetic_layer
export code
export comparison_layer
export compose
export compose_to_file!
export composition
export composition_to_file!
export explore
export explore_learn_compose
export hamming
export lazy
export lazy_param
export learn_compose
export manhattan
export max_icn_length
export minkowski
export optimize!
export regularization
export show_layers
export symbols
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
include("composition.jl")

# Genetic Algorithm and learning
include("genetic.jl")
include("explore.jl")
include("learn.jl")

end
