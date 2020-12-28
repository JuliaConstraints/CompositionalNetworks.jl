module CompositionalNetworks

# Imports
import Evolutionary
import Random: bitrand

# Exports utilities
export lazy, lazy_param

# Include utils
include("utils.jl")

# Includes layers
include("transformation.jl")
include("arithmetic.jl")
include("aggregation.jl")
include("comparison.jl")

# Genetic Algorithm
include("genetic.jl")

end
