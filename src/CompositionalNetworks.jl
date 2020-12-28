module CompositionalNetworks

# Imports
import Evolutionary
import Random: bitrand
import OrderedCollections: LittleDict

# Exports utilities
export lazy, lazy_param

# Export ICN
export ICN, compose

# Include utils
include("utils.jl")

# Includes layers
include("layer.jl")
include("transformation.jl")
include("arithmetic.jl")
include("aggregation.jl")
include("comparison.jl")

# Include ICN
include("icn.jl")

# Genetic Algorithm
include("genetic.jl")

end
