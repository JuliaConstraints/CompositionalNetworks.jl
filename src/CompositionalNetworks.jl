module CompositionalNetworks

# Imports
import Evolutionary

# Exports utilities
export lazy, lazy_param

# Include utils
include("utils.jl")

# Includes layers
include("transformation.jl")
include("arithmetic.jl")
include("aggregation.jl")
include("comparison.jl")

end
