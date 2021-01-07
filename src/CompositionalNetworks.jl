module CompositionalNetworks

# Imports
import Evolutionary: GA, tournament, singlepoint, flip, optimize, minimizer, Options, summary, trace, rouletteinv
import Random: bitrand, falses
import OrderedCollections: LittleDict
import DataFrames: DataFrame, Not
import CSV: CSV
import Dictionaries: Dictionary, set!

# Exports utilities
export lazy, lazy_param, csv2space
export hamming

# Export ICN
export ICN, compose, show_layers, show_composition

# Include utils
include("utils.jl")
include("io.jl")
include("hamming.jl")

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
