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
export ICN, compose, show_layers, show_composition, optimize!, optimize_and_compose

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

"""
    optimize_and_compose(;
        nvars, dom_size, param=nothing, icn=ICN(nvars, dom_size, param),
        X, X_sols, global_iter=100, local_iter=100, metric=hamming, popSize=200
    )
Create an ICN, optimize it, and return its composition.
"""
function optimize_and_compose(; nvars, dom_size, param=nothing,
    X=[], X_sols=[], global_iter=100, local_iter=100, metric=hamming, popSize=200
)
    icn = ICN(nvars=nvars, dom_size=dom_size, param=param)
    optimize!(icn, X, X_sols, global_iter, local_iter; metric=metric, popSize=200)
    @info show_composition(icn)
    return compose(icn)
end

end
