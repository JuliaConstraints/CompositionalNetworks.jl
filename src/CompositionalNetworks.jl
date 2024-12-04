module CompositionalNetworks

# imports
import ConstraintCommons: incsert!, extract_parameters
import ConstraintDomains: explore
import Dictionaries: Dictionary, set!
import Distances
import JuliaFormatter: SciMLStyle, format_text
import OrderedCollections: LittleDict
import Random: bitrand
import TestItems: @testitem
import Unrolled: @unroll
import ExproniconLite: JLFunction, has_symbol

#=
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
export explore_learn_compose
export hamming
export incsert!
export lazy
export lazy_param
export learn_compose
export manhattan
export max_icn_length
export minkowski
export nbits
export optimize!
export regularization
export show_layers
export symbols
export transformation_layer
export weights
export weights!
export weights_bias
=#

#=
# Include utils
include("utils.jl")
include("metrics.jl")

# Includes layers
include("layer.jl")
include("layers/transformation.jl")
include("layers/arithmetic.jl")
include("layers/aggregation.jl")
include("layers/comparison.jl")

# Include ICN
include("icn.jl")
include("composition.jl")
include("learn.jl")
=#

include("learn_and_explore.jl")
include("configuration.jl")
include("layers.jl")
include("icn.jl")
include("metrics.jl")

end
