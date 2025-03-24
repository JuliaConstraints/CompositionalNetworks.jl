module CompositionalNetworks

# SECTION - Imports
import ConstraintCommons: incsert!, extract_parameters, USUAL_CONSTRAINT_PARAMETERS
import ConstraintDomains: explore, SetDomain
import Dictionaries: Dictionary, set!
import Distances
import ExproniconLite: JLFunction, has_symbol, codegen_ast, xtuple, sprint_expr
import JuliaFormatter: SciMLStyle, format_text
import OrderedCollections: LittleDict
import Random: bitrand
import TestItems: @testitem
import Unrolled: @unroll

# SECTION - Exports
export hamming, minkowski, manhattan, weights_bias
export AbstractOptimizer, GeneticOptimizer, LocalSearchOptimizer, optimize!
export generate_configurations, explore_learn
export AbstractLayer, Transformation, Aggregation, LayerCore, Arithmetic, Comparison, SimpleFilter, PairedMap
export AbstractSolution, Solution, NonSolution, Configuration, Configurations, solutions
export AbstractICN, check_weights_validity, generate_new_valid_weights, apply!, evaluate, ICN, create_icn
export compose

# SECTION - Includes
# layers
include("layer.jl")
include("layers/aggregation.jl")
include("layers/arithmetic.jl")
include("layers/comparison.jl")
include("layers/simple_filter.jl")
include("layers/pairedmap.jl")
include("layers/transformation.jl")

# optimization
include("configuration.jl")
include("icn.jl")
include("optimizer.jl")
include("learn_and_explore.jl")
include("metrics.jl")
include("compose.jl")

end
