module CompositionalNetworks

# SECTION - Imports
import ConstraintCommons
import ConstraintCommons: AbstractLanguage, incsert!, extract_parameters,
                          USUAL_CONSTRAINT_PARAMETERS
import ConstraintDomains: explore, DiscreteDomain, domain_size
import Dictionaries: AbstractDictionary, Dictionary, set!
import Distances
import ExproniconLite: JLFunction, has_symbol, codegen_ast, xtuple, sprint_expr
import JuliaFormatter: SciMLStyle, format_text
import OrderedCollections: LittleDict
import Random: bitrand
import TestItems: @testitem

# SECTION - Exports
export hamming, minkowski, manhattan, weights_bias
export AbstractOptimizer, GeneticOptimizer, JuMPExactOptimizer, LocalSearchOptimizer,
       StructuralEnumerationOptimizer, optimize!
export generate_configurations, explore_learn
export AbstractLayer,
       Transformation, Aggregation, LayerCore, Arithmetic, Comparison, SimpleFilter,
       PairedMap, PairMask, GroupReduction, EventMap, SegmentMap, Pointwise, Language,
       WeightedIntervalSegment, ICNBranchPlan, ICNStructurePlan,
       bind_branch_parameters, layers_for_parameters, structure_for_parameters
export AbstractSolution, Solution, NonSolution, Configuration, Configurations, solutions
export AbstractICN,
       check_weights_validity, generate_new_valid_weights, apply!, evaluate,
       reduce_icn_outputs, icn_zero_set, ICN, create_icn
export Composition, AdditiveComposition, GroupedComposition, CompositionIR, CompositionLayer
export RoutedComposition, InputReference, ParameterReference, ReshapedReference, resolve_parameters
export ComposedICN, ICNComponent, ComponentReference, learnable_composition, learnable_graph, learnable_binding,
       composition_weight_blocks, zero_set_loss
export ViewReference, ViewNode, ViewICN, learnable_views, view_weight_blocks
export compose, compose_values, compose_matrix_rows, compose_parameter_rows,
       composition, composition_ir, composition_workspace, parameterized_composition,
       MatrixRowsComposition, ParameterRowsComposition, ParameterBindingComposition
export symbols, code, canonical_key, composition_to_file!
export AbstractIncrementalComposition, AbstractIncrementalCompositionState,
       AbstractIncrementalWorkspace
export IncrementalCompositionState, IncrementalWorkspace,
       AlignedPairComposition, AlignedPairCompositionState, AlignedPairWorkspace,
       ParameterBindingCompositionState, ParameterBindingWorkspace,
       CyclicIndexComposition, IndicatorIndexComposition,
       IndexRelationCompositionState, IndexRelationWorkspace, CyclicIndexWorkspace,
       FunctionalGraphComposition, FunctionalGraphCompositionState,
       FunctionalGraphWorkspace,
       LanguageDistanceComposition, LanguageDistanceCompositionState,
       PairDistanceCollisionComposition, PairDistanceCollisionCompositionState,
       PairwiseDisjunctionCompositionState, PairwiseDisjunctionWorkspace,
       EventProfileCompositionState, EventProfileWorkspace,
       MatrixRowsCompositionState, MatrixRowsWorkspace,
       ParameterRowsCompositionState, ParameterRowsWorkspace
export GroupedCompositionState
export incremental_composition, incremental_rebuild!, incremental_state
export incremental_candidate_value!, incremental_supported, incremental_update!,
       incremental_value, incremental_workspace

# SECTION - Includes
# layers
include("layer.jl")
include("layers/aggregation.jl")
include("layers/arithmetic.jl")
include("layers/comparison.jl")
include("layers/simple_filter.jl")
include("layers/pairedmap.jl")
include("layers/pairmask.jl")
include("layers/group_reduction.jl")
include("layers/eventmap.jl")
include("layers/segmentmap.jl")
include("layers/pointwise.jl")
include("layers/language.jl")
include("layers/transformation.jl")

# optimization
include("configuration.jl")
include("icn.jl")
include("optimizer.jl")
include("exact_optimizer.jl")
include("learn_and_explore.jl")
include("metrics.jl")
include("inplace.jl")
include("compose.jl")
include("composition_ir.jl")
include("composition_routing.jl")
include("composed_icn.jl")
include("view_icn.jl")
include("incremental.jl")

end
