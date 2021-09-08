module ICNBenchmarks

# usings
using BenchmarkTools
using CompositionalNetworks
using ConstraintDomains
using Constraints: Constraints, Constraint, make_error, usual_constraints
using CSV
using DrWatson
using Tables

# constants
export ALL_PARAMETERS
export BENCHED_CONSTRAINTS
export DEFAULT_CONCEPTS
export DEFAULT_DOMAINS_SIZE
export DEFAULT_LANGUAGES
export DEFAULT_METRICS

# others
export search_space
export usual_constraints

# includes
include("constants.jl")
include("search_space.jl")
include("extra_constraints.jl")

end
