module ICNBenchmarks

# usings
using BenchmarkTools
using CompositionalNetworks
using ConstraintDomains
using Constraints
using CSV
using DrWatson
using Tables

# imports
import Constraints: make_error

# constants
export ALL_PARAMETERS
export BENCHED_CONSTRAINTS

# others
export search_space

# includes
include("constants.jl")
include("search_space.jl")
include("extra_constraints.jl")

end
