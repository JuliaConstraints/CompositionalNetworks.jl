module ICNBenchmarks

# usings
using BenchmarkTools
using CompositionalNetworks
using ConstraintDomains
using Constraints
using CSV
using DrWatson
using Tables

# constants
export ALL_PARAMETERS
export DEFAULT_CONCEPTS
export DEFAULT_DOMAINS_SIZE
export DEFAULT_LANGUAGES
export DEFAULT_METRICS

# others
export search_space

# includes
include("constants.jl")
include("search_space.jl")

end
