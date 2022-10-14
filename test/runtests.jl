using CompositionalNetworks
using ConstraintDomains
using Dictionaries
using Evolutionary
using Memoization
using Test
using ThreadPools

CN = CompositionalNetworks

import CompositionalNetworks: AbstractOptimizer

include("genetic.jl")

@testset "CompositionalNetworks.jl" begin
    include("layers.jl")
    include("icn.jl")
end
