using TestItemRunner
using TestItems

@run_package_tests

@testitem "ICN: genetic algo" tags = [:icn, :genetic] default_imports=false begin
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
    include("icn.jl")
end
