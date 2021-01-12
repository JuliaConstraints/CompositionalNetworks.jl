using CompositionalNetworks
using ConstraintDomains
using Test

CN = CompositionalNetworks

@testset "CompositionalNetworks.jl" begin
    include("layers.jl")
    include("icn.jl")
end
