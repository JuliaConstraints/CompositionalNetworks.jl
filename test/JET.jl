@testset "Code linting (JET.jl)" begin
    JET.test_package(CompositionalNetworks; target_defined_modules = true)
end
