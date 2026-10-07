@testset "Code linting (JET.jl)" begin
    if pkgversion(JET) < v"0.12"
        JET.test_package(CompositionalNetworks; target_defined_modules = true)
    else
        JET.test_package(CompositionalNetworks;
            target_modules = (CompositionalNetworks,))
    end
end
