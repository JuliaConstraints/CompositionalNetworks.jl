@testset "Aqua.jl" begin
    import Aqua
    import CompositionalNetworks

    Aqua.test_all(CompositionalNetworks; deps_compat = false)

    @testset "Piracies: CompositionalNetworks" begin
        Aqua.test_piracies(CompositionalNetworks;)
    end

    @testset "Dependencies compatibility (no extras)" begin
        Aqua.test_deps_compat(
            CompositionalNetworks;
            check_extras = false            # ignore = [:Random]
        )
    end
end
