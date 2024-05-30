@testset "Aqua.jl" begin
    import Aqua
    import CompositionalNetworks

    # TODO: Fix the broken tests and remove the `broken = true` flag
    Aqua.test_all(
        CompositionalNetworks;
        ambiguities=(broken=true,),
        deps_compat=false,
        piracies=(broken=false,),
    )

    @testset "Ambiguities: CompositionalNetworks" begin
        #     Aqua.test_ambiguities(CompositionalNetworks;)
    end

    @testset "Piracies: CompositionalNetworks" begin
        Aqua.test_piracies(CompositionalNetworks;)
    end

    @testset "Dependencies compatibility (no extras)" begin
        Aqua.test_deps_compat(
            CompositionalNetworks;
            check_extras=false,            # ignore = [:Random]
        )
    end
end
