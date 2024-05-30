using PerfChecker
using Test

using CompositionalNetworks
using ConstraintDomains

@testset "PerfChecker.jl" begin
    # Title of the alloc check (for logging purpose)
    title = "Explore, Learn, and Compose"

    # Dependencies needed to execute pre_alloc and alloc
    dependencies = [CompositionalNetworks, ConstraintDomains]

    # Target of the alloc check
    targets = [CompositionalNetworks]

    # Code specific to the package being checked
    domains = fill(domain([1, 2, 3]), 3)

    # Code to trigger precompilation before the alloc check
    pre_alloc() = foreach(_ -> explore_learn_compose(domains, allunique), 1:10)

    # Code being allocations check
    alloc() = explore_learn_compose(domains, allunique)

    # Actual call to PerfChecker
    alloc_check(
        title,
        dependencies,
        targets,
        pre_alloc,
        alloc;
        path = @__DIR__,
        threads = 10,
    )
end
