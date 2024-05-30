using PerfChecker
using BenchmarkTools

using CompositionalNetworks
using ConstraintDomains

# Target of the becnhamrk
target = CompositionalNetworks

# Code specific to the package being checked
domains = fill(domain([1, 2, 3, 4]), 4)

# Code to trigger precompilation before the becnh (optional)
foreach(_ -> explore_learn_compose(domains, allunique), 1:10)

# Code being benchmarked
t = @benchmark explore_learn_compose(domains, allunique) evals = 1 samples = 10 seconds =
    3600

# Store the bench results
store_benchmark(t, target; path = @__DIR__)
