```@meta
CurrentModule = CompositionalNetworks
```

# CompositionalNetworks.jl

`CompositionalNetworks.jl`, a Julia package for Interpretable Compositional Networks (ICN), a variant of neural networks, allowing the user to get interpretable results, unlike regular artificial neural networks.

The current state of our ICN focuses on the composition of error functions for `LocalSearchSolvers.jl`, but produces results independently of it and export it to either/both Julia functions or/and human readable output.

### How does it work?

The package comes with a basic ICN for learning global constraints. The ICN is composed of 4 layers: `transformation`, `arithmetic`, `aggregation`, and `comparison`. Each contains several operations that can be composed in various ways.
Given a `concept` (a predicate over the variables' domains), a metric (`hamming` by default), and the variables' domains, we learn the binary weights of the ICN. 

## Installation

```julia
] add CompositionalNetworks
```

As the package is in a beta version, some changes in the syntax and features are likely to occur. However, those changes should be minimal between minor versions. Please update with caution.

## Quickstart

```julia
# 4 variables in 1:4
doms = [domain([1,2,3,4]) for i in 1:4]

# allunique concept (that is used to define the :all_different constraint)
err = explore_learn_compose(allunique, domains=doms)
# > interpretation: identity ∘ count_positive ∘ sum ∘ count_eq_left

# test our new error function
@assert err([1,2,3,3], dom_size = 4) > 0.0

# export an all_different function to file "current/path/test_dummy.jl" 
compose_to_file!(icn, "all_different", "test_dummy.jl")
```

The output file should produces a function that can be used as follows (assuming the maximum domain size is `7`)

```julia
import CompositionalNetworks

all_different([1,2,3,4,5,6,7]; dom_size = 7)
# > 0.0 (which means true, no errors)
```

Please see `JuliaConstraints/Constraints.jl/learn.jl` for an extensive example of ICN learning and compositions.

## Contributing

Contributions to this package are more than welcome and can be arbitrarily, and not exhaustively, split as follows:
- Adding (useful) operations in one of the $4$ existing layers
- Creating other ICNs from scratch or with only some of the original operations
- Creating an ICN with a layer structure
- Creating other compositional networks which target other problems
- Just making stuff better, faster, user-friendlier, etc.

### Contact
Do not hesitate to contact me (@azzaare) or other members of JuliaConstraints on GitHub (file an issue), the julialang discourse forum, the julialang slack channel, the julialang zulip server, or the Human of Julia (HoJ) discord server.