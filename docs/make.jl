using CompositionalNetworks
using ConstraintDomains
using Documenter

makedocs(;
    modules=[CompositionalNetworks, ConstraintDomains],
    authors="Jean-François Baffier",
    repo="https://github.com/JuliaConstraints/CompositionalNetworks.jl/blob/{commit}{path}#L{line}",
    sitename="CompositionalNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://JuliaConstraints.github.io/CompositionalNetworks.jl",
        assets = ["assets/favicon.ico"; "assets/github_buttons.js"; "assets/custom.css"],
    ),
    pages=[
        "Home" => "index.md",
        "ICNs" => "icn.md",
        "Layers" => [
            "Transformation" => "transformation.md",
            "Arithmetic" => "arithmetic.md",
            "Aggregation" => "aggregation.md",
            "Comparison" => "comparison.md",
        ],
        "Domains" => [
            "domain.md",
        ],
        "Library" => [
            "public.md",
            "internal.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/JuliaConstraints/CompositionalNetworks.jl.git",
    devbranch="main",
)

