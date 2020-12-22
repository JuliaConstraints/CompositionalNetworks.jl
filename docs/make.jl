using CompositionalNetworks
using Documenter

makedocs(;
    modules=[CompositionalNetworks],
    authors="Jean-François Baffier",
    repo="https://github.com/JuliaConstraints/CompositionalNetworks.jl/blob/{commit}{path}#L{line}",
    sitename="CompositionalNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaConstraints.github.io/CompositionalNetworks.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaConstraints/CompositionalNetworks.jl",
)
