using InterpretableCompositionalNetworks
using Documenter

makedocs(;
    modules=[InterpretableCompositionalNetworks],
    authors="Jean-François Baffier",
    repo="https://github.com/JuliaConstraints/InterpretableCompositionalNetworks.jl/blob/{commit}{path}#L{line}",
    sitename="InterpretableCompositionalNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaConstraints.github.io/InterpretableCompositionalNetworks.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaConstraints/InterpretableCompositionalNetworks.jl",
)
