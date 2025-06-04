const PairedMap = LayerCore(
    :PairedMap,
    true,
    (:(AbstractVector{<:Real}),) => AbstractVector{<:Real},
    (
        id = :((x) -> identity(x)),
        sub = :((x; pair_vars) -> abs.(x .- pair_vars)),
        sum = :((x; pair_vars) -> (x .+ pair_vars)),
        prod = :((x; pair_vars) -> (x .* pair_vars))
    )
)
