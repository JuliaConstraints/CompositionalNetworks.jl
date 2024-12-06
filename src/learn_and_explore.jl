export generate_configurations, explore_learn

function generate_configurations(concept::Function, domains::Vector{<:SetDomain}; parameters...)::Configurations
	output = explore(domains, concept; parameters...)
	Set([Solution.(output[1])..., NonSolution.(output[2])...])
end

function explore_learn(
    domains::Vector{<:SetDomain},
    concept::Function,
    optimizer_config::T;
    icn = ICN(;parameters=[:dom_size, :num_variables]),
    configurations = nothing,
    metric_function = hamming,
    parameters...,
) where T <: AbstractOptimizer
    if isnothing(configurations)
        configurations = generate_configurations(concept, domains; parameters...)
    end

    dom_size = maximum(length, domains)
    num_variables = length(configurations[1].x)
    return optimize!(
        icn,
        configurations,
        metric_function,
        optimizer_config;
        num_variables = num_variables,
        dom_size = dom_size,
        parameters...,
    )
end
