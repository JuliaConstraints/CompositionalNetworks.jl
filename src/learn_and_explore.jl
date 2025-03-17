function generate_configurations(concept::Function, domains::Vector{<:SetDomain}; parameters...)::Configurations
    output = explore(domains, concept; parameters...)
    Set([Solution.(output[1])..., NonSolution.(output[2])...])
end

function explore_learn(
    domains::Vector{<:SetDomain},
    concept::Function,
    optimizer_config::T;
    icn=ICN(; parameters=[:dom_size, :num_variables]),
    configurations=nothing,
    metric_function=hamming,
    parameters...,
) where {T<:AbstractOptimizer}
    #=
    if :vals in icn.parameters && haskey(parameters, :vals)
        vals = parameters[:vals]
        param = Base.structdiff((; parameters...,), NamedTuple{(:vals,)})
        params = [(val=i, param...) for i in vals]

        new_icn = deepcopy(icn)
        delete!(new_icn.parameters.dict, :vals)
        push!(new_icn.parameters, :val)

        p = Pair{<:AbstractICN,Bool}[]
        Threads.@threads for i in 1:length(params)

            if isnothing(configurations)
                concept_new = ((x; parames...) -> concept(x; vals=(f = y -> (z = copy(y); z[i] = parames[:val]; z); f(vals)), param...))
                configurations = generate_configurations(concept_new, domains; params[i]...)
            end

            icn.constants[:dom_size] = maximum(length, domains)
            icn.constants[:numvars] = length(rand(configurations).x)

            deep_icn = deepcopy(new_icn)

            push!(p, optimize!(deep_icn, configurations, metric_function, optimizer_config; icn.constants..., params[i]...))
        end
        return p
    else
    =#
    if isnothing(configurations)
        configurations = generate_configurations(concept, domains; parameters...)
    end

    icn.constants[:dom_size] = maximum(length, domains)
    icn.constants[:numvars] = length(rand(configurations).x)

    return optimize!(icn, configurations, metric_function, optimizer_config; icn.constants..., parameters...)
    # end
end
