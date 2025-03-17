module LocalSearchSolversExt

import CompositionalNetworks: CompositionalNetworks, AbstractICN, Configurations
import CompositionalNetworks: LocalSearchOptimizer, apply!, weights_bias, regularization
import CompositionalNetworks: evaluate, solutions
import LocalSearchSolvers: model, domain, variable!, constraint!, objective!, solver, solve!
import LocalSearchSolvers: LocalSearchSolvers, has_solution, best_values

function CompositionalNetworks.LocalSearchOptimizer(; options::LocalSearchSolvers.Options=LocalSearchSolvers.Options())
    return LocalSearchOptimizer(options)
end

function mutually_exclusive(layer_weights, w)
    x = length(w)
    return iszero(x) ? 1.0 : max(0.0, x - l)
end

no_empty_layer(x; X=nothing) = max(0, 1 - sum(x))

parameter_specific_operations(x; X=nothing) = 0.0

function CompositionalNetworks.optimize!(
    icn::T,
    configurations::Configurations,
    metric_function::Function,
    optimizer_config::LocalSearchOptimizer; parameters...) where {T<:AbstractICN}

    @debug "starting debug opt"
    m = model(; kind=:icn)
    n = length(icn.weights)

    # All variables are boolean
    d = domain([false, true])

    # Add variables
    foreach(_ -> variable!(m, d), 1:n)

    # Add constraint
    start = 1
    for (i, layer) in enumerate(icn.layers)
        stop = start + icn.weightlen[i] - 1
        if layer.mutex
            f(x; X=nothing) = mutually_exclusive(icn.weightlen[i], x)
            constraint!(m, f, start:stop)
        else
            constraint!(m, no_empty_layer, start:stop)
        end
        start = stop + 1
    end

    function fitness(w)
        weights_validity = apply!(icn, w)
        return sum(
                   x -> abs(evaluate(icn, x; weights_validity=weights_validity, parameters...) - metric_function(x.x, solution_vector)), configurations
               ) + weights_bias(w) + regularization(icn)
    end

    objective!(m, fitness)

    # Create solver and solve
    s = solver(m; options=optimizer_config.options)
    solve!(s)
    @debug "pool" s.pool best_values(s.pool) best_values(s) s.pool.configurations

    # Return best values

    weights_validity = if has_solution(s)
        apply!(icn, BitVector(collect(best_values(s))))
    else
        CompositionalNetworks.generate_new_valid_weights!(icn)
        true
    end

    return icn => weights_validity
end

end
