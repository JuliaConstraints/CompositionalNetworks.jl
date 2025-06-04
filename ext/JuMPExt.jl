module JuMPExt

using JuMP
using Juniper
using Ipopt
using Gurobi

# Original imports
import CompositionalNetworks: CompositionalNetworks, AbstractICN, Configurations
import CompositionalNetworks: JuMPOptimizer, apply!, weights_bias, regularization
import CompositionalNetworks: evaluate, solutions

function CompositionalNetworks.optimize!(
        icn::T,
        configurations::Configurations,
        metric_function::Union{Function, Vector{Function}},
        optimizer_config::JuMPOptimizer;
        parameters...
) where {T <: AbstractICN}
    # Create model
    m = Model()

    # Set up MINLP solver
    nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    mip_solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

    set_optimizer(
        m,
        optimizer_with_attributes(
            Juniper.Optimizer,
            "nl_solver" => nl_solver,
            "mip_solver" => mip_solver,
            "log_levels" => []
        )
    )

    n = length(icn.weights)

    # All variables are binary
    @variable(m, w[1:n], Bin)

    # Add constraints
    start = 1
    for (i, layer) in enumerate(icn.layers)
        stop = start + icn.weightlen[i] - 1
        idx_range = start:stop

        if layer.mutex
            # Mutually exclusive constraint - at most one variable can be true
            # Equivalent to: max(0.0, sum(w[idx_range]) - 1) = 0
            @constraint(m, sum(w[j] for j in idx_range) <= 1)
        else
            # No empty layer constraint - at least one variable must be true
            # Equivalent to: max(0, 1 - sum(w[idx_range])) = 0
            @constraint(m, sum(w[j] for j in idx_range) >= 1)
        end

        start = stop + 1
    end

    # Define fitness function - keeping the original structure
    function fitness(w_values)
        # Convert JuMP variables to BitVector
        w_bits = BitVector([value(w_values[i]) > 0.5 for i in 1:length(w_values)])

        weights_validity = apply!(icn, w_bits)

        s = if metric_function isa Function
            metric_function(
                icn,
                configurations,
                solution_vector;
                weights_validity = weights_validity,
                parameters...
            )
        else
            minimum(
                met -> met(
                    icn,
                    configurations,
                    solution_vector;
                    weights_validity = weights_validity,
                    parameters...
                ),
                metric_function
            )
        end
        return s + weights_bias(w_bits) + regularization(icn)
    end

    # Define objective using the fitness function
    @NLobjective(m, Min, fitness(w))

    # Solve model
    optimize!(m)

    # Return solution
    if termination_status(m) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        w_sol = value.(w) .> 0.5  # Convert to BitVector
        weights_validity = apply!(icn, BitVector(w_sol))
        return icn => weights_validity
    else
        # No solution found, generate new valid weights
        CompositionalNetworks.generate_new_valid_weights!(icn)
        return icn => true
    end
end

end
