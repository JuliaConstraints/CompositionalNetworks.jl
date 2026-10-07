module JuMPExactExt

import CompositionalNetworks
import CompositionalNetworks: AbstractICN, Configurations, JuMPExactOptimizer
import CompositionalNetworks: apply!, regularization, solutions, training_metric_error
import CompositionalNetworks: training_metric_valid, weights_bias
import JuMP

const MOI = JuMP.MOI

function _tie_break_coefficients(icn::AbstractICN)
    count = length(icn.weights)
    count > 0 || throw(ArgumentError("an ICN must contain at least one weight"))
    coefficients = [index / Float64(count)^4 for index in 1:count]
    non_mutex_count = sum(
        icn.weightlen[index] for index in eachindex(icn.layers) if !icn.layers[index].mutex
    )
    if non_mutex_count > 0
        offset = 1
        coefficient = inv(non_mutex_count + 1)
        for index in eachindex(icn.layers)
            range = offset:(offset + icn.weightlen[index] - 1)
            if !icn.layers[index].mutex
                @inbounds for weight_index in range
                    coefficients[weight_index] += coefficient
                end
            end
            offset = last(range) + 1
        end
    end
    return coefficients
end

function _has_candidate(model)
    return JuMP.result_count(model) > 0 && JuMP.primal_status(model) in (
        MOI.FEASIBLE_POINT,
        MOI.NEARLY_FEASIBLE_POINT
    )
end

function _remaining_seconds(deadline)
    return isfinite(deadline) ? max(0.0, deadline - time()) : Inf
end

function _record_telemetry!(
        optimizer_config,
        model;
        candidates_evaluated,
        master_solves,
        exact,
        termination_reason,
        last_status,
        best_training_error,
        best_objective,
        lower_bound,
        elapsed_seconds
)
    telemetry = optimizer_config.telemetry
    empty!(telemetry)
    telemetry[:backend] = JuMP.solver_name(model)
    telemetry[:candidates_evaluated] = candidates_evaluated
    telemetry[:master_solves] = master_solves
    telemetry[:exact] = exact
    telemetry[:termination_reason] = termination_reason
    telemetry[:optimizer_termination_status] = Symbol(string(last_status))
    telemetry[:time_limit] = optimizer_config.time_limit
    telemetry[:time_limit_reached] = termination_reason == :time_limit
    telemetry[:max_candidates] = optimizer_config.max_candidates
    telemetry[:enumerate_training_exact] = optimizer_config.enumerate_training_exact
    telemetry[:candidate_limit_reached] = termination_reason == :candidate_limit
    telemetry[:best_training_error] = best_training_error
    telemetry[:best_objective] = best_objective
    telemetry[:lower_bound] = lower_bound
    telemetry[:elapsed_seconds] = elapsed_seconds
    return telemetry
end

function CompositionalNetworks.optimize!(
        icn::T,
        configurations::Configurations,
        metric_function::Union{Function, Vector{Function}},
        optimizer_config::JuMPExactOptimizer;
        candidate_observer = nothing,
        parameters...
) where {T <: AbstractICN}
    started_at = time()
    deadline = isfinite(optimizer_config.time_limit) ?
               started_at + optimizer_config.time_limit : Inf
    solution_vector = [configuration.x for configuration in solutions(configurations)]
    isempty(solution_vector) && throw(ArgumentError(
        "exact ICN learning requires at least one solution in the training configurations",
    ))

    model = JuMP.Model(optimizer_config.optimizer_factory)
    for attribute in optimizer_config.attributes
        JuMP.set_optimizer_attribute(model, first(attribute), last(attribute))
    end
    optimizer_config.silent && JuMP.set_silent(model)

    count = length(icn.weights)
    weights = JuMP.@variable(model, [1:count], Bin)
    offset = 1
    for (index, layer) in enumerate(icn.layers)
        range = offset:(offset + icn.weightlen[index] - 1)
        if layer.mutex
            JuMP.@constraint(model, sum(weights[range]) == 1)
        else
            JuMP.@constraint(model, sum(weights[range]) >= 1)
        end
        offset = last(range) + 1
    end

    coefficients = _tie_break_coefficients(icn)
    JuMP.@objective(model,
        Min,
        sum(coefficients[index] * weights[index] for index in eachindex(weights)),)

    working_icn = deepcopy(icn)
    best_weights = nothing
    best_training_error = Inf
    best_objective = Inf
    candidates_evaluated = 0
    master_solves = 0
    lower_bound = 0.0
    exact = false
    termination_reason = :unknown
    last_status = MOI.OPTIMIZE_NOT_CALLED
    tolerance = 64eps(Float64)

    while candidates_evaluated < optimizer_config.max_candidates
        remaining = _remaining_seconds(deadline)
        if iszero(remaining)
            termination_reason = :time_limit
            break
        elseif isfinite(remaining)
            JuMP.set_time_limit_sec(model, remaining)
        end

        JuMP.optimize!(model)
        master_solves += 1
        last_status = JuMP.termination_status(model)

        if last_status == MOI.INFEASIBLE
            exact = !isnothing(best_weights)
            termination_reason = :exhausted
            break
        elseif !_has_candidate(model)
            termination_reason = last_status == MOI.TIME_LIMIT ? :time_limit :
                                 :backend_stopped
            break
        end

        candidate = BitVector(JuMP.value(weight) >= 0.5 for weight in weights)
        candidate_bound = sum(
            coefficients[index] * candidate[index] for index in eachindex(candidate)
        )
        if last_status == MOI.OPTIMAL
            lower_bound = candidate_bound
            if !optimizer_config.enumerate_training_exact &&
               best_objective <= lower_bound + tolerance
                exact = true
                termination_reason = :lower_bound
                break
            end
        end

        structural_valid = apply!(working_icn, candidate)
        if !structural_valid
            candidates_evaluated += 1
            isnothing(candidate_observer) || candidate_observer((;
                weights = copy(candidate),
                structural_valid = false,
                training_error = Inf,
                objective = Inf,
            ))
            JuMP.@constraint(model,
                sum(candidate[index] ? 1 - weights[index] : weights[index]
                    for index in eachindex(weights)) >= 1,)
            continue
        end
        training_error = Float64(training_metric_error(
            working_icn,
            configurations,
            solution_vector,
            metric_function;
            weights_validity = structural_valid,
            parameters...
        ))
        isfinite(training_error) || throw(ArgumentError(
            "the ICN training metric must return a finite value for valid weights",
        ))
        training_error >= -tolerance || throw(ArgumentError(
            "JuMPExactOptimizer requires a non-negative ICN training metric",
        ))
        training_error = max(0.0, training_error)
        objective = training_error + weights_bias(candidate) + regularization(working_icn)
        candidates_evaluated += 1

        if !isnothing(candidate_observer)
            candidate_observer((;
                weights = copy(candidate),
                structural_valid,
                training_error,
                objective
            ))
        end
        if objective < best_objective
            best_weights = copy(candidate)
            best_training_error = training_error
            best_objective = objective
        end

        if iszero(training_error) && last_status == MOI.OPTIMAL &&
           !optimizer_config.enumerate_training_exact
            exact = true
            lower_bound = candidate_bound
            termination_reason = :zero_error
            break
        elseif last_status != MOI.OPTIMAL
            termination_reason = last_status == MOI.TIME_LIMIT ? :time_limit :
                                 :backend_stopped
            break
        end

        JuMP.@constraint(model,
            sum(candidate[index] ? 1 - weights[index] : weights[index]
        for index in eachindex(weights)) >= 1,)
    end

    if termination_reason == :unknown
        termination_reason = :candidate_limit
    end
    isnothing(best_weights) && error(
        "the exact ICN optimizer did not obtain a structurally valid candidate " *
        "(termination: $(termination_reason), backend: $(last_status))",
    )

    structural_valid = apply!(icn, best_weights)
    validity = training_metric_valid(
        icn,
        configurations,
        solution_vector,
        metric_function;
        weights_validity = structural_valid,
        parameters...
    )
    _record_telemetry!(
        optimizer_config,
        model;
        candidates_evaluated,
        master_solves,
        exact,
        termination_reason,
        last_status,
        best_training_error,
        best_objective,
        lower_bound,
        elapsed_seconds = time() - started_at
    )
    return icn => validity
end

end
