function generate_configurations(x::Function, y::Vector{<:SetDomain})::Configurations
	output = ConstraintDomains.explore(y, x)
	Set([Solution.(output[1])..., NonSolution.(output[2])...])
end
