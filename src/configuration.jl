export AbstractSolution, Solution, NonSolution, Configuration, Configurations, solutions

abstract type AbstractSolution end

struct Solution <: AbstractSolution
	x
end

struct NonSolution <: AbstractSolution
	x
end

const Configuration{T} = T where T <: AbstractSolution # alias

const Configurations{N} = Set{<:Configuration}

solutions(x::Configurations; non_solutions=false) = Iterators.filter(r -> isa(r, ifelse(non_solutions, NonSolution, Solution)), x)
