abstract type AbstractSolution end

struct Solution <: AbstractSolution
	x
end

struct NonSolution <: AbstractSolution
	x
end

const Configuration{T} = T where T <: AbstractSolution # alias

const Configurations{N} = Set{<:Configuration}
