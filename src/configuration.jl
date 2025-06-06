abstract type AbstractSolution end

struct Solution <: AbstractSolution
    x::Any
end

struct NonSolution <: AbstractSolution
    x::Any
end

struct UnknownSolution <: AbstractSolution
    x::Any
end

const Configuration{T} = T where {T <: AbstractSolution} # alias

const Configurations{N} = Set{<:Configuration}

function solutions(x::Configurations; non_solutions = false)
    Iterators.filter(r -> isa(r, ifelse(non_solutions, NonSolution, Solution)), x)
end
