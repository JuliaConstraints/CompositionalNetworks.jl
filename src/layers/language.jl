"""
Common ICN layer for every value supplied through the `language` keyword. Concrete
language representations specialize the operations in `ConstraintCommons`; they do not
change the ICN structure or its operation names.
"""
const Language = LayerCore(
    :Language,
    true,
    (:(AbstractVector),) => AbstractVector{<:Real},
    (
        accept = :(
            (x; language) -> [Float64(!ConstraintCommons.accept(language, x))]
        ),
        reject = :(
            (x; language) -> [Float64(ConstraintCommons.accept(language, x))]
        ),
        distance = :(
            (x; language) ->
                [Float64(ConstraintCommons.language_distance(language, x))]
        ),
    ),
)

@testitem "Language layer is representation-independent" begin
    using ConstraintCommons
    using Test

    automaton = Automaton(
        Dict((:start, 0) => :finish, (:start, 1) => :start),
        :start,
        :finish,
    )
    diagram = MDD([Dict((:root, 0) => :finish)])
    automaton_layers = layers_for_parameters((; language = automaton))
    diagram_layers = layers_for_parameters((; language = diagram))

    @test automaton_layers == diagram_layers
    @test first(automaton_layers) === Language
    @test Set(keys(Language.fn)) == Set((:accept, :reject, :distance))
end
