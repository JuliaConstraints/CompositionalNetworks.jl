module LearnedScoringCases
import CompositionalNetworks as CN
using TOML, SHA

const BANK_SHA = "4df0b409d303d1b666a2d0aba3367c1dc08a2b380da109e50dee83b042b8ea90"

# Bank recovery and composition happen in the factory, outside scoring.
# The prepared state owns its input and retains only the compiled function.
function recovered_decoder(bank, witness)
    bytes = read(bank)
    bytes2hex(sha256(bytes)) == BANK_SHA || error("unexpected learned bank")
    saved = TOML.parse(String(bytes))["witnesses"][witness]
    expected = witness == 2 ? ("all_equal", "numeric list") :
               witness == 52 ? ("ordered", "order <, without offsets") :
               error("this workload qualifies witnesses 2 and 52")
    (saved["family"], saved["variant"]) == expected || error("witness identity changed")
    network = CN.learnable_composition((;); max_depth = 1)
    weights = BitVector(saved["weights"])
    CN.check_weights_validity(network, weights) || error("invalid learned weights")
    CN.apply!(network, weights) || error("could not recover learned weights")
    return first(CN.compose(network))
end

function full_square(::Val{operations}, x) where {operations}
    total = 0
    @inbounds for i in eachindex(x), j in eachindex(x)
        total += CN._pairwise_counts(operations, x, i, j)
    end
    return total
end

score_counts(state) = state.decoder(state.input)
score_square(state) = full_square(state.operations, state.input)
score_triangle(state) = invoke(CN._aggregate_pairwise_sum,
    Tuple{typeof(state.operations),Any}, state.operations, state.input)

# Keep the oracle accumulator local instead of boxing it in the verification
# closure. The full-square traversal independently applies each atomic predicate.
function expected_counts(x, witness)
    total = 0
    for i in eachindex(x), j in eachindex(x)
        j < i || continue
        total += witness == 2 ? ((x[j] < x[i]) + (x[j] > x[i])) :
                 ((x[j] == x[i]) + (x[j] > x[i]))
    end
    return total
end

function pair_counts(parameters)
    witness = Int(get(parameters, "witness", 2))
    n = Int(get(parameters, "n", 1000))
    bank = get(parameters, "bank", get(ENV, "JULIACONSTRAINTS_ICN_BANK", ""))
    isempty(bank) && error("set JULIACONSTRAINTS_ICN_BANK to the qualified exact bank")
    decoder = recovered_decoder(bank, witness)
    operations = witness == 2 ? Val((:count_less_left, :count_great_left)) :
                 Val((:count_equal_left, :count_great_left))
    input_type = get(parameters, "element", "int") == "float64" ? Float64 : Int
    original = input_type.(mod.(collect(1:n), 23))
    expected = expected_counts(original, witness)
    method = get(parameters, "method", "compiled")
    operation = method == "square" ? score_square :
                method == "original-triangle" ? score_triangle : score_counts
    return (
        prepare = () -> (; decoder, operations, input = copy(original)),
        operation,
        verify = (state, result) -> result == expected && state.input == original,
    )
end
end
