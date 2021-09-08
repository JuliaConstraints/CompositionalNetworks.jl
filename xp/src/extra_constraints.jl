# This files stores constraints or concepts that are not yet available within usual_constraints

# Sample for all_different
# const all_different = Constraint(
#     concept = concept_all_different,
#     error = make_error(:all_different),
#     syms = Set([:permutable]),
# )

## no overlap
function concept_no_overlap(x, l)
    for i in 1:length(x), j in i+1:length(x)-1
        x[i]+l[i] > x[j] && x[j]+l[j] > x[i] && return false
    end
    return true
end

const no_overlap = Constraint(
    concept = concept_no_overlap,
    error = make_error(:no_overlap),
)

push!(BENCHED_CONSTRAINTS, :no_overlap => no_overlap)
