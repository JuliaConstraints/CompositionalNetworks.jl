# all_diff 
concept_all_diff(x) = allunique(x)

#dist_diff
function concept_dist_diff(x) 
    for i in 1:length(x)-3
        if abs(x[i] - x[i+1]) == abs(x[i+2] - x[i+3])
            return false
        end
    end
    return true
end

#equals
concept_all_eq(x) = all(y -> y == x[1], x[2:end])

#less_than_param
concept_less_than_param(x, param) = x[1] ≤ param


#ordered
concept_ordered(x) = issorted(x)

#no_overlap

#sum of vars = c

#concept_list = [concept_all_diff]
concept_list = [concept_all_diff, concept_dist_diff, concept_all_eq, concept_less_than_param, concept_ordered]

