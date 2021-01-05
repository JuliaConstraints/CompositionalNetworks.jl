# # Test with manually weighted ICN

# icn = ICN(nvars = 4, dom_size = 4, param = 2)
# show_layers(icn)
# icn.weigths = vcat(trues(18), falses(6))

# f = compose(icn)
# @test show_composition(icn) == "identity ∘ sum ∘ sum ∘ (param_minus_val + val_minus_param" *
#     " + count_bounding_param + count_g_param + count_l_param + count_eq_param +" *
#     " contiguous_vals_minus_rev + contiguous_vals_minus + count_l_right + count_g_right " * "+ count_l_left + count_g_left + count_lesser + count_greater + count_eq_right + " * "count_eq_left + count_eq + identity)"

# v = [1,2,4,3]
# @test f(v) == 67

## Test GA

X = csv2space("../data/csv/complete_ad-4-4.csv"; filter = :solutions)
@test hamming([1,2,3,3], X) == 1

icn = ICN(nvars = 4, dom_size = 4)

using Evolutionary
using Random

function _optimize2(icn = icn, X = X;ga=GA(), pop_size=50)
    pop = [bitrand(17) for i in 1:pop_size]
    @info typeof(pop)
    @info pop

    f = weigths -> _loss(X, icn, weigths, metric)

    Evolutionary.optimize(sum, pop, ga)
end

_ga = GA(
    populationSize=5,
    crossoverRate=0.8,
    epsilon = 0.05,
    selection = rouletteinv,
    crossover=singlepoint,
    mutation=flip,
)

_optimize2(;ga=_ga, pop_size=5)




CN._optimize(icn, X; pop_size = 5)

