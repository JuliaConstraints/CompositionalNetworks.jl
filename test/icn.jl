icn = ICN(nvars = 4, dom_size = 4, param = 2)

icn.weigths = vcat(trues(18), falses(6))

f = compose(icn)

v = [1,2,4,3]

@test f(v) == 67
