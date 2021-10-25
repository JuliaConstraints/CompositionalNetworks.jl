"""
    hamming(x, X)
Compute the hamming distance of `x` over a collection of solutions `X`, i.e. the minimal number of variables to switch in `x`to reach a solution.
"""
hamming(x, X) = mapreduce(y -> Distances.hamming(x,y), min, X)

"""
    minkowski(x, X, p)
"""
minkowski(x, X, p) = mapreduce(Distances.minkowski(x, y, p), min, X)

"""
    manhattan(x, X)
"""
manhattan(x, X) = mapreduce(y -> Distances.cityblock(x, y), min, X)

"""
    weigths_bias(x)
A metric that bias `x` towards operations with a lower bit. Do not affect the main metric.
"""
weigths_bias(x) = sum(p -> p[1] * log2(1. + p[2]), enumerate(x)) / length(x)^4
