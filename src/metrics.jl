"""
    hamming(x, X)
Compute the hamming distance of `x` over a collection of solutions `X`, i.e. the minimal number of variables to switch in `x`to reach a solution.
"""
hamming(x, X) = mapreduce(y -> sum(x .!= y), min, X; init = length(x))

"""
    minkowski(x, X, p)
"""
minkowski(x, X, p) = mapreduce(y -> sum(abs(x .- y)^p)^(1/p), min, X; init = sum(x.^p)^(1/p))

"""
    manhattan(x, X)
"""
manhattan(x, X) = minkowski(x, X, 1)

