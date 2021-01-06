"""
    hamming(x, X)
Compute the hamming distance of `x` over a collection of solutions `X`, i.e. the minimal number of variables to switch in `x`to reach a solution.
"""
hamming(x, X) = mapreduce(y -> sum(x .!= y), min, X; init = length(x))

# hamming(x, X::DataFrame) = _hamming(x, Array(X))
