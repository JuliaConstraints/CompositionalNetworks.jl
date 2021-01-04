hamming(x, X) = mapreduce(y -> sum(x .!= y), min, X; init = length(x))

hamming(x, X::DataFrame) = _hamming(x, Array(X))
