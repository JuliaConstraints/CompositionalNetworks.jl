# TODO: large test set

# Transformation layer
v1 = [1, 5, 2, 4, 3]
v2 = [1, 2, 3, 2, 1]

@test ICN._identity(v1) == v1
@test ICN._identity(v2) == v2

# @test _count_eq(v) == 
