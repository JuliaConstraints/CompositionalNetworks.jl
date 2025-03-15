const SimpleFilter = LayerCore(
    :SimpleFilter,
    true,
    (:(AbstractVector{<:Real}),) => AbstractVector{<:Real},
    (
        id=:((x) -> identity(x)),
        filter_equal_val=:((x; val) -> filter(t -> t == val, x)),
        filter_ge_val=:((x; val) -> filter(t -> t >= val, x)),
        filter_great_val=:((x; val) -> filter(t -> t > val, x)),
        filter_less_val=:((x; val) -> filter(t -> t < val, x)),
        filter_le_val=:((x; val) -> filter(t -> t <= val, x)),
        filter_ne_val=:((x; val) -> filter(t -> t != val, x)),
        filter_op_val=:((x; val, op_filter) -> filter(t -> op_filter(t, val), x)),
    )
)
