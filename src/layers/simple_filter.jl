const SimpleFilter = LayerCore(
    :SimpleFilter,
    true,
    (:(AbstractVector{<:Real}),) => AbstractVector{<:Real},
    (
        id=:((x) -> identity(x)),
        filter_unique=:((x) -> unique(x)),
        filter_elem=:((x; id) -> [x[id]]),
        filter_id=:((x; id) -> [x[id], 0 < x[id] <= length(x) ? -x[x[id]] : typemax(eltype(x))]),
        filter_equal_val=:((x; val) -> filter(t -> t == val, x)),
        filter_ge_val=:((x; val) -> filter(t -> t >= val, x)),
        filter_great_val=:((x; val) -> filter(t -> t > val, x)),
        filter_less_val=:((x; val) -> filter(t -> t < val, x)),
        filter_le_val=:((x; val) -> filter(t -> t <= val, x)),
        filter_ne_val=:((x; val) -> filter(t -> t != val, x)),
        filter_op_val=:((x; val, op_filter) -> filter(t -> op_filter(t, val), x)),
        filter_equal_vals=:((x; vals) -> filter(t -> t in vals, x)),
        filter_ne_vals=:((x; vals) -> filter(t -> !(t in vals), x)),
        filter_op_vals=:((x; vals, op_filter) -> filter(t -> op_filter(t, vals), x)),
    )
)
