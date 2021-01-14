# Transformation Layer

The transformation layer of our basic ICN can be constructed using `transformation_layer(param=false)`.


```@docs
CompositionalNetworks.transformation_layer
```

## Non-parametric transformations

Follows a list of the current non-parametric operations available in any transformation layer.

```@docs
CompositionalNetworks._tr_identity
CompositionalNetworks._tr_count_eq
CompositionalNetworks._tr_count_eq_left
CompositionalNetworks._tr_count_eq_right
CompositionalNetworks._tr_count_greater
CompositionalNetworks._tr_count_lesser
CompositionalNetworks._tr_count_g_left
CompositionalNetworks._tr_count_l_left
CompositionalNetworks._tr_count_g_right
CompositionalNetworks._tr_count_l_right
CompositionalNetworks._tr_contiguous_vals_minus
CompositionalNetworks._tr_contiguous_vals_minus_rev
```

Note that all functions are extended to a vectorized version with the `lazy` function.

```@docs
CompositionalNetworks.lazy
```

## Parametric transformations

And finally a list of the parametric ones.

```@docs
CompositionalNetworks._tr_count_eq_param
CompositionalNetworks._tr_count_l_param
CompositionalNetworks._tr_count_g_param
CompositionalNetworks._tr_count_bounding_param
CompositionalNetworks._tr_val_minus_param
CompositionalNetworks._tr_param_minus_val
```

Note that all functions are extended to a vectorized version with the `lazy_param` function.

```@docs
CompositionalNetworks.lazy_param
```