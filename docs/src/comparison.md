# Comparison Layer

The comparison layer of our basic ICN can be constructed using `comparison_layer(param=false)`. All operations are mutually exclusive.

```@docs
CompositionalNetworks.comparison_layer
```

## Non-parametric comparisons

Follows a list of the current non-parametric operations available in any comparison layer.

```@docs
CompositionalNetworks._co_identity
CompositionalNetworks._co_euclidian
CompositionalNetworks._co_abs_diff_val_vars
CompositionalNetworks._co_val_minus_vars
CompositionalNetworks._co_vars_minus_val
```

## Parametric comparisons

And finally a list of the parametric ones.

```@docs
CompositionalNetworks._co_abs_diff_val_param
CompositionalNetworks._co_val_minus_param
CompositionalNetworks._co_param_minus_val
CompositionalNetworks._co_euclidian_param
```