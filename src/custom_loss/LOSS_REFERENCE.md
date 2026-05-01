# PySR/SymbolicRegression.jl Custom Loss Reference

## Function Signature

```julia
function your_loss_name(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    # loss logic - return a non-negative scalar (Inf on eval failure)
    return loss_value
end
```

The loss function is called **once per individual** during fitness evaluation. It receives the expression tree, the training dataset, and the search `options`, and must return a scalar of type `L` (typically `Float64`).

A 4-arg batched form is also supported:

```julia
function your_loss_name(
    tree, full_dataset::Dataset{T,L}, options::AbstractOptions, idx,
)::L where {T,L}
    # idx is a Vector{Int} of training-row indices for this batch (or nothing)
    return loss_value
end
```

If `idx` is provided, you can index `full_dataset.X[:, idx]` and `full_dataset.y[idx]` for batched evaluation. Most users do not need this.

---

## Available API

```julia
using DynamicExpressions: AbstractExpression, AbstractExpressionNode,
    eval_tree_array, get_tree
using ..CoreModule: AbstractOptions, Dataset, DATA_TYPE, LOSS_TYPE
```

### Dataset

```julia
dataset.X        # AbstractMatrix{T} - features (n_features × n_samples)
dataset.y        # AbstractVector{T} - targets (length n_samples)
dataset.weights  # AbstractVector{T} or Nothing - per-sample weights (may be nothing)
dataset.n        # Int - number of samples
```

These work transparently on both `BasicDataset` and `SubDataset` (a lazy view used during batching).

### Tree evaluation

```julia
prediction, completed = eval_tree_array(tree, dataset.X, options)
# prediction :: Vector{T} or nothing
# completed  :: Bool - false if numeric overflow, divide-by-zero, etc.
```

**Always** check `completed` and `isnothing(prediction)` and return `L(Inf)` on failure.

### Options access (selected)

```julia
options.maxsize          # Int
options.elementwise_loss # Function or SupervisedLoss
options.parsimony        # Float - the per-size cost penalty (handled separately!)
```

---

## Default Implementation

```julia
function mse_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return L(Inf)
    end
    diff = prediction .- dataset.y
    return L(sum(abs2, diff) / length(diff))
end
```

---

## Critical Constraints

> **DO NOT penalize raw expression complexity.** Parsimony (`size * options.parsimony`) is added separately by `loss_to_cost` after your loss returns. The Pareto frontier is built using `member.loss` (your raw return), so penalizing size in the loss both double-counts complexity and distorts the size/accuracy tradeoff curve.

> **Default behavior should remain MSE-like.** A custom loss should be MSE (or another fidelity term) **plus** an augmentation, not a replacement. Wildly different loss scales break the Pareto-frontier comparison and the score-function logarithmic scaling.

> **Return `LOSS_TYPE` (typically `Float64`).** Use `L(value)` to convert.

> **Handle eval failure.** When `eval_tree_array` returns `(nothing, false)`, return `L(Inf)`.

> **No automatic `dimensional_regularization`.** When a custom loss is active, the dimensional-units regularization is NOT auto-added. If you use units (`X_units` / `y_units`), call `dimensional_regularization(tree, dataset, options)` yourself and add it to your return value.

---

## Augmentation Ideas

These are **examples**, not prescriptions. The whole point of evolving the loss is to discover augmentations that work better than MSE on average across our task suite.

### Entropy / diversity bonus

Discourage low-information expressions like `((x0 - x0) + (x0 + (x0 - (x0 + (x0 - x0))))) * (...)`. Reward expressions that touch more distinct variables, use a variety of operators, or reduce to a non-constant function. For example:

```julia
function mse_with_diversity_bonus(tree, dataset, options)
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed || isnothing(prediction)
        return Float64(Inf)
    end
    mse = sum(abs2, prediction .- dataset.y) / length(dataset.y)

    # Cheap diversity proxy: variance of predictions. A constant tree has
    # variance 0; a tree that actually uses inputs has positive variance.
    pred_var = sum(abs2, prediction .- (sum(prediction) / length(prediction))) / length(prediction)
    diversity_bonus = pred_var < 1e-10 ? 1.0 : 0.0  # Inf-ish penalty if collapsed to a constant

    return mse + 0.1 * diversity_bonus
end
```

### Robust losses

Huber / log-cosh / quantile loss for noisy targets. Useful when y has outliers.

### Scale-aware losses

Log-MSE or relative-MSE for problems with wide y-ranges where MSE is dominated by large-y points.

### Train/holdout disagreement (advanced)

Use the 4-arg batched form to penalize predictions whose holdout error diverges from training error — a soft anti-overfitting term.

---

## Multiprocessing Caveat

The active loss function is held in a module-global ref (`_ACTIVE_DYNAMIC_LOSS`). In Julia threading mode (the PySR default), all threads share this ref and a single `load_loss_from_string!` is enough. In `:multiprocessing` mode, each worker process has its own copy of the module — you must arrange to load the loss on each worker (e.g. via `@everywhere`) before search starts. The same caveat applies to the existing `CustomMutationsModule` / `CustomSurvivalModule` / `CustomSelectionModule`.
