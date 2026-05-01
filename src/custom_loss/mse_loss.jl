# Custom Loss: mse_loss
# =====================
# Default loss: per-sample mean squared error. Behavior matches the built-in
# `_eval_loss` path on unweighted, unitless datasets, exposed as a named
# operator so the meta-evolution loop has a concrete parent to refine.
#
# For weighted datasets or unit-bearing data, the in-module `default_loss`
# (used when no operator is loaded) handles `dataset.weights` and
# `dimensional_regularization` via `_eval_loss`. This baseline assumes
# unweighted, unitless data, which matches every SRBench dataset on the
# PySR pipeline.

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
