module CustomLossModule

using DynamicExpressions:
    AbstractExpression, AbstractExpressionNode, get_tree, eval_tree_array
using ..CoreModule: AbstractOptions, Dataset, DATA_TYPE, LOSS_TYPE
using ..LossFunctionsModule: _eval_loss, evaluator, _ACTIVE_DYNAMIC_LOSS

export apply_custom_loss,
    load_loss_from_string!,
    load_loss_from_file!,
    clear_dynamic_losses!,
    list_available_losses,
    reload_custom_losses!

# Registry for dynamically loaded losses. The active loss lives in
# LossFunctionsModule._ACTIVE_DYNAMIC_LOSS (single source of truth) so the
# eval_loss dispatcher can read it without crossing module boundaries.
const DYNAMIC_LOSSES = Dict{Symbol,Function}()

"""
    default_loss(tree, dataset, options; regularization=true) -> L

Default loss: delegates to the built-in `_eval_loss` path (elementwise loss
from `options.elementwise_loss`, weighted if `is_weighted(dataset)`, with
optional dimensional regularization). Behavior-identical to having no custom
loss installed.
"""
function default_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions;
    regularization::Bool=true,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    return _eval_loss(tree, dataset, options, regularization)
end

"""
    apply_custom_loss(tree, dataset, options; regularization=true, idx=nothing) -> L

Dispatch to the active custom loss or the default. Reuses the same `evaluator`
helper as `options.loss_function`, so 3-arg `f(tree, dataset, options)` and
4-arg `f(tree, full_dataset, options, idx)` signatures both work.

Note: when a custom loss is active, `dimensional_regularization` is NOT
applied automatically; the custom loss must add any constraint penalties it
wants. This matches the existing `options.loss_function` semantics.
"""
function apply_custom_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Dataset{T,L},
    options::AbstractOptions;
    regularization::Bool=true,
    idx=nothing,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    func = _ACTIVE_DYNAMIC_LOSS[]
    if func === nothing
        return default_loss(tree, dataset, options; regularization=regularization)
    end
    inner = tree isa AbstractExpression ? get_tree(tree) : tree
    return evaluator(func, inner, dataset, options, idx)::L
end

"""
    load_loss_from_string!(name::Symbol, code::String) -> Function

Load a loss function from Julia code at runtime. Expected signature:
    function loss_name(tree, dataset, options) -> L

Returns the loaded function and sets it active.
"""
function load_loss_from_string!(name::Symbol, code::String)
    try
        expr = Meta.parse("begin\n$code\nend")
        Base.eval(@__MODULE__, expr)

        func = Base.eval(@__MODULE__, name)

        if !isa(func, Function)
            error("Code did not define a function named '$name'")
        end

        DYNAMIC_LOSSES[name] = func
        _ACTIVE_DYNAMIC_LOSS[] = func

        return func
    catch e
        @debug "Failed to load loss '$name' from code" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    load_loss_from_file!(name::Symbol, filepath::String) -> Function
"""
function load_loss_from_file!(name::Symbol, filepath::String)
    if !isfile(filepath)
        error("Loss file not found: $filepath")
    end
    code = read(filepath, String)
    return load_loss_from_string!(name, code)
end

"""
    clear_dynamic_losses!()

Clear all dynamically loaded losses, restoring default MSE behavior.
"""
function clear_dynamic_losses!()
    empty!(DYNAMIC_LOSSES)
    _ACTIVE_DYNAMIC_LOSS[] = nothing
    return nothing
end

"""
    list_available_losses() -> Vector{Symbol}
"""
function list_available_losses()
    return collect(keys(DYNAMIC_LOSSES))
end

"""
    reload_custom_losses!()
"""
function reload_custom_losses!()
    if !isempty(DYNAMIC_LOSSES)
        _ACTIVE_DYNAMIC_LOSS[] = last(values(DYNAMIC_LOSSES))
    else
        _ACTIVE_DYNAMIC_LOSS[] = nothing
    end
    return nothing
end

end # module
