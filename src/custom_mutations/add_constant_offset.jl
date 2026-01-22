# Custom Mutation: add_constant_offset
# =====================================
# This mutation selects a random subtree and wraps it with an addition
# of a random constant: `subtree` -> `subtree + c`
#
# This is different from built-in mutations:
# - mutate_constant: only perturbs EXISTING constants
# - add_node: adds operators at LEAF nodes only
# - insert_node: inserts operator but uses random leaves, not the subtree
#
# This mutation introduces a new constant offset to any part of the tree,
# which can help discover formulas with additive terms.

# Note: This file is `include`d into CustomMutationsModule, so it has access
# to: AbstractExpressionNode, NodeSampler, constructorof, set_node!, etc.

"""
    add_constant_offset(tree, options, nfeatures, rng)

Wrap a random subtree with addition of a random constant.
`subtree` becomes `subtree + c` where `c` is sampled from normal distribution.
"""
function add_constant_offset(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}
    # Find the + operator index (binary operators are indexed by their position)
    plus_idx = findfirst(op -> op == (+), options.operators.binops)

    if plus_idx === nothing
        # No + operator available, return tree unchanged
        return tree
    end

    # Sample a random node to wrap
    node = rand(rng, NodeSampler(; tree))

    # Create a random constant
    constant_value = randn(rng, T)  # Sample from normal distribution
    constant_node = constructorof(N)(T; val=constant_value)

    # Create new node: node + constant
    # Randomly decide if constant goes on left or right
    if rand(rng, Bool)
        new_node = constructorof(N)(; op=plus_idx, children=(copy(node), constant_node))
    else
        new_node = constructorof(N)(; op=plus_idx, children=(constant_node, copy(node)))
    end

    # Replace the selected node with the wrapped version
    set_node!(node, new_node)

    return tree
end
