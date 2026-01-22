# PySR/SymbolicRegression.jl Mutation Reference

This document describes all built-in mutation operators in SymbolicRegression.jl.
Use this as a reference when creating new custom mutations.

## Expression Tree Structure

Expressions are represented as trees where:
- **Leaf nodes** are either:
  - Constants: `node.constant == true`, value in `node.val`
  - Variables: `node.constant == false`, feature index in `node.feature`
- **Internal nodes** have:
  - `node.degree`: 1 for unary ops, 2 for binary ops
  - `node.op`: index into the operator list
  - Children accessed via `get_child(node, i)` or `node.l`, `node.r` for binary

## Key Imports for Custom Mutations

```julia
using Random: AbstractRNG, default_rng
using DynamicExpressions:
    AbstractExpressionNode,
    AbstractExpression,
    NodeSampler,
    constructorof,
    set_node!,
    get_contents,
    with_contents,
    get_child,
    set_child!,
    count_nodes,
    has_constants,
    has_operators
```

## Built-in Mutations

### 1. mutate_constant
**Weight:** 0.0353 (low - constants optimized separately)
**Purpose:** Randomly perturb an existing constant's value

```julia
function mutate_constant(tree, temperature, options, rng)
    # Find a random constant node
    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && t.constant)))
    # Perturb by a temperature-dependent factor
    node.val = node.val * mutate_factor(typeof(node.val), temperature, options, rng)
    return tree
end
```

### 2. mutate_operator
**Weight:** 3.63 (high - commonly used)
**Purpose:** Change one operator to another of the same arity

```julia
function mutate_operator(tree, options, rng)
    # Find a random operator node
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree != 0))
    # Replace with random operator of same arity
    node.op = rand(rng, 1:(options.nops[node.degree]))
    return tree
end
```

### 3. mutate_feature
**Weight:** 0.1
**Purpose:** Change which input variable a leaf references

```julia
function mutate_feature(tree, nfeatures, rng)
    # Find a random variable node (not constant)
    node = rand(rng, NodeSampler(; tree, filter=t -> (t.degree == 0 && !t.constant)))
    # Change to different feature
    node.feature = rand(rng, filter(!=(node.feature), 1:nfeatures))
    return tree
end
```

### 4. swap_operands
**Weight:** 0.00608
**Purpose:** Swap left and right children of a binary operator

```julia
function swap_operands(tree, rng)
    # Find a binary operator
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 1))
    # Swap children
    n1, n2 = get_child(node, 1), get_child(node, 2)
    set_child!(node, n2, 1)
    set_child!(node, n1, 2)
    return tree
end
```

### 5. rotate_tree
**Weight:** 1.42
**Purpose:** Tree rotation - restructures tree without changing semantics (for associative ops)

```julia
function randomly_rotate_tree!(tree, rng)
    # Find a node where rotation is valid (has operator child)
    # Perform rotation: swap parent-child relationship
    # This can help escape local optima by changing tree structure
    return tree
end
```

### 6. add_node (append/prepend)
**Weight:** 0.0771
**Purpose:** Add a new operator at a leaf position

```julia
function append_random_op(tree, options, nfeatures, rng)
    # Find a random leaf
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree == 0))
    # Replace leaf with: op(random_leaves...)
    # One child may be the original leaf's value
    newnode = constructorof(typeof(tree))(;
        op=rand(rng, 1:options.nops[arity]),
        children=tuple_of_random_leaves
    )
    set_node!(node, newnode)
    return tree
end
```

### 7. insert_node
**Weight:** 2.44 (high)
**Purpose:** Insert an operator at a random position, carrying subtree as one child

```julia
function insert_random_op(tree, options, nfeatures, rng)
    # Pick any node
    node = rand(rng, NodeSampler(; tree))
    # Create new operator with this node as one child, random leaves as others
    newnode = constructorof(typeof(tree))(;
        op=rand(rng, 1:options.nops[arity]),
        children=(copy(node), random_leaf, ...)  # node carried forward
    )
    set_node!(node, newnode)
    return tree
end
```

### 8. delete_node
**Weight:** 0.369
**Purpose:** Remove an operator, replacing it with one of its children

```julia
function delete_random_op!(tree, rng)
    # Find an operator node
    node = rand(rng, NodeSampler(; tree, filter=t -> t.degree > 0))
    # Pick one child to keep
    carry = get_child(node, rand(rng, 1:node.degree))
    # Replace node with the child
    # (handled specially if node is root)
    return tree_or_carry
end
```

### 9. simplify
**Weight:** 0.00148
**Purpose:** Apply algebraic simplification rules (e.g., x*1 -> x)

```julia
# Uses DynamicExpressions.simplify_tree! and combine_operators
# This is a "return immediately" mutation - doesn't need re-evaluation
```

### 10. randomize
**Weight:** 0.00695
**Purpose:** Replace entire tree with a new random tree

```julia
function randomize_tree(tree, curmaxsize, options, nfeatures, rng)
    tree_size = rand(rng, 1:curmaxsize)
    return gen_random_tree_fixed_size(tree_size, options, nfeatures, T, rng)
end
```

### 11. optimize
**Weight:** 0.0 (disabled by default, expensive)
**Purpose:** Run constant optimization (BFGS) on tree constants

```julia
# Uses ConstantOptimizationModule.optimize_constants
# This is a "return immediately" mutation with num_evals cost
```

### 12. do_nothing
**Weight:** 0.431 (high)
**Purpose:** Keep tree unchanged - helps with selection pressure

```julia
# Simply returns the tree as-is
# Useful for not always forcing changes
```

### 13/14. form_connection / break_connection
**Weight:** 0.5 / 0.1 (GraphNode only)
**Purpose:** For graph-based expressions, create/break shared subexpressions

---

## Template for New Custom Mutations

Save as `your_mutation_name.jl` in the `custom_mutations/` directory:

```julia
# Custom Mutation: your_mutation_name
# Description: What this mutation does

"""
    your_mutation_name(tree, options, nfeatures, rng)

Detailed description of the mutation.
"""
function your_mutation_name(
    tree::N,
    options,
    nfeatures::Int,
    rng::AbstractRNG,
) where {T,N<:AbstractExpressionNode{T}}

    # Your mutation logic here
    # Common patterns:
    #   - rand(rng, NodeSampler(; tree, filter=...)) to select nodes
    #   - set_node!(node, newnode) to replace a node
    #   - constructorof(N)(...) to create new nodes
    #   - get_child(node, i) / set_child!(node, child, i) for tree manipulation

    return tree  # Always return the (possibly modified) tree
end
```

Then enable in `config.toml`:
```toml
[custom_mutations]
your_mutation_name = 0.5  # weight
```

---

## Ideas for New Mutations

1. **Constant folding**: Evaluate subtrees that are purely constants
2. **Symmetry exploitation**: Detect symmetric patterns and simplify
3. **Gradient-guided**: Use gradient info to guide constant changes
4. **Subtree caching**: Reuse successful subtrees from hall of fame
5. **Dimensional hinting**: Use unit analysis to guide mutations
6. **Pattern templates**: Insert common mathematical patterns (e.g., `a*x + b`)
