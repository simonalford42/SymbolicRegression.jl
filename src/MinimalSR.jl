module MinimalSR

using PythonCall
using Random
using Optim

_mean(v) = isempty(v) ? 0.0 : sum(v) / length(v)

# ─── Node hierarchy ─────────────────────────────────────────────────────────

abstract type Node end

mutable struct ConstNode <: Node
    value::Float64
end

mutable struct VarNode <: Node
    feature::Int  # 1-based feature index into X's columns
end

mutable struct OpNode <: Node
    op::Symbol
    left::Node
    right::Union{Node, Nothing}  # nothing ⇒ unary
end

isleaf(::ConstNode) = true
isleaf(::VarNode) = true
isleaf(::OpNode) = false

Base.copy(n::ConstNode) = ConstNode(n.value)
Base.copy(n::VarNode) = VarNode(n.feature)
Base.copy(n::OpNode) = OpNode(n.op, copy(n.left), isnothing(n.right) ? nothing : copy(n.right))

tree_size(::ConstNode) = 1
tree_size(::VarNode) = 1
tree_size(n::OpNode) = 1 + tree_size(n.left) + (isnothing(n.right) ? 0 : tree_size(n.right))

tree_height(::ConstNode) = 1
tree_height(::VarNode) = 1
tree_height(n::OpNode) = 1 + max(tree_height(n.left), isnothing(n.right) ? 0 : tree_height(n.right))

node_string(n::ConstNode) = string(n.value)
node_string(n::VarNode) = "x$(n.feature - 1)"
function node_string(n::OpNode)
    isnothing(n.right) && return string(n.op, "(", node_string(n.left), ")")
    return string("(", node_string(n.left), " ", n.op, " ", node_string(n.right), ")")
end

# ─── Individual ─────────────────────────────────────────────────────────────

mutable struct Individual
    tree::Node
    loss::Float64
    cost::Float64
    complexity::Int
    birth::Int
    ref::Int
    parent_ref::Union{Int, Nothing}
end

Base.copy(m::Individual) = Individual(copy(m.tree), m.loss, m.cost, m.complexity, m.birth, m.ref, m.parent_ref)

# ─── Running search statistics ──────────────────────────────────────────────

mutable struct RunningSearchStatistics
    frequencies::Vector{Float64}
    normalized_frequencies::Vector{Float64}
    window_size::Int
end

function RunningSearchStatistics(maxsize::Int; window_size::Int=100_000)
    freqs = ones(Float64, maxsize)
    RunningSearchStatistics(freqs, freqs ./ sum(freqs), window_size)
end

function update_size!(stats::RunningSearchStatistics, size::Int)
    if 1 <= size <= length(stats.frequencies)
        stats.frequencies[size] += 1.0
    end
end

function move_window!(stats::RunningSearchStatistics)
    smallest_frequency_allowed = 1.0
    max_loops = 1000
    total = sum(stats.frequencies)
    total <= stats.window_size && return
    difference = total - stats.window_size
    num_loops = 0
    while difference > 0
        indices = findall(>(smallest_frequency_allowed), stats.frequencies)
        isempty(indices) && break
        num_remaining = length(indices)
        max_subtract = minimum(stats.frequencies[indices]) - smallest_frequency_allowed
        amount = min(difference / num_remaining, max_subtract)
        stats.frequencies[indices] .-= amount
        total_subtracted = amount * num_remaining
        difference -= total_subtracted
        num_loops += 1
        if num_loops > max_loops || total_subtracted < 1e-6
            break
        end
    end
end

function normalize!(stats::RunningSearchStatistics)
    total = sum(stats.frequencies)
    if total <= 0
        stats.normalized_frequencies .= 1.0 / length(stats.frequencies)
    else
        stats.normalized_frequencies .= stats.frequencies ./ total
    end
end

# ─── Config ─────────────────────────────────────────────────────────────────

no_mutation_function(_engine, _parent) = nothing

Base.@kwdef mutable struct EngineConfig
    population_size::Int = 100
    populations::Int = 1
    niterations::Int = 100
    ncycles_per_iteration::Int = 1
    maxsize::Int = 30
    maxdepth::Int = 10
    max_evals::Union{Int, Nothing} = nothing
    parsimony::Float64 = 0.0
    tournament_selection_n::Int = 5
    tournament_selection_p::Float64 = 1.0
    crossover_probability::Float64 = 0.1
    skip_mutation_failures::Bool = true
    use_frequency::Bool = false
    use_frequency_in_tournament::Bool = false
    adaptive_parsimony_scaling::Float64 = 0.0
    annealing::Bool = false
    alpha::Float64 = 1.0
    perturbation_factor::Float64 = 0.1
    probability_negate_constant::Float64 = 0.0
    migration::Bool = false
    hof_migration::Bool = false
    fraction_replaced::Float64 = 0.0
    fraction_replaced_hof::Float64 = 0.0
    topn::Int = 10
    should_optimize_constants::Bool = false
    optimize_probability::Float64 = 0.0
    optimizer_iterations::Int = 8
    optimizer_nrestarts::Int = 1
    optimizer_f_calls_limit::Union{Int, Nothing} = nothing
    should_simplify::Bool = false
    binary_operators::Vector{Symbol} = [:+, :-, :*, :/]
    unary_operators::Vector{Symbol} = Symbol[]
    constants::Vector{Float64} = Float64[]
    mutation_function::Function = no_mutation_function
    constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    nested_constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    random_state::Int = 0
    # If log_file is a non-empty path, run_engine appends one JSONL line per
    # HOF update recording the cycle, eval_count, progress and full Pareto
    # frontier. Equations use the internal x0..xN variable names.
    log_file::String = ""
end

# ─── Search policy interface ────────────────────────────────────────────────

abstract type AbstractMinimalSRPolicy end

resolve_policy(policy::AbstractMinimalSRPolicy) = policy

# Coerce Py (or Julia-native) collections at the Python interop boundary.
as_strings(v) = v isa Py ? String.(pyconvert(Vector{Any}, v)) : Vector{String}(v)
as_symbols(v) = v isa Py ? Symbol.(pyconvert(Vector{Any}, v)) : [Symbol(x) for x in v]
as_floats(v) = v isa Py ? Float64.(pyconvert(Vector{Any}, v)) : Vector{Float64}(v)
as_matrix(v) = v isa Py ? Matrix{Float64}(pyconvert(Array, v)) : Matrix{Float64}(v)
as_vector(v) = v isa Py ? Float64.(vec(pyconvert(Array, v))) : Float64.(vec(v))

function as_symbol_float_dict(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Float64}(Symbol(k) => Float64(x) for (k, x) in d)
end

function as_constraint(v)
    v isa Py || return v
    out = pyconvert(Any, v)
    out isa AbstractVector && return [x isa Py ? pyconvert(Int, x) : Int(x) for x in out]
    return out isa Py ? pyconvert(Int, out) : out
end

function as_constraints(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Any}(Symbol(k) => as_constraint(x) for (k, x) in d)
end

function as_nested_constraint(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Int}(Symbol(k) => (x isa Py ? pyconvert(Int, x) : Int(x)) for (k, x) in d)
end

function as_nested_constraints(v)
    d = v isa Py ? pyconvert(Dict{Any, Any}, v) : v
    return Dict{Symbol, Any}(Symbol(k) => as_nested_constraint(x) for (k, x) in d)
end

# ─── Samplers ───────────────────────────────────────────────────────────────

# Knuth's algorithm, matches SymbolicRegression.jl/src/Utils.jl:poisson_sample
function poisson_sample(rng::AbstractRNG, λ::Float64)
    iszero(λ) && return 0
    k, p, L = 0, 1.0, exp(-λ)
    while p > L
        k += 1
        p *= rand(rng)
    end
    return k - 1
end

function weighted_choice(rng, arr, weights)
    r = rand(rng)
    acc = 0.0
    for (i, w) in enumerate(weights)
        acc += w
        r <= acc && return arr[i]
    end
    return arr[end]
end

sample_indices(rng, n::Int, k::Int) = randperm(rng, n)[1:min(k, n)]

# ─── Operator tables ────────────────────────────────────────────────────────

# Scalar-valued, broadcast-ready. `evaluate_tree` applies them with `.`.
safe_exp(x) = (o = exp(x); isfinite(o) ? o : NaN)
safe_log(x) = x > 0 ? log(x) : NaN
safe_sqrt(x) = x >= 0 ? sqrt(x) : NaN
safe_sin(x) = isfinite(x) ? sin(x) : NaN
safe_cos(x) = isfinite(x) ? cos(x) : NaN
safe_pow(x, y) = clamp(abs(x), 1e-12, 1e6) ^ clamp(y, -6.0, 6.0)
square(x) = x * x

const BINARY_OP_FNS = Dict{Symbol, Function}(
    :+ => +,
    :- => -,
    :* => *,
    :/ => /,
    :^ => safe_pow,
)

const UNARY_OP_FNS = Dict{Symbol, Function}(
    :abs => abs,
    :exp => safe_exp,
    :log => safe_log,
    :sqrt => safe_sqrt,
    :sin => safe_sin,
    :cos => safe_cos,
    :square => square,
)

# ─── Evaluation ─────────────────────────────────────────────────────────────

evaluate_tree(n::ConstNode, X::Matrix{Float64}) = fill(n.value, size(X, 1))
evaluate_tree(n::VarNode, X::Matrix{Float64}) = @view X[:, n.feature]

function evaluate_tree(n::OpNode, X::Matrix{Float64})
    l = evaluate_tree(n.left, X)
    isnothing(n.right) && return UNARY_OP_FNS[n.op].(l)
    r = evaluate_tree(n.right, X)
    return BINARY_OP_FNS[n.op].(l, r)
end

# ─── Tree walk ──────────────────────────────────────────────────────────────

const NodeTriple = Tuple{Node, Union{OpNode, Nothing}, Union{Symbol, Nothing}}

function nodes_with_parent(root::Node)
    out = NodeTriple[]
    stack = NodeTriple[(root, nothing, nothing)]
    while !isempty(stack)
        node, parent, side = pop!(stack)
        push!(out, (node, parent, side))
        if node isa OpNode
            !isnothing(node.right) && push!(stack, (node.right, node, :right))
            push!(stack, (node.left, node, :left))
        end
    end
    return out
end

leaf_nodes(root::Node) = [n for (n, _, _) in nodes_with_parent(root) if isleaf(n)]
constant_nodes(root::Node) = [n for n in leaf_nodes(root) if n isa ConstNode]

function replace_subtree(root::Node, parent::Union{OpNode, Nothing}, side::Union{Symbol, Nothing}, subtree::Node)
    isnothing(parent) && return subtree
    side === :left ? (parent.left = subtree) : (parent.right = subtree)
    return root
end

# ─── Engine ─────────────────────────────────────────────────────────────────

mutable struct RegularizedEvolutionEngine
    X::Matrix{Float64}
    y::Vector{Float64}
    cfg::EngineConfig
    rng::AbstractRNG
    n_features::Int
    binary_ops::Vector{Symbol}
    unary_ops::Vector{Symbol}
    birth_counter::Int
    ref_counter::Int
    eval_count::Int
    eval_budget::Union{Int, Nothing}
    loss_normalization::Float64
    current_temperature::Float64
end

function RegularizedEvolutionEngine(X::Matrix{Float64}, y::Vector{Float64}, cfg::EngineConfig)
    rng = Xoshiro(cfg.random_state)
    baseline_loss = _mean((y .- _mean(y)) .^ 2)
    !isfinite(baseline_loss) && (baseline_loss = 1.0)
    return RegularizedEvolutionEngine(
        X, y, cfg, rng,
        size(X, 2),
        copy(cfg.binary_operators),
        copy(cfg.unary_operators),
        0, 0, 0,
        isnothing(cfg.max_evals) ? nothing : max(0, cfg.max_evals),
        baseline_loss >= 0.01 ? baseline_loss : 0.01,
        1.0,
    )
end

function next_birth!(engine::RegularizedEvolutionEngine)
    engine.birth_counter += 1
    return engine.birth_counter
end

function next_ref!(engine::RegularizedEvolutionEngine)
    engine.ref_counter += 1
    return engine.ref_counter
end

budget_remaining(engine::RegularizedEvolutionEngine) = isnothing(engine.eval_budget) ? nothing : max(0, engine.eval_budget - engine.eval_count)
has_budget(engine::RegularizedEvolutionEngine) = isnothing(engine.eval_budget) || engine.eval_count < engine.eval_budget

# ─── Random tree construction ───────────────────────────────────────────────

function random_terminal(engine::RegularizedEvolutionEngine)
    if rand(engine.rng) < 0.5
        return VarNode(rand(engine.rng, 1:engine.n_features))
    elseif !isempty(engine.cfg.constants)
        return ConstNode(Float64(rand(engine.rng, engine.cfg.constants)))
    end
    return ConstNode(randn(engine.rng))
end

function sample_operator_arity(engine::RegularizedEvolutionEngine; max_added_nodes=nothing)
    arities = Int[]
    weights = Float64[]
    if !isempty(engine.unary_ops) && (isnothing(max_added_nodes) || max_added_nodes >= 1)
        push!(arities, 1); push!(weights, length(engine.unary_ops))
    end
    if !isempty(engine.binary_ops) && (isnothing(max_added_nodes) || max_added_nodes >= 2)
        push!(arities, 2); push!(weights, length(engine.binary_ops))
    end
    isempty(arities) && return 0
    weights ./= sum(weights)
    return weighted_choice(engine.rng, arities, weights)
end

sample_operator(engine::RegularizedEvolutionEngine, arity::Int) = rand(engine.rng, arity == 1 ? engine.unary_ops : engine.binary_ops)

function append_random_op(engine::RegularizedEvolutionEngine, tree::Node; arity=nothing)
    tree = copy(tree)
    leaves = [(n, p, s) for (n, p, s) in nodes_with_parent(tree) if isleaf(n)]
    isempty(leaves) && return tree
    _, parent, side = rand(engine.rng, leaves)
    picked_arity = isnothing(arity) ? sample_operator_arity(engine) : arity
    picked_arity <= 0 && return tree
    op = sample_operator(engine, picked_arity)
    new_node = picked_arity == 1 ?
        OpNode(op, random_terminal(engine), nothing) :
        OpNode(op, random_terminal(engine), random_terminal(engine))
    return replace_subtree(tree, parent, side, new_node)
end

function prepend_random_op(engine::RegularizedEvolutionEngine, tree::Node)
    tree = copy(tree)
    arity = sample_operator_arity(engine)
    arity <= 0 && return tree
    op = sample_operator(engine, arity)
    arity == 1 && return OpNode(op, tree, nothing)
    return rand(engine.rng) < 0.5 ?
        OpNode(op, tree, random_terminal(engine)) :
        OpNode(op, random_terminal(engine), tree)
end

function insert_random_op(engine::RegularizedEvolutionEngine, tree::Node)
    tree = copy(tree)
    nodes = nodes_with_parent(tree)
    isempty(nodes) && return tree
    target, parent, side = rand(engine.rng, nodes)
    arity = sample_operator_arity(engine)
    arity <= 0 && return tree
    op = sample_operator(engine, arity)
    wrapped = if arity == 1
        OpNode(op, copy(target), nothing)
    elseif rand(engine.rng) < 0.5
        OpNode(op, copy(target), random_terminal(engine))
    else
        OpNode(op, random_terminal(engine), copy(target))
    end
    return replace_subtree(tree, parent, side, wrapped)
end

function random_tree_fixed_size(engine::RegularizedEvolutionEngine, node_count::Int)
    target = max(1, node_count)
    tree = random_terminal(engine)
    cur_size = 1
    while cur_size < target
        remaining = target - cur_size
        picked_arity = sample_operator_arity(engine; max_added_nodes=remaining)
        picked_arity <= 0 && break
        tree = append_random_op(engine, tree; arity=picked_arity)
        cur_size += picked_arity
    end
    return tree
end

function random_tree(engine::RegularizedEvolutionEngine, max_depth::Int, full::Bool; depth::Int=0)
    depth >= max_depth && return random_terminal(engine)
    !full && depth > 0 && rand(engine.rng) < 0.3 && return random_terminal(engine)
    if !isempty(engine.unary_ops) && rand(engine.rng) < 0.25
        op = rand(engine.rng, engine.unary_ops)
        return OpNode(op, random_tree(engine, max_depth, full; depth=depth + 1), nothing)
    end
    op = rand(engine.rng, engine.binary_ops)
    return OpNode(
        op,
        random_tree(engine, max_depth, full; depth=depth + 1),
        random_tree(engine, max_depth, full; depth=depth + 1),
    )
end

# ─── Constraints ────────────────────────────────────────────────────────────

function max_nestedness(node::Node, op::Symbol)
    function dfs(cur::Node)
        cur isa OpNode || return 0
        left_depth = dfs(cur.left)
        right_depth = isnothing(cur.right) ? 0 : dfs(cur.right)
        here = cur.op === op ? 1 : 0
        return here + max(left_depth, right_depth)
    end
    depth = dfs(node)
    is_self = (node isa OpNode && node.op === op) ? 1 : 0
    return depth - is_self
end

function check_constraints(engine::RegularizedEvolutionEngine, tree::Node)
    constraints = engine.cfg.constraints
    nested = engine.cfg.nested_constraints
    for (node, _, _) in nodes_with_parent(tree)
        node isa OpNode || continue
        if haskey(constraints, node.op)
            c = constraints[node.op]
            if isnothing(node.right)
                if c isa Real && c >= 0 && tree_size(node.left) > Int(c)
                    return false
                end
            elseif c isa AbstractVector && length(c) >= 2
                lmax, rmax = c[1], c[2]
                lmax isa Real && lmax >= 0 && tree_size(node.left) > Int(lmax) && return false
                rmax isa Real && rmax >= 0 && tree_size(node.right) > Int(rmax) && return false
            end
        end
        if haskey(nested, node.op)
            for (child_op, max_allowed) in nested[node.op]
                max_allowed < 0 && continue
                max_nestedness(node, child_op) > max_allowed && return false
            end
        end
    end
    return true
end

valid_tree(engine::RegularizedEvolutionEngine, tree::Node) =
    tree_size(tree) <= engine.cfg.maxsize &&
    tree_height(tree) <= engine.cfg.maxdepth &&
    check_constraints(engine, tree)

# ─── Fitness ────────────────────────────────────────────────────────────────

function evaluate_candidate(engine::RegularizedEvolutionEngine, tree::Node)
    has_budget(engine) || return nothing
    engine.eval_count += 1
    pred = Vector{Float64}(evaluate_tree(tree, engine.X))
    if length(pred) != length(engine.y) || !all(isfinite.(pred) .& (abs.(pred) .< 1e12))
        return (Inf, Inf, tree_size(tree))
    end
    mse = _mean((engine.y .- pred) .^ 2)
    complexity = tree_size(tree)
    cost = (mse / engine.loss_normalization) + engine.cfg.parsimony * complexity
    !isfinite(cost) && return (Inf, Inf, complexity)
    return (mse, cost, complexity)
end

function create_individual(engine::RegularizedEvolutionEngine, tree::Node; parent_ref::Union{Int, Nothing}=nothing)
    out = evaluate_candidate(engine, tree)
    out === nothing && return nothing
    loss, cost, complexity = out
    return Individual(tree, loss, cost, complexity, next_birth!(engine), next_ref!(engine), parent_ref)
end

spawn_from_existing(engine::RegularizedEvolutionEngine, member::Individual; parent_ref::Union{Int, Nothing}=nothing) =
    Individual(copy(member.tree), member.loss, member.cost, member.complexity, next_birth!(engine), next_ref!(engine), isnothing(parent_ref) ? member.ref : parent_ref)

# ─── Selection / survival ───────────────────────────────────────────────────

function tournament_select(population::Vector{Individual}, stats::RunningSearchStatistics, cfg::EngineConfig, rng)
    n = length(population)
    k = min(cfg.tournament_selection_n, n)
    candidate_idx = sample_indices(rng, n, k)
    adjusted_costs = Float64[]
    for idx in candidate_idx
        m = population[idx]
        cost = m.cost
        if cfg.use_frequency_in_tournament && 1 <= m.complexity <= cfg.maxsize
            freq = stats.normalized_frequencies[m.complexity]
            cost *= exp(clamp(cfg.adaptive_parsimony_scaling * freq, -50.0, 50.0))
        end
        push!(adjusted_costs, cost)
    end
    order = sortperm(adjusted_costs)
    p = cfg.tournament_selection_p
    p >= 1.0 && return candidate_idx[order[1]]
    weights = [p * ((1 - p) ^ i) for i in 0:(k - 1)]
    weights ./= sum(weights)
    place = weighted_choice(rng, 1:k, weights)
    return candidate_idx[order[place]]
end

function oldest_survival(population::Vector{Individual}, rng, exclude_indices::Set{Int})
    candidates = [i for i in eachindex(population) if i ∉ exclude_indices]
    isempty(candidates) && return rand(rng, eachindex(population))
    births = [population[i].birth for i in candidates]
    return candidates[argmin(births)]
end

# ─── Mutation operators ─────────────────────────────────────────────────────

function default_mutation(engine::RegularizedEvolutionEngine, tree::Node, mutation::Symbol)
    tree = copy(tree)
    nodes = nodes_with_parent(tree)
    leaves = leaf_nodes(tree)
    constants = [n for n in leaves if n isa ConstNode]
    mutation === :do_nothing && return tree

    if mutation === :mutate_constant
        if !isempty(constants)
            node = rand(engine.rng, constants)
            temperature = clamp(engine.current_temperature, 0.0, 1.0)
            bottom = 0.1
            max_change = engine.cfg.perturbation_factor * temperature + 1.0 + bottom
            factor = max_change ^ rand(engine.rng)
            rand(engine.rng) < 0.5 && (factor = 1.0 / factor)
            rand(engine.rng) > engine.cfg.probability_negate_constant && (factor *= -1.0)
            node.value = clamp(node.value * factor, -1e6, 1e6)
        end
        return tree

    elseif mutation === :mutate_feature
        vars_only = [n for n in leaves if n isa VarNode]
        if !isempty(vars_only)
            node = rand(engine.rng, vars_only)
            if engine.n_features > 1
                choices = [i for i in 1:engine.n_features if i != node.feature]
                node.feature = rand(engine.rng, choices)
            else
                node.feature = rand(engine.rng, 1:engine.n_features)
            end
        end
        return tree

    elseif mutation === :mutate_operator
        op_nodes = [n for (n, _, _) in nodes if n isa OpNode]
        if !isempty(op_nodes)
            node = rand(engine.rng, op_nodes)
            if isnothing(node.right) && !isempty(engine.unary_ops)
                node.op = rand(engine.rng, engine.unary_ops)
            elseif !isnothing(node.right) && !isempty(engine.binary_ops)
                node.op = rand(engine.rng, engine.binary_ops)
            end
        end
        return tree

    elseif mutation === :swap_operands
        binary_nodes = [n for (n, _, _) in nodes if n isa OpNode && !isnothing(n.right)]
        if !isempty(binary_nodes)
            node = rand(engine.rng, binary_nodes)
            node.left, node.right = node.right, node.left
        end
        return tree

    elseif mutation === :delete_node
        deletable = [(n, p, s) for (n, p, s) in nodes if n isa OpNode]
        if !isempty(deletable)
            node, parent, side = rand(engine.rng, deletable)
            repl = if isnothing(node.right)
                copy(node.left)
            else
                rand(engine.rng) < 0.5 ? copy(node.left) : copy(node.right)
            end
            tree = replace_subtree(tree, parent, side, repl)
        end
        return tree

    elseif mutation === :rotate_tree
        valid = Tuple{OpNode, Union{OpNode, Nothing}, Union{Symbol, Nothing}, Vector{Symbol}}[]
        for (node, parent, side) in nodes
            node isa OpNode || continue
            pivot_sides = Symbol[]
            node.left isa OpNode && push!(pivot_sides, :left)
            !isnothing(node.right) && node.right isa OpNode && push!(pivot_sides, :right)
            !isempty(pivot_sides) && push!(valid, (node, parent, side, pivot_sides))
        end
        if !isempty(valid)
            node, parent, side, pivot_sides = rand(engine.rng, valid)
            pivot_side = rand(engine.rng, pivot_sides)
            pivot = pivot_side === :left ? node.left : node.right
            pivot isa OpNode || return tree  # defensive, shouldn't hit
            grand_sides = Symbol[]
            push!(grand_sides, :left)  # OpNode always has left
            !isnothing(pivot.right) && push!(grand_sides, :right)
            grand_side = rand(engine.rng, grand_sides)
            grand_child = grand_side === :left ? pivot.left : pivot.right
            pivot_side === :left ? (node.left = grand_child) : (node.right = grand_child)
            grand_side === :left ? (pivot.left = node) : (pivot.right = node)
            tree = replace_subtree(tree, parent, side, pivot)
        end
        return tree

    elseif mutation === :add_node
        return rand(engine.rng) < 0.5 ? append_random_op(engine, tree) : prepend_random_op(engine, tree)

    elseif mutation === :insert_node
        return insert_random_op(engine, tree)

    elseif mutation === :simplify || mutation === :optimize
        return tree

    elseif mutation === :randomize
        target_size = rand(engine.rng, 1:engine.cfg.maxsize)
        return random_tree_fixed_size(engine, target_size)

    elseif startswith(string(mutation), "custom_mutation_")
        target, parent, side = rand(engine.rng, nodes)
        if !isempty(engine.binary_ops)
            op = rand(engine.rng, engine.binary_ops)
            other = random_terminal(engine)
            wrapped = rand(engine.rng) < 0.5 ? OpNode(op, copy(target), other) : OpNode(op, other, copy(target))
            tree = replace_subtree(tree, parent, side, wrapped)
        end
        return tree
    end
    return tree
end

function default_crossover(engine::RegularizedEvolutionEngine, parent1::Node, parent2::Node)
    t1 = copy(parent1)
    t2 = copy(parent2)
    n1 = nodes_with_parent(t1)
    n2 = nodes_with_parent(t2)
    node1, p1, s1 = rand(engine.rng, n1)
    node2, p2, s2 = rand(engine.rng, n2)
    t1 = replace_subtree(t1, p1, s1, copy(node2))
    t2 = replace_subtree(t2, p2, s2, copy(node1))
    return t1, t2
end

# ─── Migration ──────────────────────────────────────────────────────────────

function default_migration(engine::RegularizedEvolutionEngine, populations::Vector{Vector{Individual}}, pop_idx::Int, dominating::Vector{Individual})
    cfg = engine.cfg
    target = populations[pop_idx]
    isempty(target) && return
    function replace_from!(candidates::Vector{Individual}, frac::Float64)
        (isempty(candidates) || frac <= 0) && return
        n = poisson_sample(engine.rng, max(0.0, length(target) * frac))
        n <= 0 && return
        n = min(n, length(target))
        for _ in 1:n
            dst = rand(engine.rng, 1:length(target))
            src = copy(rand(engine.rng, candidates))
            src.birth = next_birth!(engine)
            src.ref = next_ref!(engine)
            target[dst] = src
        end
    end
    if cfg.migration
        best_of_each = Individual[]
        for pop in populations
            isempty(pop) && continue
            topk = sort(pop, by=m -> m.cost)[1:max(1, min(cfg.topn, length(pop)))]
            append!(best_of_each, copy.(topk))
        end
        replace_from!(best_of_each, cfg.fraction_replaced)
    end
    cfg.hof_migration && replace_from!(dominating, cfg.fraction_replaced_hof)
end

# ─── Acceptance ─────────────────────────────────────────────────────────────

function accept_candidate(engine::RegularizedEvolutionEngine, parent::Individual, child::Individual, stats::RunningSearchStatistics, temperature::Float64)
    isfinite(child.cost) || return false
    prob = 1.0
    if engine.cfg.annealing
        delta = child.cost - parent.cost
        denom = max(1e-8, temperature * engine.cfg.alpha)
        prob *= exp(clamp(-delta / denom, -50.0, 50.0))
    end
    if engine.cfg.use_frequency
        old_size = min(max(parent.complexity, 1), engine.cfg.maxsize)
        new_size = min(max(child.complexity, 1), engine.cfg.maxsize)
        old_f = stats.normalized_frequencies[old_size]
        new_f = stats.normalized_frequencies[new_size]
        prob *= old_f / max(new_f, 1e-12)
    end
    prob = min(prob, 1e6)
    prob >= 1.0 && return true
    return rand(engine.rng) < prob
end

# ─── Constant optimization ──────────────────────────────────────────────────

function set_constants!(tree::Node, vals::AbstractVector{<:Real})
    for (i, n) in enumerate(constant_nodes(tree))
        n.value = Float64(vals[i])
    end
end

function optimize_constants(engine::RegularizedEvolutionEngine, member::Individual)
    consts = constant_nodes(member.tree)
    isempty(consts) && return member, 0
    budget = budget_remaining(engine)
    !isnothing(budget) && budget <= 1 && return member, 0

    maxfun = isnothing(engine.cfg.optimizer_f_calls_limit) ? 10_000 : engine.cfg.optimizer_f_calls_limit
    !isnothing(budget) && (maxfun = min(maxfun, budget))

    initial = Float64[n.value for n in consts]
    best_tree = copy(member.tree)
    best_loss = member.loss
    evals_before = engine.eval_count

    function obj(vals::AbstractVector)
        has_budget(engine) || return 1e30
        trial = copy(member.tree)
        set_constants!(trial, vals)
        scored = evaluate_candidate(engine, trial)
        scored === nothing && return 1e30
        l = scored[1]
        return isfinite(l) ? l : 1e30
    end

    # Build starts: initial + nrestarts perturbed versions.
    starts = Vector{Vector{Float64}}()
    push!(starts, copy(initial))
    for _ in 1:max(0, engine.cfg.optimizer_nrestarts - 1)
        noise = Float64[randn(engine.rng) for _ in 1:length(initial)]
        push!(starts, initial .* (1.0 .+ 0.5 .* noise))
    end

    opts = Optim.Options(
        iterations=max(1, engine.cfg.optimizer_iterations),
        f_calls_limit=max(1, maxfun),
        g_tol=1e-8,
    )
    for x0 in starts
        has_budget(engine) || break
        try
            result = Optim.optimize(obj, x0, Optim.NelderMead(), opts)
            fval = Optim.minimum(result)
            if isfinite(fval) && fval < best_loss
                trial = copy(member.tree)
                set_constants!(trial, Optim.minimizer(result))
                best_tree = trial
                best_loss = fval
            end
        catch
            continue
        end
    end

    if best_loss < member.loss
        scored = evaluate_candidate(engine, best_tree)
        if scored !== nothing
            loss, cost, complexity = scored
            evals_used = engine.eval_count - evals_before
            return Individual(best_tree, loss, cost, complexity,
                              next_birth!(engine), next_ref!(engine), member.ref), evals_used
        end
    end
    return member, engine.eval_count - evals_before
end

# ─── Simplification (constant folding, recursive) ───────────────────────────

function simplify_tree(n::Node)
    n isa OpNode || return copy(n)
    new_left = simplify_tree(n.left)
    new_right = isnothing(n.right) ? nothing : simplify_tree(n.right)
    if new_left isa ConstNode && new_right isa ConstNode && haskey(BINARY_OP_FNS, n.op)
        out = BINARY_OP_FNS[n.op](new_left.value, new_right.value)
        isfinite(out) && return ConstNode(clamp(out, -1e6, 1e6))
    end
    return OpNode(n.op, new_left, new_right)
end

# ─── Main loop ──────────────────────────────────────────────────────────────

function initialize_population(engine::RegularizedEvolutionEngine)
    pop = Individual[]
    init_length = 3
    while length(pop) < engine.cfg.population_size && has_budget(engine)
        tree = random_terminal(engine)
        for _ in 1:init_length
            tree = append_random_op(engine, tree)
        end
        if !valid_tree(engine, tree)
            tree = random_tree(engine, min(3, engine.cfg.maxdepth), false)
        end
        member = create_individual(engine, tree)
        member !== nothing && push!(pop, member)
    end
    if isempty(pop)
        fallback_loss = _mean((engine.y .- _mean(engine.y)) .^ 2)
        push!(pop, Individual(
            ConstNode(_mean(engine.y)),
            fallback_loss,
            (fallback_loss / engine.loss_normalization) + engine.cfg.parsimony,
            1, next_birth!(engine), next_ref!(engine), nothing,
        ))
    end
    return pop
end

function regularized_cycle!(
    engine::RegularizedEvolutionEngine,
    population::Vector{Individual},
    stats::RunningSearchStatistics,
    temperature::Float64,
    mutation_function::Function,
)
    engine.current_temperature = clamp(temperature, 0.0, 1.0)
    n_evol_cycles = Int(ceil(length(population) / max(1, engine.cfg.tournament_selection_n)))
    for _ in 1:n_evol_cycles
        has_budget(engine) || return
        if rand(engine.rng) > engine.cfg.crossover_probability
            pidx = tournament_select(population, stats, engine.cfg, engine.rng)
            parent = population[pidx]
            child_tree = mutation_function(engine, parent)
            if child_tree === nothing
                engine.cfg.skip_mutation_failures && continue
                replacement = spawn_from_existing(engine, parent)
                victim = oldest_survival(population, engine.rng, Set{Int}())
                population[victim] = replacement
                continue
            end
            child = create_individual(engine, child_tree; parent_ref=parent.ref)
            child === nothing && return
            accepted = accept_candidate(engine, parent, child, stats, temperature)
            !accepted && engine.cfg.skip_mutation_failures && continue
            replacement = accepted ? child : spawn_from_existing(engine, parent)
            victim = oldest_survival(population, engine.rng, Set{Int}())
            population[victim] = replacement
        else
            p1 = tournament_select(population, stats, engine.cfg, engine.rng)
            p2 = tournament_select(population, stats, engine.cfg, engine.rng)
            if p1 == p2 && length(population) > 1
                p2 = (p2 % length(population)) + 1
            end
            pair = nothing
            for _attempt in 1:10
                t1, t2 = default_crossover(engine, population[p1].tree, population[p2].tree)
                if valid_tree(engine, t1) && valid_tree(engine, t2)
                    pair = (t1, t2)
                    break
                end
            end
            if pair === nothing
                engine.cfg.skip_mutation_failures && continue
                c1 = spawn_from_existing(engine, population[p1]; parent_ref=population[p1].ref)
                c2 = spawn_from_existing(engine, population[p2]; parent_ref=population[p2].ref)
            else
                t1, t2 = pair
                c1 = create_individual(engine, t1; parent_ref=population[p1].ref)
                c1 === nothing && return
                c2 = create_individual(engine, t2; parent_ref=population[p2].ref)
                c2 === nothing && return
            end
            v1 = oldest_survival(population, engine.rng, Set{Int}())
            v2 = oldest_survival(population, engine.rng, Set([v1]))
            population[v1] = c1
            population[v2] = c2
        end
    end
end

function optimize_and_simplify!(engine::RegularizedEvolutionEngine, population::Vector{Individual})
    for i in eachindex(population)
        member = population[i]
        if engine.cfg.should_simplify
            simplified = simplify_tree(member.tree)
            if valid_tree(engine, simplified)
                new_complexity = tree_size(simplified)
                new_cost = (member.loss / engine.loss_normalization) + engine.cfg.parsimony * new_complexity
                member = Individual(simplified, member.loss, new_cost, new_complexity,
                                    member.birth, member.ref, member.parent_ref)
            end
        end
        if engine.cfg.should_optimize_constants && rand(engine.rng) < engine.cfg.optimize_probability
            member, _ = optimize_constants(engine, member)
        end
        population[i] = member
    end
end

function calculate_pareto_frontier_from_dict(hof_by_complexity::Dict{Int, Individual})
    dominating = Individual[]
    best_so_far = Inf
    for c in sort(collect(keys(hof_by_complexity)))
        member = hof_by_complexity[c]
        if member.loss < best_so_far
            push!(dominating, copy(member))
            best_so_far = member.loss
        end
    end
    return dominating
end

# ── Baseline logging infra — do not remove or modify in experiments ──
# Escape a string for JSON output. Handles the subset of characters MiniSR
# equations actually produce (backslashes, quotes, control chars).
function _json_escape(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"'
            write(buf, "\\\"")
        elseif c == '\\'
            write(buf, "\\\\")
        elseif c == '\n'
            write(buf, "\\n")
        elseif c == '\r'
            write(buf, "\\r")
        elseif c == '\t'
            write(buf, "\\t")
        elseif c < ' '
            write(buf, string("\\u", lpad(string(UInt16(c), base=16), 4, '0')))
        else
            write(buf, c)
        end
    end
    return String(take!(buf))
end

function _json_float(x::Float64)
    if isnan(x); return "NaN"; end
    if isinf(x); return x > 0 ? "Infinity" : "-Infinity"; end
    return string(x)
end

function write_hof_log(engine::RegularizedEvolutionEngine, cycle::Int, dominating::Vector{Individual})
    path = engine.cfg.log_file
    isempty(path) && return
    budget = engine.eval_budget
    progress = (isnothing(budget) || budget == 0) ? 0.0 : engine.eval_count / budget
    try
        open(path, "a") do io
            write(io, "{\"cycle\":", string(cycle))
            write(io, ",\"eval_count\":", string(engine.eval_count))
            write(io, ",\"progress\":", _json_float(progress))
            write(io, ",\"frontier\":[")
            for (i, m) in enumerate(dominating)
                i > 1 && write(io, ",")
                eqn = node_string(m.tree)
                write(io, "{\"complexity\":", string(m.complexity))
                write(io, ",\"loss\":", _json_float(m.loss))
                write(io, ",\"equation\":\"", _json_escape(eqn), "\"}")
            end
            write(io, "]}\n")
        end
    catch err
        # Don't let logging errors kill the search.
        @warn "write_hof_log failed" path=path err=err
    end
end

# ─── Generic entry point ────────────────────────────────────────────────────

function fit_minimal_sr(X_in, y_in, variable_names_in;
    binary_operators=String["+", "-", "*", "/"],
    unary_operators=String[],
    constants=Float64[],
    mutation_function=no_mutation_function,
    policy,
    constraints=Dict{String, Any}(),
    nested_constraints=Dict{String, Any}(),
    kwargs...,
)
    X = as_matrix(X_in)
    y = as_vector(y_in)
    variable_names = as_strings(variable_names_in)
    # Normalize Py strings in kwargs (e.g. log_file) to Julia String.
    normalized_kwargs = Dict{Symbol, Any}()
    for (k, v) in pairs(kwargs)
        normalized_kwargs[k] = v isa Py ? pyconvert(Any, v) : v
    end
    cfg = EngineConfig(;
        binary_operators=as_symbols(binary_operators),
        unary_operators=as_symbols(unary_operators),
        constants=as_floats(constants),
        mutation_function=mutation_function,
        constraints=as_constraints(constraints),
        nested_constraints=as_nested_constraints(nested_constraints),
        normalized_kwargs...,
    )
    engine = RegularizedEvolutionEngine(X, y, cfg)
    dominating, n_evals = run_engine(engine, resolve_policy(policy))
    rows = Vector{Dict{String, Any}}()
    for m in sort(dominating, by=z -> z.complexity)
        eqn = node_string(m.tree)
        for i in reverse(eachindex(variable_names))
            eqn = replace(eqn, Regex("\\bx$(i - 1)\\b") => variable_names[i])
        end
        push!(rows, Dict("complexity" => m.complexity, "loss" => m.loss, "equation" => eqn))
    end
    return Dict("rows" => rows, "n_evals" => n_evals)
end

include("MinimalSRConfig.jl")

end
