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
    constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    nested_constraints::Dict{Symbol, Any} = Dict{Symbol, Any}()
    random_state::Int = 0
    # If log_file is a non-empty path, PySR-compatible searches append one JSONL
    # line per archive update recording the cycle, eval_count, progress and full
    # Pareto frontier. Equations use the internal x0..xN variable names.
    log_file::String = ""
end

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

# ─── Policy configuration helpers ──────────────────────────────────────────

const MS = @__MODULE__
const Population = Vector{Individual}

function minimal_random_subtree(engine::RegularizedEvolutionEngine)
    rand(engine.rng) < 0.5 && return random_terminal(engine)
    arity = sample_operator_arity(engine; max_added_nodes=2)
    arity <= 0 && return random_terminal(engine)
    op = sample_operator(engine, arity)
    arity == 1 && return OpNode(op, random_terminal(engine), nothing)
    return OpNode(op, random_terminal(engine), random_terminal(engine))
end

function minimal_replace_subtree_mutation(engine::RegularizedEvolutionEngine, tree::Node)
    tree = copy(tree)
    nodes = nodes_with_parent(tree)
    isempty(nodes) && return tree
    _, parent, side = rand(engine.rng, nodes)
    subtree = minimal_random_subtree(engine)
    return replace_subtree(tree, parent, side, subtree)
end

function default_replace_subtree_mutation(engine::RegularizedEvolutionEngine, parent::Individual)
    for _attempt in 1:10
        proposal = minimal_replace_subtree_mutation(engine, parent.tree)
        valid_tree(engine, proposal) && return proposal
    end
    return nothing
end

function simple_tournament_select(population::Vector{Individual}, cfg::EngineConfig, rng)
    n = length(population)
    k = min(max(1, cfg.tournament_selection_n), n)
    candidate_idx = sample_indices(rng, n, k)
    costs = [population[i].cost for i in candidate_idx]
    return candidate_idx[argmin(costs)]
end

function topk_survive!(population::Vector{Individual}, offspring::Vector{Individual}, cfg::EngineConfig)
    isempty(offspring) && return nothing
    combined = vcat(copy.(population), copy.(offspring))
    sort!(combined, by=m -> (m.cost, m.loss, m.complexity, m.birth))
    keep = min(length(population), cfg.population_size, length(combined))
    empty!(population)
    append!(population, combined[1:keep])
    return nothing
end

function default_minimal_kwargs(; kwargs...)
    defaults = (
        binary_operators=String["+", "-", "*", "/"],
        unary_operators=String[],
        constants=Float64[],
        constraints=Dict{String, Any}(),
        nested_constraints=Dict{String, Any}(),
        population_size=100,
        populations=1,
        niterations=100,
        ncycles_per_iteration=1,
        maxsize=30,
        maxdepth=10,
        parsimony=0.0,
        tournament_selection_n=5,
        tournament_selection_p=1.0,
        crossover_probability=0.1,
        skip_mutation_failures=true,
        use_frequency=false,
        use_frequency_in_tournament=false,
        adaptive_parsimony_scaling=0.0,
        annealing=false,
        alpha=1.0,
        perturbation_factor=0.1,
        probability_negate_constant=0.0,
        migration=false,
        hof_migration=false,
        fraction_replaced=0.0,
        fraction_replaced_hof=0.0,
        topn=10,
        should_optimize_constants=false,
        optimize_probability=0.0,
        optimizer_iterations=8,
        optimizer_nrestarts=1,
        should_simplify=false,
    )
    return merge(defaults, (; kwargs...))
end

const PYSR_COMPAT_MUTATION_NAMES = [
    :add_node,
    :insert_node,
    :delete_node,
    :do_nothing,
    :mutate_constant,
    :mutate_operator,
    :mutate_feature,
    :swap_operands,
    :rotate_tree,
    :randomize,
    :simplify,
    :optimize,
    :custom_mutation_1,
    :custom_mutation_2,
    :custom_mutation_3,
    :custom_mutation_4,
    :custom_mutation_5,
    :custom_mutation_6,
]

const PYSR_COMPAT_MUTATION_WEIGHTS = Dict{Symbol, Float64}(
    :add_node => 2.47,
    :insert_node => 0.0112,
    :delete_node => 0.87,
    :do_nothing => 0.273,
    :mutate_constant => 0.0346,
    :mutate_operator => 0.293,
    :mutate_feature => 0.1,
    :swap_operands => 0.198,
    :rotate_tree => 4.26,
    :randomize => 0.000502,
    :simplify => 0.00209,
    :optimize => 0.0,
    :custom_mutation_1 => 0.0,
    :custom_mutation_2 => 0.0,
    :custom_mutation_3 => 0.0,
    :custom_mutation_4 => 0.0,
    :custom_mutation_5 => 0.0,
    :custom_mutation_6 => 0.0,
)

function conditioned_pysr_mutation_weights(engine::RegularizedEvolutionEngine, tree::Node)
    nodes = nodes_with_parent(tree)
    leaves = leaf_nodes(tree)
    n_constants = count(n -> n isa ConstNode, leaves)
    names = copy(PYSR_COMPAT_MUTATION_NAMES)
    w = Dict(name => max(0.0, PYSR_COMPAT_MUTATION_WEIGHTS[name]) for name in names)
    if isleaf(tree)
        w[:mutate_operator] = 0.0
        w[:swap_operands] = 0.0
        w[:delete_node] = 0.0
        w[:simplify] = 0.0
        if tree isa VarNode
            w[:optimize] = 0.0
            w[:mutate_constant] = 0.0
        else
            w[:mutate_feature] = 0.0
        end
    end
    if !any(n isa OpNode && !isnothing(n.right) for (n, _, _) in nodes)
        w[:swap_operands] = 0.0
    end
    w[:mutate_constant] *= min(8, n_constants) / 8.0
    engine.n_features <= 1 && (w[:mutate_feature] = 0.0)
    tree_size(tree) >= engine.cfg.maxsize && ((w[:add_node] = 0.0); (w[:insert_node] = 0.0))
    !engine.cfg.should_simplify && (w[:simplify] = 0.0)
    return names, w
end

function sample_pysr_mutation_choice(engine::RegularizedEvolutionEngine, tree::Node)
    names, w = conditioned_pysr_mutation_weights(engine, tree)
    weights = [w[n] for n in names]
    total = sum(weights)
    total <= 0 && return :do_nothing
    weights ./= total
    return weighted_choice(engine.rng, names, weights)
end

function pysr_weighted_mutation(engine::RegularizedEvolutionEngine, parent::Individual)
    mutation = sample_pysr_mutation_choice(engine, parent.tree)
    for _attempt in 1:10
        proposal = default_mutation(engine, parent.tree, mutation)
        valid_tree(engine, proposal) && return proposal
    end
    return nothing
end

function pysr_compat_kwargs(; kwargs...)
    defaults = (
        binary_operators=String["+", "-", "*", "/"],
        unary_operators=String["sin", "cos", "exp", "log", "sqrt", "square"],
        constants=Float64[],
        constraints=Dict{String, Any}(),
        nested_constraints=Dict{String, Any}(),
        population_size=27,
        populations=31,
        niterations=100,
        ncycles_per_iteration=380,
        maxsize=30,
        maxdepth=16,
        parsimony=0.0,
        tournament_selection_n=15,
        tournament_selection_p=0.982,
        crossover_probability=0.0259,
        skip_mutation_failures=true,
        use_frequency=true,
        use_frequency_in_tournament=true,
        adaptive_parsimony_scaling=1040.0,
        annealing=false,
        alpha=3.17,
        perturbation_factor=0.129,
        probability_negate_constant=0.00743,
        migration=true,
        hof_migration=true,
        fraction_replaced=0.00036,
        fraction_replaced_hof=0.0614,
        topn=12,
        should_optimize_constants=true,
        optimize_probability=0.14,
        optimizer_iterations=8,
        optimizer_nrestarts=2,
        optimizer_f_calls_limit=10_000,
        should_simplify=true,
    )
    return merge(defaults, (; kwargs...))
end

# ─── State/config ───────────────────────────────────────────────────────────

abstract type AbstractPolicyState end

mutable struct BasicPolicyState <: AbstractPolicyState
    archive::Vector{Individual}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
end

function BasicPolicyState(cfg::MS.EngineConfig)
    return BasicPolicyState(Individual[], false, zeros(Int, max(1, cfg.populations)))
end

mutable struct PySRPolicyState <: AbstractPolicyState
    best_by_complexity::Dict{Int, Individual}
    frontier::Vector{Individual}
    per_population_stats::Vector{MS.RunningSearchStatistics}
    current_temperature::Float64
    counted_population_cycles::Vector{Int}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
    last_logged_cycle::Int
end

function PySRPolicyState(cfg::MS.EngineConfig)
    n = max(1, cfg.populations)
    return PySRPolicyState(
        Dict{Int, Individual}(),
        Individual[],
        [MS.RunningSearchStatistics(cfg.maxsize) for _ in 1:n],
        1.0,
        zeros(Int, n),
        false,
        zeros(Int, n),
        -2,
    )
end

mutable struct EngineState{P<:AbstractPolicyState}
    engine::MS.RegularizedEvolutionEngine
    populations::Vector{Population}
    policy_state::P
    current_iteration::Int
    current_population::Int
    current_inner_cycle::Int
    completed_population_cycles::Vector{Int}
end

Base.@kwdef struct MinimalSRPolicy
    init_state::Function
    loss_function::Function
    survival::Function
    selection::Function
    mutation::Function
    acceptance::Function
    crossover::Function
    update_population::Function
    update_state!::Function
end

Base.@kwdef struct MinimalSRConfig
    engine_config::MS.EngineConfig
    policy::MinimalSRPolicy
end

engine(state::EngineState) = state.engine
current_stats(state::EngineState{PySRPolicyState}) =
    state.policy_state.per_population_stats[state.current_population]

result_members(policy_state::BasicPolicyState) = policy_state.archive
result_members(policy_state::PySRPolicyState) = policy_state.frontier

function engine_config_from_kwargs(; kwargs...)
    return MS.EngineConfig(; kwargs...)
end

function engine_config_from_namedtuple(nt::NamedTuple)
    engine_fields = Set(fieldnames(MS.EngineConfig))
    kwargs = Dict{Symbol, Any}()
    for key in keys(nt)
        key in engine_fields || continue
        value = nt[key]
        value = value isa Py ? pyconvert(Any, value) : value
        if key === :binary_operators || key === :unary_operators
            value = as_symbols(value)
        elseif key === :constants
            value = as_floats(value)
        elseif key === :constraints
            value = as_constraints(value)
        elseif key === :nested_constraints
            value = as_nested_constraints(value)
        end
        kwargs[key] = value
    end
    return MS.EngineConfig(; kwargs...)
end

function initialize_state(X, y, variable_names, config::MinimalSRConfig)
    _ = variable_names
    eng = MS.RegularizedEvolutionEngine(Matrix{Float64}(X), Float64.(vec(y)), config.engine_config)
    policy_state = config.policy.init_state(config)
    return EngineState(
        eng,
        Population[],
        policy_state,
        0,
        1,
        1,
        Int[],
    )
end

function initialize_populations!(state::EngineState, _X, _y, _config::MinimalSRConfig)
    n = max(1, state.engine.cfg.populations)
    state.populations = [MS.initialize_population(state.engine) for _ in 1:n]
    state.completed_population_cycles = zeros(Int, n)
    return nothing
end

has_budget(state::EngineState, _config::MinimalSRConfig) = MS.has_budget(state.engine)

function cycles_for_population(population::Population, cfg::MS.EngineConfig)
    return Int(ceil(length(population) / max(1, cfg.tournament_selection_n)))
end

should_crossover(rng, cfg::MS.EngineConfig, population::Population) =
    length(population) >= 2 && rand(rng) <= cfg.crossover_probability

function make_individual(tree::Node, _X, _y, state::EngineState, config::MinimalSRConfig;
                         parent_ref::Union{Int, Nothing}=nothing)
    out = config.policy.loss_function(tree, state, config)
    out === nothing && return nothing
    loss, cost, complexity = out
    return Individual(
        tree,
        loss,
        cost,
        complexity,
        MS.next_birth!(state.engine),
        MS.next_ref!(state.engine),
        parent_ref,
    )
end

function trees_from_crossover_result(result)
    result === nothing && return Node[]
    result isa Node && return Node[result]
    result isa Tuple && return Node[tree for tree in result if tree isa Node]
    result isa AbstractVector && return Node[tree for tree in result if tree isa Node]
    return Node[]
end

# ─── Generic engine ─────────────────────────────────────────────────────────

function fit_minimal_sr(X_in, y_in, variable_names_in; config::MinimalSRConfig)
    X = MS.as_matrix(X_in)
    y = MS.as_vector(y_in)
    variable_names = MS.as_strings(variable_names_in)
    state = initialize_state(X, y, variable_names, config)
    initialize_populations!(state, X, y, config)
    config.policy.update_state!(state.populations, state, config)

    for iteration in 1:config.engine_config.niterations
        has_budget(state, config) || break
        state.current_iteration = iteration

        for pop_index in eachindex(state.populations)
            has_budget(state, config) || break
            state.current_population = pop_index

            for inner in 1:max(1, config.engine_config.ncycles_per_iteration)
                has_budget(state, config) || break
                state.current_inner_cycle = inner
                config.policy.update_state!(state.populations, state, config)
                regularized_cycle!(state, pop_index, X, y, config)
            end

            has_budget(state, config) && optimize_and_simplify_population!(
                state.populations[pop_index], state, config
            )
            state.completed_population_cycles[pop_index] += 1
            config.policy.update_state!(state.populations, state, config)
            state.populations = config.policy.update_population(
                state.policy_state, state.populations, state, config
            )
        end
    end

    return format_result(state.policy_state, state, variable_names)
end

function regularized_cycle!(state::EngineState, pop_index::Int, X, y, config::MinimalSRConfig)
    cfg = config.engine_config
    policy = config.policy
    population = state.populations[pop_index]

    for _ in 1:cycles_for_population(population, cfg)
        has_budget(state, config) || break

        if should_crossover(state.engine.rng, cfg, population)
            parent_a = policy.selection(population, state, config)
            parent_b = policy.selection(population, state, config)
            if parent_a === parent_b && length(population) > 1
                idx = something(findfirst(member -> member === parent_b, population), 1)
                parent_b = population[(idx % length(population)) + 1]
            end
            result = policy.crossover(parent_a, parent_b, state, config)
            candidates = Individual[]
            for (tree, parent) in zip(trees_from_crossover_result(result), (parent_a, parent_b))
                child = make_individual(tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                push!(candidates, child)
            end
            isempty(candidates) && cfg.skip_mutation_failures && continue
            state.populations[pop_index] = policy.survival(population, candidates, state, config)
        else
            parent = policy.selection(population, state, config)
            child_tree = policy.mutation(parent, state, config)
            if child_tree === nothing
                cfg.skip_mutation_failures && continue
                replacement = MS.spawn_from_existing(state.engine, parent)
                state.populations[pop_index] = policy.survival(
                    population, [replacement], state, config
                )
            else
                child = make_individual(child_tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                if policy.acceptance(parent, child, state, config)
                    state.populations[pop_index] = policy.survival(
                        population, [child], state, config
                    )
                elseif !cfg.skip_mutation_failures
                    replacement = MS.spawn_from_existing(state.engine, parent)
                    state.populations[pop_index] = policy.survival(
                        population, [replacement], state, config
                    )
                end
            end
        end
        population = state.populations[pop_index]
    end
    return state
end

function optimize_and_simplify_population!(
    population::Population, state::EngineState, _config::MinimalSRConfig
)
    return MS.optimize_and_simplify!(state.engine, population)
end

function format_result(
    policy_state::AbstractPolicyState, state::EngineState, variable_names::Vector{String}
)
    rows = Vector{Dict{String, Any}}()
    members = result_members(policy_state)
    if isempty(members)
        best = state.populations[1][1]
        for pop in state.populations, member in pop
            member.cost < best.cost && (best = member)
        end
        members = [best]
    end
    for member in sort(members, by=m -> m.complexity)
        eqn = MS.node_string(member.tree)
        for i in reverse(eachindex(variable_names))
            eqn = replace(eqn, Regex("\\bx$(i - 1)\\b") => variable_names[i])
        end
        push!(rows, Dict("complexity" => member.complexity, "loss" => member.loss, "equation" => eqn))
    end
    return Dict("rows" => rows, "n_evals" => state.engine.eval_count)
end

# ─── Shared callbacks ───────────────────────────────────────────────────────

mse_loss_function(tree::Node, state::EngineState, _config::MinimalSRConfig) =
    MS.evaluate_candidate(state.engine, tree)

always_accept(_parent::Individual, _child::Individual, _state::EngineState, _config::MinimalSRConfig) =
    true

function subtree_swap_crossover(parent_a::Individual, parent_b::Individual, state::EngineState,
                                _config::MinimalSRConfig)
    for _attempt in 1:10
        t1, t2 = MS.default_crossover(state.engine, parent_a.tree, parent_b.tree)
        MS.valid_tree(state.engine, t1) && MS.valid_tree(state.engine, t2) && return (t1, t2)
    end
    return nothing
end

# ─── Basic policy ───────────────────────────────────────────────────────────

function basic_selection(population::Population, state::EngineState, config::MinimalSRConfig)
    idx = MS.simple_tournament_select(population, config.engine_config, state.engine.rng)
    return population[idx]
end

function basic_survival(population::Population, candidates::Vector{Individual},
                        _state::EngineState, config::MinimalSRConfig)
    output = copy(population)
    MS.topk_survive!(output, candidates, config.engine_config)
    return output
end

function basic_mutation(parent::Individual, state::EngineState, _config::MinimalSRConfig)
    return MS.default_replace_subtree_mutation(state.engine, parent)
end

init_basic_state(config::MinimalSRConfig) = BasicPolicyState(config.engine_config)

function basic_update_state!(populations::Vector{Population}, state::EngineState{BasicPolicyState},
                             config::MinimalSRConfig)
    policy_state = state.policy_state
    pop_indices = if !policy_state.archive_initialized
        collect(eachindex(populations))
    else
        [
            i for i in eachindex(populations) if
            state.completed_population_cycles[i] > policy_state.archive_counted_population_cycles[i]
        ]
    end
    isempty(pop_indices) && return nothing

    combined = copy(policy_state.archive)
    for i in pop_indices
        append!(combined, copy.(populations[i]))
    end
    sort!(combined, by=m -> (m.loss, m.cost, m.complexity, m.birth))
    empty!(policy_state.archive)
    seen = Set{String}()
    for member in combined
        isfinite(member.loss) || continue
        key = MS.node_string(member.tree)
        key in seen && continue
        push!(seen, key)
        push!(policy_state.archive, copy(member))
        length(policy_state.archive) >= max(1, config.engine_config.topn) && break
    end
    for i in pop_indices
        policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
    end
    policy_state.archive_initialized = true
    return nothing
end

basic_update_population(_policy_state::BasicPolicyState, populations::Vector{Population},
                        _state::EngineState, _config::MinimalSRConfig) = populations

function basic_policy()
    return MinimalSRPolicy(;
        init_state=init_basic_state,
        loss_function=mse_loss_function,
        survival=basic_survival,
        selection=basic_selection,
        mutation=basic_mutation,
        acceptance=always_accept,
        crossover=subtree_swap_crossover,
        update_population=basic_update_population,
        update_state! = basic_update_state!,
    )
end

function default_minimal_config(; kwargs...)
    nt = default_minimal_kwargs(; kwargs...)
    cfg = engine_config_from_namedtuple(nt)
    return MinimalSRConfig(; engine_config=cfg, policy=basic_policy())
end

fit_default_sr(args...; kwargs...) =
    fit_minimal_sr(args...; config=default_minimal_config(; kwargs...))

# ─── PySR-compatible policy ─────────────────────────────────────────────────

function pysr_selection(population::Population, state::EngineState, config::MinimalSRConfig)
    stats = current_stats(state)
    idx = MS.tournament_select(population, stats, config.engine_config, state.engine.rng)
    return population[idx]
end

function pysr_survival(population::Population, candidates::Vector{Individual},
                       state::EngineState, _config::MinimalSRConfig)
    output = copy(population)
    used = Set{Int}()
    for candidate in candidates
        isempty(output) && break
        victim = MS.oldest_survival(output, state.engine.rng, used)
        push!(used, victim)
        output[victim] = candidate
    end
    return output
end

function pysr_mutation(parent::Individual, state::EngineState, _config::MinimalSRConfig)
    return MS.pysr_weighted_mutation(state.engine, parent)
end

function pysr_acceptance(parent::Individual, child::Individual, state::EngineState,
                         _config::MinimalSRConfig)
    return MS.accept_candidate(
        state.engine, parent, child, current_stats(state), state.policy_state.current_temperature
    )
end

init_pysr_state(config::MinimalSRConfig) = PySRPolicyState(config.engine_config)

function pysr_update_state!(populations::Vector{Population}, state::EngineState{PySRPolicyState},
                            config::MinimalSRConfig)
    policy_state = state.policy_state
    was_archive_initialized = policy_state.archive_initialized
    archive_pop_indices = if !policy_state.archive_initialized
        collect(eachindex(populations))
    else
        [
            i for i in eachindex(populations) if
            state.completed_population_cycles[i] > policy_state.archive_counted_population_cycles[i]
        ]
    end
    if !isempty(archive_pop_indices)
        for i in archive_pop_indices, member in populations[i]
            c = member.complexity
            1 <= c <= config.engine_config.maxsize || continue
            best = get(policy_state.best_by_complexity, c, nothing)
            if isnothing(best) || member.loss < best.loss
                policy_state.best_by_complexity[c] = copy(member)
            end
        end
        policy_state.frontier = MS.calculate_pareto_frontier_from_dict(
            policy_state.best_by_complexity
        )
        for i in archive_pop_indices
            policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
        end
        policy_state.archive_initialized = true
        if !isempty(config.engine_config.log_file)
            cycle = was_archive_initialized ?
                (state.current_iteration - 1) * length(populations) + (state.current_population - 1) :
                -1
            if cycle > policy_state.last_logged_cycle
                MS.write_hof_log(state.engine, cycle, policy_state.frontier)
                policy_state.last_logged_cycle = cycle
            end
        end
    end

    cfg = config.engine_config
    stats = current_stats(state)

    MS.normalize!(stats)
    if cfg.annealing && cfg.ncycles_per_iteration > 1
        denom = max(1, cfg.ncycles_per_iteration - 1)
        policy_state.current_temperature = 1.0 - (state.current_inner_cycle - 1) / denom
    else
        policy_state.current_temperature = 1.0
    end
    state.engine.current_temperature = clamp(policy_state.current_temperature, 0.0, 1.0)

    pop_idx = state.current_population
    completed = state.completed_population_cycles[pop_idx]
    if completed > policy_state.counted_population_cycles[pop_idx]
        pop = populations[state.current_population]
        for member in pop
            MS.update_size!(stats, member.complexity)
        end
        MS.move_window!(stats)
        policy_state.counted_population_cycles[pop_idx] = completed
    end
    return nothing
end

function pysr_update_population(policy_state::PySRPolicyState, populations::Vector{Population},
                                state::EngineState, _config::MinimalSRConfig)
    MS.default_migration(state.engine, populations, state.current_population, policy_state.frontier)
    return populations
end

function pysr_policy()
    return MinimalSRPolicy(;
        init_state=init_pysr_state,
        loss_function=mse_loss_function,
        survival=pysr_survival,
        selection=pysr_selection,
        mutation=pysr_mutation,
        acceptance=pysr_acceptance,
        crossover=subtree_swap_crossover,
        update_population=pysr_update_population,
        update_state! = pysr_update_state!,
    )
end

function pysr_compat_config(; kwargs...)
    nt = pysr_compat_kwargs(; kwargs...)
    cfg = engine_config_from_namedtuple(nt)
    return MinimalSRConfig(; engine_config=cfg, policy=pysr_policy())
end

fit_pysr_compat_sr(args...; kwargs...) =
    fit_minimal_sr(args...; config=pysr_compat_config(; kwargs...))

end
