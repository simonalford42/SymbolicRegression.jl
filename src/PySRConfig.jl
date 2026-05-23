# PySR policy configuration for SkeletonSR.jl.

module PySRConfig

using Logging: @warn
using Random: AbstractRNG, rand, randn, randperm

using ..SkeletonSR:
    AbstractPolicyState,
    ConstNode,
    EngineConfig,
    EngineState,
    Individual,
    Node,
    OpNode,
    Population,
    EvolutionEngine,
    SkeletonSRConfig,
    SkeletonSRPolicy,
    VarNode,
    append_random_op,
    engine_config_from_namedtuple,
    evaluate_tree,
    fit_skeleton_sr,
    insert_random_op,
    isleaf,
    leaf_nodes,
    next_birth!,
    next_ref!,
    node_string,
    nodes_with_parent,
    optimize_and_simplify!,
    prepend_random_op,
    random_tree_fixed_size,
    replace_subtree,
    tree_size,
    valid_tree,
    weighted_choice

# ─── Running search statistics ──────────────────────────────────────────────

Base.@kwdef struct PySROptions
    parsimony::Float64 = 0.0
    tournament_selection_n::Int = 15
    tournament_selection_p::Float64 = 0.982
    crossover_probability::Float64 = 0.0259
    skip_mutation_failures::Bool = true
    use_frequency::Bool = true
    use_frequency_in_tournament::Bool = true
    adaptive_parsimony_scaling::Float64 = 1040.0
    annealing::Bool = false
    alpha::Float64 = 3.17
    perturbation_factor::Float64 = 0.129
    probability_negate_constant::Float64 = 0.00743
    migration::Bool = true
    hof_migration::Bool = true
    fraction_replaced::Float64 = 0.00036
    fraction_replaced_hof::Float64 = 0.0614
    topn::Int = 12
    should_optimize_constants::Bool = true
    optimize_probability::Float64 = 0.14
    optimizer_iterations::Int = 8
    optimizer_nrestarts::Int = 2
    optimizer_f_calls_limit::Union{Int, Nothing} = 10_000
    should_simplify::Bool = true
    log_file::String = ""
end

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

# ─── PySR low-level operators ───────────────────────────────────────────────

sample_indices(rng, n::Int, k::Int) = randperm(rng, n)[1:min(k, n)]

# Knuth's algorithm, matches SymbolicRegression.jl/src/Utils.jl:poisson_sample.
function poisson_sample(rng::AbstractRNG, λ::Float64)
    iszero(λ) && return 0
    k, p, L = 0, 1.0, exp(-λ)
    while p > L
        k += 1
        p *= rand(rng)
    end
    return k - 1
end

function tournament_select(population::Vector{Individual}, stats::RunningSearchStatistics,
                           options::PySROptions, maxsize::Int, rng)
    n = length(population)
    k = min(options.tournament_selection_n, n)
    candidate_idx = sample_indices(rng, n, k)
    adjusted_costs = Float64[]
    for idx in candidate_idx
        member = population[idx]
        cost = member.cost
        if options.use_frequency_in_tournament && 1 <= member.complexity <= maxsize
            freq = stats.normalized_frequencies[member.complexity]
            cost *= exp(clamp(options.adaptive_parsimony_scaling * freq, -50.0, 50.0))
        end
        push!(adjusted_costs, cost)
    end
    order = sortperm(adjusted_costs)
    p = options.tournament_selection_p
    p >= 1.0 && return candidate_idx[order[1]]
    weights = [p * ((1 - p) ^ i) for i in 0:(k - 1)]
    weights ./= sum(weights)
    place = weighted_choice(rng, 1:k, weights)
    return candidate_idx[order[place]]
end

function oldest_survival(population::Vector{Individual}, rng, exclude_indices::Set{Int})
    candidates = [i for i in eachindex(population) if !(i in exclude_indices)]
    isempty(candidates) && return rand(rng, eachindex(population))
    births = [population[i].birth for i in candidates]
    return candidates[argmin(births)]
end

function pysr_apply_mutation(
    engine::EvolutionEngine, tree::Node, mutation::Symbol, options::PySROptions
)
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
            max_change = options.perturbation_factor * temperature + 1.0 + bottom
            factor = max_change ^ rand(engine.rng)
            rand(engine.rng) < 0.5 && (factor = 1.0 / factor)
            rand(engine.rng) > options.probability_negate_constant && (factor *= -1.0)
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
            pivot isa OpNode || return tree
            grand_sides = Symbol[:left]
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
    end
    return tree
end

function pysr_subtree_crossover(engine::EvolutionEngine, parent1::Node, parent2::Node)
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

function pysr_migration(engine::EvolutionEngine, populations::Vector{Vector{Individual}},
                        pop_idx::Int, dominating::Vector{Individual}, options::PySROptions)
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
    if options.migration
        best_of_each = Individual[]
        for pop in populations
            isempty(pop) && continue
            topk = sort(pop, by=m -> m.cost)[1:max(1, min(options.topn, length(pop)))]
            append!(best_of_each, copy.(topk))
        end
        replace_from!(best_of_each, options.fraction_replaced)
    end
    options.hof_migration && replace_from!(dominating, options.fraction_replaced_hof)
end

function accept_candidate(engine::EvolutionEngine, parent::Individual, child::Individual,
                          stats::RunningSearchStatistics, temperature::Float64,
                          options::PySROptions)
    isfinite(child.cost) || return false
    prob = 1.0
    if options.annealing
        delta = child.cost - parent.cost
        denom = max(1e-8, temperature * options.alpha)
        prob *= exp(clamp(-delta / denom, -50.0, 50.0))
    end
    if options.use_frequency
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

function write_hof_log(engine::EvolutionEngine, cycle::Int,
                       dominating::Vector{Individual}, options::PySROptions)
    path = options.log_file
    isempty(path) && return
    budget = engine.eval_budget
    progress = (isnothing(budget) || budget == 0) ? 0.0 : engine.eval_count / budget
    try
        open(path, "a") do io
            write(io, "{\"cycle\":", string(cycle))
            write(io, ",\"eval_count\":", string(engine.eval_count))
            write(io, ",\"progress\":", _json_float(progress))
            write(io, ",\"frontier\":[")
            for (i, member) in enumerate(dominating)
                i > 1 && write(io, ",")
                eqn = node_string(member.tree)
                write(io, "{\"complexity\":", string(member.complexity))
                write(io, ",\"loss\":", _json_float(member.loss))
                write(io, ",\"equation\":\"", _json_escape(eqn), "\"}")
            end
            write(io, "]}\n")
        end
    catch err
        @warn "write_hof_log failed" path=path err=err
    end
end

function pysr_mse_loss_function(tree::Node, complexity::Int, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    pred = Vector{Float64}(evaluate_tree(tree, engine.X))
    if length(pred) != length(engine.y) || !all(isfinite.(pred) .& (abs.(pred) .< 1e12))
        return (Inf, Inf)
    end
    loss = sum((engine.y .- pred) .^ 2) / length(engine.y)
    baseline_loss = sum((engine.y .- (sum(engine.y) / length(engine.y))) .^ 2) / length(engine.y)
    !isfinite(baseline_loss) && (baseline_loss = 1.0)
    loss_normalization = baseline_loss >= 0.01 ? baseline_loss : 0.01
    cost = loss / loss_normalization
    isfinite(cost) || return (Inf, Inf)
    return (loss, cost + state.policy_state.options.parsimony * complexity)
end

function pysr_subtree_swap_crossover(parent_a::Individual, parent_b::Individual,
                                     state::EngineState, _config::SkeletonSRConfig)
    for _attempt in 1:10
        t1, t2 = pysr_subtree_crossover(state.engine, parent_a.tree, parent_b.tree)
        valid_tree(state.engine, t1) && valid_tree(state.engine, t2) && return (t1, t2)
    end
    return nothing
end

# ─── PySR policy ─────────────────────────────────────────────────────────────

mutable struct PySRState <: AbstractPolicyState
    options::PySROptions
    best_by_complexity::Dict{Int, Individual}
    archive::Vector{Individual}
    per_population_stats::Vector{RunningSearchStatistics}
    current_temperature::Float64
    counted_population_cycles::Vector{Int}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
    last_logged_cycle::Int
end

function PySRState(cfg::EngineConfig, options::PySROptions)
    n = max(1, cfg.populations)
    return PySRState(
        options,
        Dict{Int, Individual}(),
        Individual[],
        [RunningSearchStatistics(cfg.maxsize) for _ in 1:n],
        1.0,
        zeros(Int, n),
        false,
        zeros(Int, n),
        -2,
    )
end

current_stats(state::EngineState{PySRState}) =
    state.policy_state.per_population_stats[state.current_population]

const PYSR_MUTATION_NAMES = [
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
]

const PYSR_MUTATION_WEIGHTS = Dict{Symbol, Float64}(
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
)

function conditioned_pysr_mutation_weights(
    engine::EvolutionEngine, tree::Node, options::PySROptions
)
    nodes = nodes_with_parent(tree)
    leaves = leaf_nodes(tree)
    n_constants = count(n -> n isa ConstNode, leaves)
    names = copy(PYSR_MUTATION_NAMES)
    w = Dict(name => max(0.0, PYSR_MUTATION_WEIGHTS[name]) for name in names)
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
    !options.should_simplify && (w[:simplify] = 0.0)
    return names, w
end

function sample_pysr_mutation_choice(
    engine::EvolutionEngine, tree::Node, options::PySROptions
)
    names, w = conditioned_pysr_mutation_weights(engine, tree, options)
    weights = [w[n] for n in names]
    total = sum(weights)
    total <= 0 && return :do_nothing
    weights ./= total
    return weighted_choice(engine.rng, names, weights)
end

function pysr_weighted_mutation(
    engine::EvolutionEngine, parent::Individual, options::PySROptions
)
    mutation = sample_pysr_mutation_choice(engine, parent.tree, options)
    for _attempt in 1:10
        proposal = pysr_apply_mutation(engine, parent.tree, mutation, options)
        valid_tree(engine, proposal) && return proposal
    end
    return nothing
end

function pysr_kwargs(; kwargs...)
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

function pysr_selection(population::Population, state::EngineState, config::SkeletonSRConfig)
    stats = current_stats(state)
    idx = tournament_select(
        population, stats, state.policy_state.options, config.engine_config.maxsize, state.engine.rng
    )
    return population[idx]
end

function pysr_survival(population::Population, candidates::Vector{Individual},
                       state::EngineState, _config::SkeletonSRConfig)
    output = copy(population)
    used = Set{Int}()
    for candidate in candidates
        isempty(output) && break
        victim = oldest_survival(output, state.engine.rng, used)
        push!(used, victim)
        output[victim] = candidate
    end
    return output
end

function pysr_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
    return pysr_weighted_mutation(state.engine, parent, state.policy_state.options)
end

function pysr_acceptance(parent::Individual, child::Individual, state::EngineState,
                         _config::SkeletonSRConfig)
    return accept_candidate(
        state.engine,
        parent,
        child,
        current_stats(state),
        state.policy_state.current_temperature,
        state.policy_state.options,
    )
end

function pysr_update_state!(populations::Vector{Population}, state::EngineState{PySRState},
                            config::SkeletonSRConfig)
    policy_state = state.policy_state
    options = policy_state.options
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
        policy_state.archive = calculate_pareto_frontier_from_dict(
            policy_state.best_by_complexity
        )
        for i in archive_pop_indices
            policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
        end
        policy_state.archive_initialized = true
        if !isempty(options.log_file)
            cycle = was_archive_initialized ?
                (state.current_iteration - 1) * length(populations) + (state.current_population - 1) :
                -1
            if cycle > policy_state.last_logged_cycle
                write_hof_log(state.engine, cycle, policy_state.archive, options)
                policy_state.last_logged_cycle = cycle
            end
        end
    end

    cfg = config.engine_config
    stats = current_stats(state)

    normalize!(stats)
    if options.annealing && cfg.ncycles_per_iteration > 1
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
            update_size!(stats, member.complexity)
        end
        move_window!(stats)
        policy_state.counted_population_cycles[pop_idx] = completed
    end
    return nothing
end

function pysr_update_population(policy_state::PySRState, populations::Vector{Population},
                                state::EngineState, _config::SkeletonSRConfig)
    pysr_migration(
        state.engine,
        populations,
        state.current_population,
        policy_state.archive,
        policy_state.options,
    )
    return populations
end

function pysr_cycles_per_population(population::Population, state::EngineState, _config::SkeletonSRConfig)
    return Int(ceil(length(population) / max(1, state.policy_state.options.tournament_selection_n)))
end

function pysr_should_crossover(population::Population, state::EngineState, _config::SkeletonSRConfig)
    return length(population) >= 2 &&
        rand(state.engine.rng) <= state.policy_state.options.crossover_probability
end

pysr_skip_mutation_failures(state::EngineState, _config::SkeletonSRConfig) =
    state.policy_state.options.skip_mutation_failures

function pysr_postprocess_population!(population::Population, state::EngineState, _config::SkeletonSRConfig)
    options = state.policy_state.options
    return optimize_and_simplify!(
        population;
        state=state,
        config=_config,
        should_simplify=options.should_simplify,
        should_optimize_constants=options.should_optimize_constants,
        optimize_probability=options.optimize_probability,
        optimizer_iterations=options.optimizer_iterations,
        optimizer_nrestarts=options.optimizer_nrestarts,
        optimizer_f_calls_limit=options.optimizer_f_calls_limit,
    )
end

function pysr_policy(options::PySROptions)
    return SkeletonSRPolicy(;
        init_state=config -> PySRState(config.engine_config, options),
        loss_function=pysr_mse_loss_function,
        survival=pysr_survival,
        selection=pysr_selection,
        mutation=pysr_mutation,
        acceptance=pysr_acceptance,
        crossover=pysr_subtree_swap_crossover,
        cycles_per_population=pysr_cycles_per_population,
        should_crossover=pysr_should_crossover,
        skip_mutation_failures=pysr_skip_mutation_failures,
        postprocess_population! = pysr_postprocess_population!,
        update_population=pysr_update_population,
        update_state! = pysr_update_state!,
    )
end

function pysr_config(; kwargs...)
    nt = pysr_kwargs(; kwargs...)
    cfg = engine_config_from_namedtuple(nt)
    options = PySROptions(;
        parsimony=nt.parsimony,
        tournament_selection_n=nt.tournament_selection_n,
        tournament_selection_p=nt.tournament_selection_p,
        crossover_probability=nt.crossover_probability,
        skip_mutation_failures=nt.skip_mutation_failures,
        use_frequency=nt.use_frequency,
        use_frequency_in_tournament=nt.use_frequency_in_tournament,
        adaptive_parsimony_scaling=nt.adaptive_parsimony_scaling,
        annealing=nt.annealing,
        alpha=nt.alpha,
        perturbation_factor=nt.perturbation_factor,
        probability_negate_constant=nt.probability_negate_constant,
        migration=nt.migration,
        hof_migration=nt.hof_migration,
        fraction_replaced=nt.fraction_replaced,
        fraction_replaced_hof=nt.fraction_replaced_hof,
        topn=nt.topn,
        should_optimize_constants=nt.should_optimize_constants,
        optimize_probability=nt.optimize_probability,
        optimizer_iterations=nt.optimizer_iterations,
        optimizer_nrestarts=nt.optimizer_nrestarts,
        optimizer_f_calls_limit=nt.optimizer_f_calls_limit,
        should_simplify=nt.should_simplify,
        log_file=nt.log_file,
    )
    return SkeletonSRConfig(; engine_config=cfg, policy=pysr_policy(options))
end

fit_pysr_sr(args...; kwargs...) =
    fit_skeleton_sr(args...; config=pysr_config(; kwargs...))

end
