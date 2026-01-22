module CustomMutationsModule

using Random: AbstractRNG, default_rng
using TOML: TOML
using DynamicExpressions:
    AbstractExpressionNode,
    AbstractExpression,
    NodeSampler,
    constructorof,
    set_node!,
    get_contents,
    with_contents,
    get_child,
    set_child!

export load_custom_mutation_config,
    get_custom_mutation_weights,
    get_builtin_weight_overrides,
    apply_custom_mutation,
    list_enabled_custom_mutations,
    reload_custom_mutations!,
    setup_custom_mutations!,
    # New dynamic loading exports
    load_mutation_from_string!,
    load_mutation_from_file!,
    register_mutation!,
    clear_dynamic_mutations!,
    list_available_mutations

# Path to the config file
const CONFIG_DIR = joinpath(@__DIR__, "custom_mutations")
const CONFIG_PATH = joinpath(CONFIG_DIR, "config.toml")

# Global registry for loaded mutations
const MUTATION_REGISTRY = Dict{Symbol,Function}()
const MUTATION_WEIGHTS = Dict{Symbol,Float64}()
const BUILTIN_OVERRIDES = Dict{Symbol,Float64}()
const INITIALIZED = Ref(false)

# Registry for dynamically loaded mutations (separate from static ones)
const DYNAMIC_MUTATIONS = Dict{Symbol,Function}()
# Weights for dynamically loaded mutations (preserved across reload)
const DYNAMIC_WEIGHTS = Dict{Symbol,Float64}()

"""
    load_custom_mutation_config()

Load the custom mutations configuration from config.toml.
Returns a NamedTuple with (custom_mutations, builtin_weights).
"""
function load_custom_mutation_config()
    if !isfile(CONFIG_PATH)
        @warn "Custom mutations config not found at $CONFIG_PATH"
        return (custom_mutations=Dict{String,Float64}(), builtin_weights=Dict{String,Float64}())
    end

    config = TOML.parsefile(CONFIG_PATH)

    custom_mutations = get(config, "custom_mutations", Dict{String,Any}())
    builtin_weights = get(config, "builtin_weights", Dict{String,Any}())

    return (
        custom_mutations=Dict{String,Float64}(k => Float64(v) for (k, v) in custom_mutations),
        builtin_weights=Dict{String,Float64}(k => Float64(v) for (k, v) in builtin_weights),
    )
end

# ============================================================================
# STATIC MUTATIONS (compiled at package load time)
# These are always available and don't require dynamic loading
# ============================================================================

# Include the example mutation
include("custom_mutations/add_constant_offset.jl")

# Map of statically available mutations (compiled into package)
const STATIC_MUTATIONS = Dict{Symbol,Function}(
    :add_constant_offset => add_constant_offset,
)

# ============================================================================
# DYNAMIC MUTATION LOADING
# Load mutation code at runtime without recompilation
# ============================================================================

"""
    load_mutation_from_string!(name::Symbol, code::String) -> Function

Load a mutation function from Julia code string at runtime.
The code should define a function with signature:
    function mutation_name(tree, options, nfeatures, rng) -> tree

Returns the loaded function.

# Example
```julia
code = \"\"\"
function my_mutation(tree::N, options, nfeatures::Int, rng::AbstractRNG) where {T, N<:AbstractExpressionNode{T}}
    # ... mutation logic ...
    return tree
end
\"\"\"
load_mutation_from_string!(:my_mutation, code)
```
"""
function load_mutation_from_string!(name::Symbol, code::String)
    # Parse and evaluate the code in this module's scope
    # This gives the mutation access to all the imports (AbstractExpressionNode, etc.)
    try
        expr = Meta.parse("begin\n$code\nend")
        Base.eval(@__MODULE__, expr)

        # The function should now be defined in this module
        func = Base.eval(@__MODULE__, name)

        if !isa(func, Function)
            error("Code did not define a function named '$name'")
        end

        # Register in dynamic mutations
        DYNAMIC_MUTATIONS[name] = func

        # Also add to main registry so it can be used
        MUTATION_REGISTRY[name] = func

        return func
    catch e
        @error "Failed to load mutation '$name' from code" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    load_mutation_from_file!(name::Symbol, filepath::String) -> Function

Load a mutation function from a Julia file at runtime.
The file should define a function matching the given name.

# Example
```julia
load_mutation_from_file!(:my_mutation, "/path/to/my_mutation.jl")
```
"""
function load_mutation_from_file!(name::Symbol, filepath::String)
    if !isfile(filepath)
        error("Mutation file not found: $filepath")
    end
    code = read(filepath, String)
    return load_mutation_from_string!(name, code)
end

"""
    register_mutation!(name::Symbol, func::Function)

Register an already-defined function as a custom mutation.
Useful if you've defined the function elsewhere and want to use it as a mutation.
"""
function register_mutation!(name::Symbol, func::Function)
    DYNAMIC_MUTATIONS[name] = func
    MUTATION_REGISTRY[name] = func
    return func
end

"""
    clear_dynamic_mutations!()

Clear all dynamically loaded mutations, keeping only static ones.
Useful for resetting state between different evaluation runs.
"""
function clear_dynamic_mutations!()
    # Remove dynamic mutations from main registry
    for name in keys(DYNAMIC_MUTATIONS)
        delete!(MUTATION_REGISTRY, name)
        delete!(MUTATION_WEIGHTS, name)
    end
    empty!(DYNAMIC_MUTATIONS)
    empty!(DYNAMIC_WEIGHTS)
    return nothing
end

"""
    list_available_mutations() -> Vector{Symbol}

List all available mutations (both static and dynamic).
"""
function list_available_mutations()
    all_mutations = Set{Symbol}()
    union!(all_mutations, keys(STATIC_MUTATIONS))
    union!(all_mutations, keys(DYNAMIC_MUTATIONS))
    return collect(all_mutations)
end

# ============================================================================
# MUTATION REGISTRY MANAGEMENT
# ============================================================================

"""
    reload_custom_mutations!()

Reload custom mutation weights from config.
Also ensures static mutations are in the registry.
"""
function reload_custom_mutations!()
    empty!(MUTATION_REGISTRY)
    empty!(MUTATION_WEIGHTS)
    empty!(BUILTIN_OVERRIDES)

    # Always add static mutations to registry
    for (name, func) in STATIC_MUTATIONS
        MUTATION_REGISTRY[name] = func
    end

    # Re-add any dynamic mutations
    for (name, func) in DYNAMIC_MUTATIONS
        MUTATION_REGISTRY[name] = func
    end

    config = load_custom_mutation_config()

    # Load builtin overrides
    for (name, weight) in config.builtin_weights
        BUILTIN_OVERRIDES[Symbol(name)] = weight
    end

    # Set weights for mutations that are both available and configured in config.toml
    for (name, weight) in config.custom_mutations
        sym = Symbol(name)
        if weight > 0 && haskey(MUTATION_REGISTRY, sym)
            MUTATION_WEIGHTS[sym] = weight
        elseif weight > 0
            # Not an error for dynamic loading - mutation might be loaded later
            @debug "Custom mutation '$name' configured but not yet loaded"
        end
    end

    # Restore weights for dynamically loaded mutations (these override config.toml)
    for (name, weight) in DYNAMIC_WEIGHTS
        if haskey(MUTATION_REGISTRY, name)
            MUTATION_WEIGHTS[name] = weight
        end
    end

    INITIALIZED[] = true
    return nothing
end

"""
    get_custom_mutation_weights()

Get a Dict of enabled custom mutation names to their weights.
"""
function get_custom_mutation_weights()
    if !INITIALIZED[]
        reload_custom_mutations!()
    end
    return copy(MUTATION_WEIGHTS)
end

"""
    get_builtin_weight_overrides()

Get a Dict of built-in mutation weight overrides from config.
"""
function get_builtin_weight_overrides()
    if !INITIALIZED[]
        reload_custom_mutations!()
    end
    return copy(BUILTIN_OVERRIDES)
end

"""
    list_enabled_custom_mutations()

Return a list of enabled custom mutation names (those with weight > 0).
"""
function list_enabled_custom_mutations()
    if !INITIALIZED[]
        reload_custom_mutations!()
    end
    return collect(keys(MUTATION_WEIGHTS))
end

"""
    apply_custom_mutation(name::Symbol, tree, options, nfeatures, rng)

Apply a custom mutation by name to the given tree.
"""
function apply_custom_mutation(
    name::Symbol,
    tree::AbstractExpressionNode,
    options,
    nfeatures::Int,
    rng::AbstractRNG=default_rng(),
)
    if !INITIALIZED[]
        reload_custom_mutations!()
    end

    if !haskey(MUTATION_REGISTRY, name)
        @warn "Custom mutation '$name' not found in registry"
        return tree
    end

    func = MUTATION_REGISTRY[name]
    return func(tree, options, nfeatures, rng)
end

"""
    apply_custom_mutation(name::Symbol, ex::AbstractExpression, options, nfeatures, rng)

Apply a custom mutation to an AbstractExpression wrapper.
"""
function apply_custom_mutation(
    name::Symbol,
    ex::AbstractExpression,
    options,
    nfeatures::Int,
    rng::AbstractRNG=default_rng(),
)
    tree = get_contents(ex)
    new_tree = apply_custom_mutation(name, tree, options, nfeatures, rng)
    return with_contents(ex, new_tree)
end

"""
    setup_custom_mutations!(custom_mutation_names::Dict{Symbol,Symbol}, mutation_weights)

Set up custom mutations by:
1. Loading custom mutations from config
2. Populating the custom_mutation_names mapping (custom_mutation_1 => :actual_name)
3. Setting the weights in mutation_weights struct

Call this during initialization to wire up custom mutations.
Returns the list of enabled custom mutation names.
"""
function setup_custom_mutations!(
    custom_mutation_names::Dict{Symbol,Symbol},
    mutation_weights,
)
    if !INITIALIZED[]
        reload_custom_mutations!()
    end

    # Get enabled mutations
    enabled = collect(keys(MUTATION_WEIGHTS))

    # Assign to slots
    slot_names = [:custom_mutation_1, :custom_mutation_2, :custom_mutation_3,
                  :custom_mutation_4, :custom_mutation_5]

    for slot in slot_names
        custom_mutation_names[slot] = :none
    end

    for (i, name) in enumerate(enabled)
        if i > 5
            @warn "More than 5 custom mutations enabled, only first 5 will be used"
            break
        end
        slot = slot_names[i]
        custom_mutation_names[slot] = name

        # Set the weight in mutation_weights
        weight = MUTATION_WEIGHTS[name]
        setfield!(mutation_weights, slot, weight)
    end

    # Apply builtin weight overrides if mutation_weights is MutationWeights
    for (name, weight) in BUILTIN_OVERRIDES
        if hasfield(typeof(mutation_weights), name)
            setfield!(mutation_weights, name, weight)
        end
    end

    return enabled
end

"""
    setup_dynamic_mutation!(name::Symbol, code::String, weight::Float64, slot::Int=1)

Convenience function to load a dynamic mutation and set its weight in one call.
Slot should be 1-5, corresponding to custom_mutation_1 through custom_mutation_5.

# Example
```julia
setup_dynamic_mutation!(:my_mutation, code_string, 0.5, 1)
```
"""
function setup_dynamic_mutation!(name::Symbol, code::String, weight::Float64, slot::Int=1)
    if slot < 1 || slot > 5
        error("Slot must be between 1 and 5, got $slot")
    end

    # Load the mutation
    load_mutation_from_string!(name, code)

    # Set its weight (both in current weights and persistent dynamic weights)
    MUTATION_WEIGHTS[name] = weight
    DYNAMIC_WEIGHTS[name] = weight

    return name
end

end # module
