module CustomSelectionModule

using StatsBase: StatsBase
using ..CoreModule: AbstractOptions, DATA_TYPE, LOSS_TYPE
using ..PopulationModule: Population
using ..PopMemberModule: PopMember
using ..ComplexityModule: compute_complexity
using ..AdaptiveParsimonyModule: RunningSearchStatistics
using ..UtilsModule: argmin_fast, bottomk_fast

export apply_custom_selection,
    load_selection_from_string!,
    load_selection_from_file!,
    clear_dynamic_selections!,
    list_available_selections,
    reload_custom_selections!

# Active custom selection function (nothing = use default)
const ACTIVE_CUSTOM_SELECTION = Ref{Union{Nothing,Function}}(nothing)

# Registry for dynamically loaded selections
const DYNAMIC_SELECTIONS = Dict{Symbol,Function}()

"""
    default_selection(pop, running_search_statistics, options) -> PopMember

Default selection function: tournament selection with adaptive parsimony.
This is a self-contained reimplementation of `best_of_sample` from Population.jl.

Returns a copy of the selected population member.
"""
function default_selection(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    # Sample tournament_selection_n members
    sample = StatsBase.sample(pop.members, options.tournament_selection_n; replace=false)
    n = length(sample)
    p = options.tournament_selection_p

    # Compute adjusted costs (frequency-based parsimony)
    adjusted_costs = Vector{L}(undef, n)
    if options.use_frequency_in_tournament
        adaptive_parsimony_scaling = L(options.adaptive_parsimony_scaling)
        for i in 1:n
            member = sample[i]
            size = compute_complexity(member, options)
            frequency = if (0 < size <= options.maxsize)
                L(running_search_statistics.normalized_frequencies[size])
            else
                L(0)
            end
            adjusted_costs[i] = member.cost * exp(adaptive_parsimony_scaling * frequency)
        end
    else
        for i in 1:n
            adjusted_costs[i] = sample[i].cost
        end
    end

    # Tournament selection
    chosen_idx = if p == 1.0
        argmin_fast(adjusted_costs)
    else
        k = collect(0:(n - 1))
        prob_each = p * ((1 - p) .^ k)
        weights = StatsBase.Weights(prob_each, sum(prob_each))
        tournament_winner = StatsBase.sample(weights)
        if tournament_winner == 1
            argmin_fast(adjusted_costs)
        else
            bottomk_fast(adjusted_costs, tournament_winner)[2][end]
        end
    end
    return copy(sample[chosen_idx])
end

"""
    apply_custom_selection(pop, running_search_statistics, options) -> PopMember

Dispatch to either the custom selection function or the default.
Returns a copy of the selected population member.
"""
function apply_custom_selection(
    pop::Population{T,L,N},
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
)::PopMember{T,L,N} where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    func = ACTIVE_CUSTOM_SELECTION[]
    if func === nothing
        return default_selection(pop, running_search_statistics, options)
    end
    return copy(func(pop, running_search_statistics, options))
end

"""
    load_selection_from_string!(name::Symbol, code::String) -> Function

Load a selection function from Julia code string at runtime.
The code should define a function with signature:
    function selection_name(pop, running_search_statistics, options) -> PopMember

Returns the loaded function.
"""
function load_selection_from_string!(name::Symbol, code::String)
    try
        expr = Meta.parse("begin\n$code\nend")
        Base.eval(@__MODULE__, expr)

        func = Base.eval(@__MODULE__, name)

        if !isa(func, Function)
            error("Code did not define a function named '$name'")
        end

        DYNAMIC_SELECTIONS[name] = func
        ACTIVE_CUSTOM_SELECTION[] = func

        return func
    catch e
        @error "Failed to load selection '$name' from code" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    load_selection_from_file!(name::Symbol, filepath::String) -> Function

Load a selection function from a Julia file at runtime.
"""
function load_selection_from_file!(name::Symbol, filepath::String)
    if !isfile(filepath)
        error("Selection file not found: $filepath")
    end
    code = read(filepath, String)
    return load_selection_from_string!(name, code)
end

"""
    clear_dynamic_selections!()

Clear all dynamically loaded selections, resetting to default behavior.
"""
function clear_dynamic_selections!()
    empty!(DYNAMIC_SELECTIONS)
    ACTIVE_CUSTOM_SELECTION[] = nothing
    return nothing
end

"""
    list_available_selections() -> Vector{Symbol}

List all available selection functions (dynamically loaded).
"""
function list_available_selections()
    return collect(keys(DYNAMIC_SELECTIONS))
end

"""
    reload_custom_selections!()

Reload custom selections. Re-adds dynamic selections to registry.
"""
function reload_custom_selections!()
    if !isempty(DYNAMIC_SELECTIONS)
        ACTIVE_CUSTOM_SELECTION[] = last(values(DYNAMIC_SELECTIONS))
    else
        ACTIVE_CUSTOM_SELECTION[] = nothing
    end
    return nothing
end

end # module
