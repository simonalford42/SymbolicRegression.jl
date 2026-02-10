module CustomSurvivalModule

using ..CoreModule: AbstractOptions, DATA_TYPE, LOSS_TYPE
using ..PopulationModule: Population
using ..PopMemberModule: PopMember
using ..ComplexityModule: compute_complexity
using ..UtilsModule: argmin_fast

export apply_custom_survival,
    load_survival_from_string!,
    load_survival_from_file!,
    clear_dynamic_survivals!,
    list_available_survivals,
    reload_custom_survivals!

# Active custom survival function (nothing = use default)
const ACTIVE_CUSTOM_SURVIVAL = Ref{Union{Nothing,Function}}(nothing)

# Registry for dynamically loaded survivals
const DYNAMIC_SURVIVALS = Dict{Symbol,Function}()

"""
    default_survival(pop, options; exclude_indices) -> Int

Default survival function: replace the oldest member of the population.
This is the age-regularized evolution strategy from the original SymbolicRegression.jl.

Returns the index of the member to be replaced.
"""
function default_survival(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    BT = typeof(first(pop.members).birth)
    births = [(i in exclude_indices) ? typemax(BT) : pop.members[i].birth
              for i in 1:(pop.n)]
    return argmin_fast(births)
end

"""
    apply_custom_survival(pop, options; exclude_indices) -> Int

Dispatch to either the custom survival function or the default.
Returns the index of the population member to replace.
"""
function apply_custom_survival(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T<:DATA_TYPE,L<:LOSS_TYPE,N}
    func = ACTIVE_CUSTOM_SURVIVAL[]
    if func === nothing
        return default_survival(pop, options; exclude_indices)
    end
    idx = func(pop, options; exclude_indices)::Int
    @assert 1 <= idx <= pop.n "Custom survival returned index $idx, must be in 1:$(pop.n)"
    return idx
end

"""
    load_survival_from_string!(name::Symbol, code::String) -> Function

Load a survival function from Julia code string at runtime.
The code should define a function with signature:
    function survival_name(pop, options; exclude_indices=Int[]) -> Int

Returns the loaded function.
"""
function load_survival_from_string!(name::Symbol, code::String)
    try
        expr = Meta.parse("begin\n$code\nend")
        Base.eval(@__MODULE__, expr)

        func = Base.eval(@__MODULE__, name)

        if !isa(func, Function)
            error("Code did not define a function named '$name'")
        end

        DYNAMIC_SURVIVALS[name] = func
        ACTIVE_CUSTOM_SURVIVAL[] = func

        return func
    catch e
        @error "Failed to load survival '$name' from code" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    load_survival_from_file!(name::Symbol, filepath::String) -> Function

Load a survival function from a Julia file at runtime.
"""
function load_survival_from_file!(name::Symbol, filepath::String)
    if !isfile(filepath)
        error("Survival file not found: $filepath")
    end
    code = read(filepath, String)
    return load_survival_from_string!(name, code)
end

"""
    clear_dynamic_survivals!()

Clear all dynamically loaded survivals, resetting to default behavior.
"""
function clear_dynamic_survivals!()
    empty!(DYNAMIC_SURVIVALS)
    ACTIVE_CUSTOM_SURVIVAL[] = nothing
    return nothing
end

"""
    list_available_survivals() -> Vector{Symbol}

List all available survival functions (dynamically loaded).
"""
function list_available_survivals()
    return collect(keys(DYNAMIC_SURVIVALS))
end

"""
    reload_custom_survivals!()

Reload custom survivals. Re-adds dynamic survivals to registry.
"""
function reload_custom_survivals!()
    if !isempty(DYNAMIC_SURVIVALS)
        # Use the last loaded survival as active
        ACTIVE_CUSTOM_SURVIVAL[] = last(values(DYNAMIC_SURVIVALS))
    else
        ACTIVE_CUSTOM_SURVIVAL[] = nothing
    end
    return nothing
end

end # module
