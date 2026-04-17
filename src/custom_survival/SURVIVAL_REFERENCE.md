# PySR/SymbolicRegression.jl Custom Survival Reference

## Function Signature

```julia
function your_survival_name(
    pop::Population{T,L,N},
    options::AbstractOptions;
    exclude_indices::Vector{Int}=Int[],
)::Int where {T,L,N}
    # survival logic - return index of member to replace
    return idx
end
```

The survival function decides **which population member gets replaced** when a new offspring is created. It returns the **index** (1-based) of the member to be replaced.

---

## Available API

```julia
# Imports available in custom survival functions
using ..CoreModule: AbstractOptions, DATA_TYPE, LOSS_TYPE
using ..PopulationModule: Population
using ..PopMemberModule: PopMember
using ..ComplexityModule: compute_complexity
using ..UtilsModule: argmin_fast
```

### Population

```julia
pop.members      # Vector{PopMember{T,L,N}} - all population members
pop.n             # Int - number of members in population
```

### PopMember Fields

```julia
member.tree       # AbstractExpression - the symbolic expression tree
member.cost       # L - cost (includes complexity penalty, normalization)
member.loss       # L - raw loss on training data
member.birth      # Int - birth order (monotonically increasing counter)
member.ref        # Int - unique reference ID
member.parent     # Int - parent reference ID
```

Note: Access complexity via `compute_complexity(member, options)`, not `member.complexity`.

### Options Access

```julia
options.maxsize                     # Int - maximum expression size
options.tournament_selection_n      # Int - tournament size
options.population_size             # Int - population size
options.adaptive_parsimony_scaling  # Float - parsimony pressure scaling
options.use_frequency_in_tournament # Bool - whether to use frequency-based parsimony
```

### Utility Functions

```julia
argmin_fast(v)                      # Fast argmin for vectors
compute_complexity(member, options) # Compute complexity of a member's expression
```

---

## Default Implementation

The default survival function replaces the **oldest** member (age-regularized evolution):

```julia
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
```

---

## Key Patterns

### Handling `exclude_indices`

When replacing two members at once (crossover), the second call passes the first replacement index in `exclude_indices` to prevent replacing the same slot twice:

```julia
oldest1 = your_survival(pop, options)
oldest2 = your_survival(pop, options; exclude_indices=[oldest1])
```

Your function **must** respect `exclude_indices` by never returning an index that appears in it.

### Pattern: Exclude and Compute

```julia
function my_survival(pop, options; exclude_indices=Int[])::Int
    best_idx = -1
    best_val = typemin(Float64)  # or typemax for minimization

    for i in 1:(pop.n)
        i in exclude_indices && continue
        val = some_criterion(pop.members[i])
        if val > best_val  # or < for minimization
            best_val = val
            best_idx = i
        end
    end

    return best_idx
end
```

### Type Assertions and Bounds Checking

The dispatch function asserts `1 <= idx <= pop.n`, so ensure your function always returns a valid index.
