
"""
    struct ErpenbeckThoss{T<:AbstractFloat} <: QuantumModel

1D two-state Quantum system capable of modelling a molecule adsorbed on a metal surface
or a single-molecule junction.

In the two references, all of the parameters are identical except for the particle mass `m`
and the vertical shift `c` applied to the ϵ₀ state.
Both references modify the shift to ensure the quantum ground-state has an energy of 0 eV.
Note that the mass `m` is specified in atomic mass units (amu) **not** atomic units.
If a value for the vertical offset `c` is not explicitly provided whne constructing the model,
it is automatically determined in the constructor from the Morse potential zero-point energy.

# References

- PHYSICAL REVIEW B 97, 235452 (2018)
- J. Chem. Phys. 151, 191101 (2019)
"""
struct OptimErpenbeckThoss{T<:AbstractFloat} <: QuantumModel
    Γ::T
    morse::ClassicalModels.Morse{T}
    D₁::T
    D₂::T
    x₀′::T
    a′::T
    c::T
    V∞::T
    q::T
    ã::T
    x̃::T
    V̄ₖ::T
end

function OptimErpenbeckThoss(
    Γ,
    m,
    Dₑ,
    x₀,
    a,
    D₁,
    D₂,
    a′,
    V∞,
    q,
    ã,
    x̃;

    x₀′ = x₀,
    V̄ₖ  = sqrt(austrip(Γ)/2π),
    c   = nothing
)
    morse = ClassicalModels.Morse(;Dₑ, x₀, a, m)
    if isnothing(c)
        c = -NQCModels.ClassicalModels.eigenenergy(morse, 0) # Set c to offset zero-point energy
    end
    
    return OptimErpenbeckThoss( austrip(Γ),
                                morse,
                                austrip(D₁),
                                austrip(D₂),
                                austrip(x₀′),
                                austrip(a′),
                                austrip(c),
                                austrip(V∞),
                                q,
                                austrip(ã),
                                austrip(x̃),
                                austrip(V̄ₖ) )
end

function NQCModels.potential(model::OptimErpenbeckThoss, R::AbstractMatrix)
    (; c, D₁, D₂, x₀′, a′, V∞) = model  # OPTIMIZE all these
    (; q, ã, x̃, V̄ₖ) = model             # OPTIMIZE all these
    (; Dₑ, x₀, a) = model.morse         # OPTIMIZE all these

    V11 = Dₑ * (exp(-a*(r-x₀)) - 1)^2 + c #NQCModels.potential(morse, R) + c
    V22 = D₁*exp(-2a′*(R[1]-x₀′)) - D₂*exp(-a′*(R[1]-x₀′)) + V∞
    V12 = V̄ₖ * ((1-q)/2*(1 - tanh((R[1]-x̃)/ã)) + q)
    
    return [V11 V12; V12 V22]
end

function NQCModels.potential!(model::OptimErpenbeckThoss, V::AbstractMatrix, R::AbstractMatrix)
    (; c, D₁, D₂, x₀′, a′, V∞) = model # OPTIMIZE all these
    (; q, ã, x̃, V̄ₖ) = model # OPTIMIZE all these
    (; Dₑ, x₀, a) = model.morse # OPTIMIZE all these

    V11 = Dₑ * (exp(-a*(r-x₀)) - 1)^2 + c #NQCModels.potential(morse, R) + c
    V22 = D₁*exp(-2a′*(R[1]-x₀′)) - D₂*exp(-a′*(R[1]-x₀′)) + V∞
    V12 = V̄ₖ * ((1-q)/2*(1 - tanh((R[1]-x̃)/ã)) + q)

    V[1,1] = V11
    V[2,2] = V22
    V[1,2] = V12
    V[2,1] = V12

end

function NQCModels.derivative(model::OptimErpenbeckThoss, R::AbstractMatrix)
    (;morse, D₁, D₂, x₀′, a′) = model
    (;q, ã, x̃, V̄ₖ) = model

    D11 = NQCModels.derivative(morse, R)
    D22 = -2a′*D₁*exp(-2a′*(R[1]-x₀′)) + a′*D₂*exp(-a′*(R[1]-x₀′))
    D12 = -V̄ₖ * (1-q)/2 * sech((R[1]-x̃)/ã)^2 / ã
    return Hermitian([D11 D12; D12 D22])
end

function NQCModels.derivative!(model::OptimErpenbeckThoss, D::Matrix{<:Hermitian}, R::AbstractMatrix)
    NQCModels.derivative!(model, D[1,1], R)
end

function NQCModels.derivative!(model::OptimErpenbeckThoss, D::Hermitian, R::AbstractMatrix)
    (;morse, D₁, D₂, x₀′, a′) = model

    (;q, ã, x̃, V̄ₖ) = model

    D11 = NQCModels.derivative(morse, R)
    D22 = -2a′*D₁*exp(-2a′*(R[1]-x₀′)) + a′*D₂*exp(-a′*(R[1]-x₀′))
    D12 = -V̄ₖ * (1-q)/2 * sech((R[1]-x̃)/ã)^2 / ã
    
    D.data[1,1] = D11
    D.data[2,2] = D22
    D.data[1,2] = D12

    return nothing
end

NQCModels.nstates(::OptimErpenbeckThoss) = 2
NQCModels.ndofs(::OptimErpenbeckThoss) = 1
