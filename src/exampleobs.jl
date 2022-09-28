using ADMPS: norm_FL,norm_FR
using LinearAlgebra: norm
using OMEinsum
using TeneT: ALCtoAC
using ADMPS: norm_FL,norm_FR

export observable, magofβ, updown_overlap

"""
    observable(env, model::MT, type)

return the `type` observable of the `model`. Requires that `type` tensor defined in model_tensor(model, Val(:type)).
"""
function observable(env, model::MT, ::Val{:Z}) where {MT <: HamiltonianModel}
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    atype = _arraytype(ALu)
    M   = atype(model_tensor(model, Val(:bulk)))
    χ,D,Ni,Nj = size(ALu)[[1,2,4,5]]
    
    z_tol = 1
    ACu = ALCtoAC(ALu, Cu)

    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        jr = j + 1 - Nj * (j==Nj)
        z = ein"(((adf,abc),dgeb),ceh),fgh ->"(FLu[:,:,:,i,j],ACu[:,:,:,i,j],M[:,:,:,:,i,j],FRu[:,:,:,i,j],conj(ACu[:,:,:,ir,j]))
        λ = ein"(acd,ab),(bce,de) ->"(FLu[:,:,:,i,jr],Cu[:,:,i,j],FRu[:,:,:,i,j],conj(Cu[:,:,ir,j]))
        z_tol *= Array(z)[]/Array(λ)[]
    end
    return z_tol^(1/Ni/Nj)
end

"""
    residual entropy
"""
function observable(env, model::MT, ::Val{:S}) where {MT <: HamiltonianModel}
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    atype = _arraytype(ALu)
    M   = atype(model_tensor(model, Val(:Sbulk)))
    χ,D,Ni,Nj = size(ALu)[[1,2,4,5]]
    
    z_tol = 1
    ACu = ALCtoAC(ALu, Cu)

    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        jr = j + 1 - Nj * (j==Nj)
        z = ein"(((adf,abc),dgeb),ceh),fgh ->"(FLu[:,:,:,i,j],ACu[:,:,:,i,j],M[:,:,:,:,i,j],FRu[:,:,:,i,j],conj(ACu[:,:,:,ir,j]))
        λ = ein"(acd,ab),(bce,de) ->"(FLu[:,:,:,i,jr],Cu[:,:,i,j],FRu[:,:,:,i,j],conj(Cu[:,:,ir,j]))
        z_tol *= Array(z)[]/Array(λ)[]
    end
    return log(z_tol^(1/Ni/Nj))
end

function observable(env, model::MT, type) where {MT <: HamiltonianModel}
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    χ,D,Ni,Nj = size(ALu)[[1,2,4,5]]
    atype = _arraytype(ALu)
    M     = atype(model_tensor(model, Val(:bulk)))
    M_obs = atype(model_tensor(model, type      ))
    obs_tol = 0
    ACu = ALCtoAC(ALu, Cu)
    ACd = ALCtoAC(ALd, Cd)

    for j = 1:Nj,i = 1:Ni
        ir = Ni + 1 - i
        obs = ein"(((adf,abc),dgeb),fgh),ceh -> "(FL[:,:,:,i,j],ACu[:,:,:,i,j],M_obs[:,:,:,:,i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,j])
        λ = ein"(((adf,abc),dgeb),fgh),ceh -> "(FL[:,:,:,i,j],ACu[:,:,:,i,j],M[:,:,:,:,i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,j])
        obs_tol += Array(obs)[]/Array(λ)[]
    end
    if type == Val(:mag)
        obs_tol = abs(obs_tol)
    end
    return obs_tol/Ni/Nj
end

"""
    magofβ(::Ising,β)
return the analytical result for the magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofβ(model::Ising) = model.β > isingβc ? (1-sinh(2*model.β)^-4)^(1/8) : 0.

function updown_overlap(env)
    _, ALu, Cu, ARu, ALd, Cd, ARd, _ = env
    ACu = ALCtoAC(ALu, Cu)
    ACd = ALCtoAC(ALd, Cd)
    _, FLud_n = norm_FL(ALu[:,:,:,1,1], ALd[:,:,:,1,1])
    _, FRud_n = norm_FR(ARu[:,:,:,1,1], ARd[:,:,:,1,1])

    norm(ein"(ad,acb),(dce,be) ->"(FLud_n,ACu[:,:,:,1,1],ACd[:,:,:,1,1],FRud_n)[]/(ein"(ac,ab),(cd,bd) ->"(FLud_n,Cu[:,:,1,1],Cd[:,:,1,1],FRud_n)[]))
end