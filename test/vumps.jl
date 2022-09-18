using Ising_Triangle
using LinearAlgebra
using Random
using TeneT
using Test
using CUDA
using Zygote
using ADMPS: norm_FL,norm_FR
using TeneT: ALCtoAC
using OMEinsum

@testset "$(Ni)x$(Nj) ising forward with $atype" for Ni = [1], Nj = [1], atype = [Array]
    Random.seed!(100)
    β = 0.5
    model = Ising(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:bulk)))
    env = obs_env(M; χ = 10, maxiter = 10, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = false, verbose = true, savefile = false
        )
    @test observable(env, model, Val(:Z)     ) ≈ 2.789305993957602
    @test observable(env, model, Val(:mag)   ) ≈ magofβ(model) 
    @test observable(env, model, Val(:energy)) ≈ -1.745564581767667
end

@testset "$(Ni)x$(Nj) Ising_Triangle_bad forward with $atype $updown updown" for Ni = [1], Nj = [1], updown = [false, true], atype = [Array]
    Random.seed!(100)
    β = 1
    model = Ising_Triangle_bad(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:bulk)))
    env = obs_env(M; χ = 20, maxiter = 10, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = updown, verbose = true, savefile = false
        )

        M, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    ACu = ALCtoAC(ALu, Cu)
    ACd = ALCtoAC(ALd, Cd)
    _, FLud_n = norm_FL(ALu[:,:,:,1,1], ALd[:,:,:,1,1])
    _, FRud_n = norm_FR(ARu[:,:,:,1,1], ARd[:,:,:,1,1])

    overlap = norm(ein"(ad,acb),(dce,be) ->"(FLud_n,ACu[:,:,:,1,1],ACd[:,:,:,1,1],FRud_n)[]/(ein"(ac,ab),(cd,bd) ->"(FLud_n,Cu[:,:,1,1],Cd[:,:,1,1],FRud_n)[]))
    @show overlap
    @show observable(env, model, Val(:Z)     )
    @show observable(env, model, Val(:energy)) 
end

@testset "$(Ni)x$(Nj) Ising_Triangle_good forward with $atype $updown updown" for Ni = [1], Nj = [1], updown = [false, true], atype = [Array]
    Random.seed!(100)
    β = 1
    model = Ising_Triangle_good(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:bulk)))
    env = obs_env(M; χ = 20, maxiter = 10, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = updown, verbose = true, savefile = false
        )
    M, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    ACu = ALCtoAC(ALu, Cu)
    ACd = ALCtoAC(ALd, Cd)
    _, FLud_n = norm_FL(ALu[:,:,:,1,1], ALd[:,:,:,1,1])
    _, FRud_n = norm_FR(ARu[:,:,:,1,1], ARd[:,:,:,1,1])

    overlap = norm(ein"(ad,acb),(dce,be) ->"(FLud_n,ACu[:,:,:,1,1],ACd[:,:,:,1,1],FRud_n)[]/(ein"(ac,ab),(cd,bd) ->"(FLud_n,Cu[:,:,1,1],Cd[:,:,1,1],FRud_n)[]))
    @show overlap
    @show observable(env, model, Val(:Z)     )
    @show observable(env, model, Val(:energy)) 
end