using Ising_Triangle
using LinearAlgebra
using Random
using TeneT
using Test
using CUDA
using Zygote

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

@testset "$(Ni)x$(Nj) Ising_Triangle_bad forward with $atype $updown updown" for Ni = [1], Nj = [1], updown = [false,true], atype = [Array]
    Random.seed!(100)
    β = 1
    model = Ising_Triangle_bad(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:bulk)))
    env = obs_env(M; χ = 20, maxiter = 10, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = updown, verbose = true, savefile = false
        )
    @show observable(env, model, Val(:Z)     )
    @show observable(env, model, Val(:energy)) 
end

@testset "$(Ni)x$(Nj) Ising_Triangle_good forward with $atype $updown updown" for Ni = [1], Nj = [1], updown = [false,true], atype = [Array]
    Random.seed!(100)
    β = 1
    model = Ising_Triangle_good(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:bulk)))
    env = obs_env(M; χ = 20, maxiter = 10, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = false, verbose = true, savefile = false
        )
    @show observable(env, model, Val(:Z)     )
    @show observable(env, model, Val(:energy)) 
end