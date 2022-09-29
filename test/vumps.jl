using Ising_Triangle
using LinearAlgebra
using Random
using TeneT
using Test
using CUDA
using Zygote
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

@testset "$(Ni)x$(Nj) Ising_Triangle_bad Z and energy with $atype $updown updown" for Ni = [1], Nj = [1], updown = [false, true], atype = [Array]
    Random.seed!(100)
    β = 1
    model = Ising_Triangle_bad(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:bulk)))
    env = obs_env(M; χ = 20, maxiter = 10, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = updown, verbose = true, savefile = false
        )

    @show updown_overlap(env)
    @show observable(env, model, Val(:Z)     )
    @show observable(env, model, Val(:energy)) 
end

@testset "$(Ni)x$(Nj) Ising_Triangle_bad2 Z and energy with $atype $updown updown" for Ni = [1], Nj = [1], updown = [false, true], atype = [Array]
    Random.seed!(100)
    β = 1
    model = Ising_Triangle_bad2(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:bulk)))
    env = obs_env(M; χ = 20, maxiter = 10, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = updown, verbose = true, savefile = false
        )

    @show updown_overlap(env)
    @show observable(env, model, Val(:Z)     )
end

@testset "$(Ni)x$(Nj) Ising_Triangle_good Z and energy with $atype $updown updown" for Ni = [1], Nj = [1], updown = [false, true], atype = [Array]
    Random.seed!(100)
    β = 1
    model = Ising_Triangle_good(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:bulk)))
    env = obs_env(M; χ = 20, maxiter = 10, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = updown, verbose = true, savefile = false
        )

    @show updown_overlap(env)
    @show observable(env, model, Val(:Z)     )
    @show observable(env, model, Val(:energy)) 
end

@testset "$(Ni)x$(Nj) Ising_Triangle_bad residual entropy with $atype $updown updown" for Ni = [1], Nj = [1], updown = [false, true], atype = [Array]
    Random.seed!(100)
    β = 100
    model = Ising_Triangle_bad(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:Sbulk)))
    env = obs_env(M; χ = 20, maxiter = 50, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = updown, verbose = true, savefile = false
        )

    @show updown_overlap(env)
    @show observable(env, model, Val(:S)) - 0.3230659669
end

@testset "$(Ni)x$(Nj) Ising_Triangle_bad2 residual entropy with $atype $updown updown" for Ni = [1], Nj = [1], updown = [false], atype = [Array]
    Random.seed!(100)
    β = 100
    model = Ising_Triangle_bad2(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:Sbulk)))
    env = obs_env(M; χ = 32, maxiter = 100, miniter = 1, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = updown, verbose = true, savefile = true,show_every=1
        )

    @show updown_overlap(env)
    @show norm(observable(env, model, Val(:S)) - 0.3230659669)
end

@testset "$(Ni)x$(Nj) Ising_Triangle_good residual entropy with $atype $updown updown" for Ni = [1], Nj = [1], updown = [false], atype = [CuArray]
    Random.seed!(100)
    β = 100
    model = Ising_Triangle_good(Ni, Nj, β)
    M = atype(model_tensor(model, Val(:Sbulk)))
    env = obs_env(M; χ = 128, maxiter = 50, miniter = 10, 
         infolder = "./example/data/$model/", 
        outfolder = "./example/data/$model/", 
        updown = updown, verbose = true, savefile = true, show_every=1
        )

    @show updown_overlap(env)
    @show norm(observable(env, model, Val(:S)) - 0.3230659669)
    # @test observable(env, model, Val(:S)) ≈ 0.3230659669 atol = 1e-4
end