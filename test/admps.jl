using ADMPS
using CUDA
using Ising_Triangle
using Ising_Triangle: model_tensor
using LinearAlgebra
using LinearAlgebra:norm
using Random
using Test
using Optim

@testset "normality" begin
    A = rand(ComplexF64, 5, 5)
    A = A + A'

    val, vec = eigen(A)
    @test val[1] * vec[:,1] ≈ A * vec[:,1]
    @test A * vec ≈ vec * diagm(val)
    @show opnorm(vec, 2) * opnorm(vec^-1, 2)
    
end

@testset "optimize mps Ising_Triangle_bad with $atype $updown updown" for updown = [false], atype = [Array]
    Random.seed!(100)
    β = 100
    model = Ising_Triangle_bad(1, 1, β)
    M = reshape(atype(model_tensor(model, Val(:bulk))),4,4,4,4)
    Au, Ad = optimisemps(M; χ = 20, mapsteps = 20,
            infolder = "./example/data/$model/", 
            outfolder = "./example/data/$model/",   
            verbose= true, updown = updown)
    env = ADMPS.obs_env(M,Au,Ad)
    @show ADMPS.Z(env)
end

@testset "optimize mps Ising_Triangle_good with $atype $updown updown" for updown = [false,true], atype = [Array]
    Random.seed!(100)
    β = 1
    model = Ising_Triangle_good(1, 1, β)
    M = reshape(atype(model_tensor(model, Val(:bulk))),4,4,4,4)
    Au, Ad = optimisemps(M; χ = 20, mapsteps = 20,
            infolder = "./example/data/$model/", 
            outfolder = "./example/data/$model/",   
            verbose= true, updown = updown)
    env = ADMPS.obs_env(M,Au,Ad)
    @show ADMPS.Z(env)
end

@testset "optimize mps Ising_Triangle_bad2 residual entropy with $atype $updown updown" for updown = [false], atype = [Array]
    Random.seed!(100)
    β = 100
    model = Ising_Triangle_bad2(1, 1, β)
    M = reshape(atype(model_tensor(model, Val(:Sbulk))),2,2,2,2)
    Au, Ad = optimisemps(M; χ = 16, mapsteps = 10, f_tol = 1e-6,
                         infolder = "./example/data/$model/", 
                         outfolder = "./example/data/$model/",   
                         verbose= true, updown = updown
                         )
    env = ADMPS.obs_env(M,Au,Ad)
    @show norm(log(ADMPS.Z(env)) - 0.3230659669)
end

@testset "optimize mps Ising_Triangle_good residual entropy with $atype $updown updown" for updown = [false], atype = [Array]
    Random.seed!(100)
    β = 100
    model = Ising_Triangle_good(1, 1, β)
    M = reshape(atype(model_tensor(model, Val(:Sbulk))),4,4,4,4)
    Au, Ad = optimisemps(M; χ = 128, mapsteps = 20, f_tol = 1e-6,
            infolder = "./example/data/$model/", 
            outfolder = "./example/data/$model/",   
            verbose= true, updown = updown)
    env = ADMPS.obs_env(M,Au,Ad)
    @show norm(log(ADMPS.Z(env)) - 0.3230659669)
end

@testset "optimize mps J1J2_2 with $atype $updown updown" for updown = [true], atype = [Array]
    Random.seed!(100)
    J2 = 0.55
    β = 1.2
    model = J1J2_2(1, 1, J2, β)
    M = reshape(atype(model_tensor(model, Val(:bulk))),4,4,4,4)
    Au, Ad = optimisemps(M; χ = 16, mapsteps = 20,
            infolder = "./example/data/$model/", 
            outfolder = "./example/data/$model/",   
            verbose= true, updown = updown)
    env = ADMPS.obs_env(M,Au,Ad)
    # @show ADMPS.Z(env)
    @show observable(env, model, Val(:energy)   )
end