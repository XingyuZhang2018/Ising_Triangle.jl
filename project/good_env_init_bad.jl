using ADMPS
using CUDA
using Ising_Triangle
using Ising_Triangle: model_tensor
using LinearAlgebra:norm
using Random
using Test

@testset "optimize mps Ising_Triangle_bad2 residual entropy with $atype $updown updown" for updown = [false], atype = [Array]
    Random.seed!(100)
    β = 100
    model = Ising_Triangle_bad2(1, 1, β)
    M = reshape(atype(model_tensor(model, Val(:Sbulk))),2,2,2,2)
    M = reshape(ein"ailh,lkfg,bcji,jdek->abcdfehg"(M,M,M,M),4,4,4,4)
    χ = 16
    Au, Ad = optimisemps(M; χ = χ, mapsteps = 10, f_tol = 1e-10,
        infolder = "./example/data/$model/good_intial/", 
        outfolder = "./example/data/$model/good_intial/",   
        verbose= true, updown = updown)
    env = ADMPS.obs_env(M,Au,Ad)
    @show norm(log(ADMPS.Z(env))/4 - 0.3230659669)
end
