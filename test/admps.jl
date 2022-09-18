using ADMPS
using Ising_Triangle:model_tensor
using Random
using Test

@testset "optimize mps" for atype in [Array]
    Random.seed!(100)
    β = 1
    model = Ising_Triangle_good(1, 1, β)
    M = reshape(atype(model_tensor(model, Val(:bulk))),4,4,4,4)
    Au, Ad = optimisemps(M; χ = 20, mapsteps = 20,
            infolder = "./example/data/$model/", 
            outfolder = "./example/data/$model/",   
            verbose= true, updown = true)
    env = ADMPS.obs_env(M,Au,Ad)
    @show ADMPS.Z(env)
end