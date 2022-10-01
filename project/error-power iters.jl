using ADMPS
using CUDA
using Ising_Triangle
using Ising_Triangle: model_tensor
using LinearAlgebra:norm
using Random
using Test

@testset "optimize mps Ising free energy with $atype $updown updown" for updown = [false], atype = [Array]
    Random.seed!(100)
    β = log(1+sqrt(2))/2
    model = Ising_Triangle.Ising(1, 1, β)
    M = reshape(atype(model_tensor(model, Val(:bulk))),2,2,2,2)
    χ = 16
    exppara = [0;0:9]
    for i in 1:length(exppara)
        iter = 2^(exppara[i])
        Au, Ad = optimisemps(M; χ = χ, mapsteps = iter, f_tol = 1e-10,
            infolder = "./example/data/$model/", 
            outfolder = "./example/data/$model/",   
            verbose= true, updown = updown)
        env = ADMPS.obs_env(M,Au,Ad)
        
        message = "$(2^(i-1)), $(norm(log(ADMPS.Z(env)) - 0.929695402437653))\n" 
        @show message
        logfile = open("./example/data/$model/error_chi$(χ).log", "a")
        write(logfile, message)
        close(logfile)
    end
end

@testset "optimize mps Ising_Triangle_bad2 residual entropy with $atype $updown updown" for updown = [false], atype = [Array]
    Random.seed!(100)
    β = 100
    model = Ising_Triangle_bad2(1, 1, β)
    M = reshape(atype(model_tensor(model, Val(:Sbulk))),2,2,2,2)
    χ = 32
    exppara = [0;0:9]
    for i in 1:length(exppara)
        iter = 2^(exppara[i])
        Au, Ad = optimisemps(M; χ = χ, mapsteps = iter, f_tol = 1e-10,
            infolder = "./example/data/$model/", 
            outfolder = "./example/data/$model/",   
            verbose= true, updown = updown)
        env = ADMPS.obs_env(M,Au,Ad)
        
        message = "$(2^(i-1)), $(norm(log(ADMPS.Z(env)) - 0.3230659669))\n" 
        @show message
        logfile = open("./example/data/$model/error_chi$(χ).log", "a")
        write(logfile, message)
        close(logfile)
    end
end

@testset "optimize mps Ising_Triangle_good residual entropy with $atype $updown updown" for updown = [false], atype = [Array]
    Random.seed!(100)
    β = 100
    model = Ising_Triangle_good(1, 1, β)
    M = reshape(atype(model_tensor(model, Val(:Sbulk))),4,4,4,4)
    χ = 32
    exppara = [0;0:9]
    for i in 1:length(exppara)
        iter = 2^(exppara[i])
        Au, Ad = optimisemps(M; χ = χ, mapsteps = iter, f_tol = 1e-10,
            infolder = "./example/data/$model/", 
            outfolder = "./example/data/$model/",   
            verbose= true, updown = updown)
        env = ADMPS.obs_env(M,Au,Ad)
        
        message = "$(2^(i-1)), $(norm(log(ADMPS.Z(env)) - 0.3230659669))\n" 
        @show message
        logfile = open("./example/data/$model/error_chi$(χ).log", "a")
        write(logfile, message)
        close(logfile)
    end
end

