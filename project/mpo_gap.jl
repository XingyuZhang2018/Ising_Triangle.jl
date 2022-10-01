using Ising_Triangle
using Test
using OMEinsum
using KrylovKit

@testset "Ising_Triangle_good MPO gap with $atype $updown updown" for updown = [false], atype = [Array]
    β = 100
    for L in 1:10
        model = Ising_Triangle_good(1, 1, β)
        M = reshape(atype(model_tensor(model, Val(:Sbulk))),4,4,4,4)
        MPO = M
        for i in 1:(L-1)
            MPO = ein"abcd,cefg->abefdg"(MPO, M)
            MPO = reshape(MPO, 4, 4^(i+1), 4, 4^(i+1))
        end
        MPO = ein"abcd->bd"(MPO)
        λ,_ = eigsolve(MPO,2)
        message = "$(L), $(abs(λ[1])-abs(λ[2]))\n" 
        @show message
        logfile = open("./example/data/$model/gap.log", "a")
        write(logfile, message)
        close(logfile)
    end
end

@testset "Ising_Triangle_bad2 MPO gap with $atype $updown updown" for updown = [false], atype = [Array]
    β = 100
    for L in 1:10
        model = Ising_Triangle_bad2(1, 1, β)
        M = reshape(atype(model_tensor(model, Val(:Sbulk))),2,2,2,2)
        MPO = M
        for i in 1:(L-1)
            MPO = ein"abcd,cefg->abefdg"(MPO, M)
            MPO = reshape(MPO, 2, 2^(i+1), 2, 2^(i+1))
        end
        MPO = ein"abcd->bd"(MPO)
        λ,_ = eigsolve(MPO,2)
        @show λ
        message = "$(L), $(abs(λ[1])-abs(λ[2]))\n" 
        @show message
        logfile = open("./example/data/$model/gap.log", "a")
        write(logfile, message)
        close(logfile)
    end
end