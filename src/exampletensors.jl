using TeneT
using TeneT: _arraytype
using OMEinsum
using Zygote
using LinearAlgebra: I

const isingβc = log(1+sqrt(2))/2

abstract type HamiltonianModel end

export Ising, Ising_Triangle_bad, Ising_Triangle_good
export model_tensor

struct Ising <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
end

struct Ising_Triangle_bad <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
end

struct Ising_Triangle_good <: HamiltonianModel 
    Ni::Int
    Nj::Int
    β::Float64
end


"""
    model_tensor(model::Ising, type)

return the  `MT <: HamiltonianModel` `type` tensor at inverse temperature `β` for  two-dimensional
square lattice tensor-network.
"""
function model_tensor(model::Ising, ::Val{:bulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[-1. 1;1 -1]
    w = exp.(- β * ham)
    wsq = sqrt(w)
    m = ein"ia,ib,ic,id -> abcd"(wsq, wsq, wsq, wsq)

    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising_Triangle_bad, ::Val{:bulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[1. -1;-1 1]
    w = exp.(- β * ham)

    m = reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),w,w,w),4,4,4,4)
    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising_Triangle_good, ::Val{:bulk})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[1. -1;-1 1]
    w1 = exp.(- β * ham)
    w2 = exp.(- β * ham/2)

    m = reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,w2,w2,w2,w1),4,4,4,4)
    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising, ::Val{:mag})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    a = reshape(ComplexF64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 -1], 2,2,2,2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1/sqrt(2) * [cβ+sβ cβ-sβ; cβ-sβ cβ+sβ]
    m = ein"abcd,ai,bj,ck,dl -> ijkl"(a,q,q,q,q)
    M = Zygote.Buffer(m, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising, ::Val{:energy})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = ComplexF64[-1 1;1 -1]
    w = exp.(-β .* ham)
    we = ham .* w
    wsq = sqrt(w)
    wsqi = wsq^(-1)
    e = (ein"ai,im,bm,cm,dm -> abcd"(wsqi,we,wsq,wsq,wsq) + ein"am,bi,im,cm,dm -> abcd"(wsq,wsqi,we,wsq,wsq) + 
        ein"am,bm,ci,im,dm -> abcd"(wsq,wsq,wsqi,we,wsq) + ein"am,bm,cm,di,im -> abcd"(wsq,wsq,wsq,wsqi,we)) / 2
    M = Zygote.Buffer(e, 2,2,2,2,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = e
    end
    return copy(M)
end

function model_tensor(model::Ising_Triangle_bad, ::Val{:energy})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[1. -1;-1 1]
    w = exp.(- β * ham)
    we = ham .* w

    m = reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),we, w , w ),4,4,4,4) + 
        reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),w , we, w ),4,4,4,4) +
        reshape(ein"mi,mj,mk,ml,mn,mo,qp->ijklponq"(I(2),I(2),I(2),I(2),w , w , we),4,4,4,4)
    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

function model_tensor(model::Ising_Triangle_good, ::Val{:energy})
    Ni, Nj, β = model.Ni, model.Nj, model.β
    ham = Zygote.@ignore ComplexF64[1. -1;-1 1]
    w1 = exp.(- β * ham)
    w2 = exp.(- β * ham/2)
    we1 = ham .* w1
    we2 = ham .* w2/2

    m = reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),
        we2,w2,w2,w2,w1),4,4,4,4) + 
        reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,we2,w2,w2,w1),4,4,4,4) +
        reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,w2,we2,w2,w1),4,4,4,4) +
        reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,w2,w2,we2,w1),4,4,4,4) +
        reshape(ein"im,in,jo,jp,kq,kr,ls,lt,ik,kl,lj,ji,il->mqrsptno"(I(2),I(2),I(2),I(2),I(2),I(2),I(2),I(2),w2,w2,w2,w2,we1),4,4,4,4) 
    M = Zygote.Buffer(m, 4,4,4,4,Ni,Nj)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        M[:,:,:,:,i,j] = m
    end
    return copy(M)
end

