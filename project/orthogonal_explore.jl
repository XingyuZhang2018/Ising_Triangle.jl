using KrylovKit
using Random
using Test

@testset begin
    M = rand(ComplexF64,10,10)
    _,r,_ = eigsolve(r->M*r, rand(10,1), 2)
    _,l,_ = eigsolve(l->l*M, rand(1,10), 2)
    @show l[2]*r[2]
end