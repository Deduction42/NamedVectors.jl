using NamedVectors
using Test

@testset "NamedVectors.jl" begin
    syms = (:a,:b,:c)
    @test LArray{syms}(1:3) == LArray{(:a,:b,:c)}(1:3)
    @test LArray{syms,Vector{Float64}}(1:3) == LArray{(:a,:b,:c)}(1.0:3.0)
    @test SLVector{syms}(1:3) == SLVector((a=1,b=2,c=3))
    @test SLVector{syms,Float64}(1:3) == SLVector((a=1.0, b=2,c=3))
    @test SLVector{syms}(1:3)[(:a,:b)] == SLVector((a=1,b=2))

    x = [1,2,3]
    LArray{(:a,:b,:c)}(x)[:b] = 3
    @test x == [1,3,3]

    x = [1,2,3]
    LArray{(:a,:b,:c)}(x)[(:a,:b)] = 2
    @test x == [2,2,3]

    
end
