using NamedVectors
using Test
using StaticArrays
using Aqua

@testset "NamedVectors.jl" begin
    syms = (:a,:b,:c)
    @test LArray{syms}(1:3) == LArray{(:a,:b,:c)}(1:3)
    @test LArray{syms,Vector{Float64}}(1:3) == LArray{(:a,:b,:c)}(1.0:3.0)
    @test SLVector{syms}(1:3) == SLVector((a=1,b=2,c=3))
    @test SLVector{syms,Float64}(1:3) == SLVector((a=1.0, b=2,c=3))
    @test SLVector{syms}(1:3)[(:a,:b)] == SLVector((a=1,b=2))
    @test values(SLVector{syms}(1:3), (:a,:b)) == SVector(1,2)

    syms = (:a,:b,:c,:d)
    @test LVector{syms}(ones(2,2)) == LArray{syms}(ones(4))
    @test LArray{syms, SVector{4,Float32}}(ones(2,2)) === LVector{syms}(SVector{4,Float32}(ones(4)))

    x = [1,2,3]
    xL = LArray{(:a,:b,:c)}(x)

    xL[:b] = 3
    @test x == [1,3,3]

    x .= [1,2,3]
    xL[(:a,:b)] = 2
    @test x == [2,2,3]

    x .= [1,2,3]
    @test SLVector{(:b,:c)}(xL) == SLVector(b=2,c=3)
    #@test LVector{(:b,:c)}(xL) == SLVector(b=2,c=3)
    #@test LArray{(:b,:c)}(xL) == SLVector(b=2,c=3)

    @test_throws ArgumentError SymbolicIndexer{(:a,:a,:b)}()
    @test_throws ArgumentError SLVector{(:a,:a,:b)}(1:3)
    @test_throws ArgumentError LArray{(:a,:a,:b)}(1:3)
    @test_throws TypeError SLVector{(1,:a,:b)}(1:3)
end

@testset "Aqua.jl" begin
    Aqua.test_all(NamedVectors)
end