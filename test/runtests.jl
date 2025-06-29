#using Revise
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
    @test getsvec(SLVector{syms}(1:3), (:a,:b)) == SVector(1,2)

    syms = (:a,:b,:c,:d)
    @test LArray{syms, SVector{4,Float32}}(ones(2,2)) === LArray{syms}(SVector{4,Float32}(ones(4)))
    @test SLVector{syms, Float32}(ones(2,2)) === LArray{syms}(SVector{4,Float32}(ones(4)))

    #Setting indices
    x = [1,2,3]
    xL = LArray{(:a,:b,:c)}(x)

    xL[:b] = 3
    @test x == [1,3,3]

    x .= [1,2,3]
    xL[(:a,:b)] = 2
    @test x == [2,2,3]

    #Conversions between different LArrays
    xL = LArray{(:a,:b,:c)}([1.0,2.0,3.0])
    @test SLVector{(:b,:c)}(xL) === SLVector(b=2.0, c=3.0)
    @test LArray{(:b,:c)}(xL) === SLVector(b=2.0, c=3.0)
    @test LArray{(:b,:c),Vector}(xL) == LArray{(:b,:c)}([2.0, 3.0])
    @test LArray{(:b,:c),Vector{Int32}}(xL) isa LArray{(:b,:c), Vector{Int32}}

    #Indexing with an LArray
    x = 1.0:4.0
    x[LArray(a=1,b=2)] == LArray(a=1.0, b=2.0)

    #Testing bad symbolic names
    @test_throws ArgumentError SymbolicIndexer{(:a,:a,:b)}()
    @test_throws ArgumentError SLVector{(:a,:a,:b)}(1:3)
    @test_throws ArgumentError LArray{(:a,:a,:b)}(1:3)
    @test_throws TypeError SLVector{(1,:a,:b)}(1:3)
end

@testset "Aqua.jl" begin
    Aqua.test_all(NamedVectors)
end

nothing