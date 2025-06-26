import Base.@assert
import Base.@propagate_inbounds
using StaticArrays 

abstract type AbstractLabelledArray{Syms,T,N} <: AbstractArray{T,N} end
const AbstractLabelledMatrix{Syms,T} = AbstractLabelledArray{Syms,T,2} 
const AbstractLabelledVector{Syms,T} = AbstractLabelledArray{Syms,T,1} 

"""
    LArray{Syms,D<:AbstractArray,T,N}

Wraps an array of type "D<:AbstractArray{T,N}" with a list of symbolic names.
These names can be used to index into the original array linearly.
"""
struct LArray{Syms,D<:AbstractArray,T,N} <: AbstractLabelledArray{Syms,T,N}
    data::D
    function LArray{Syms,D,T,N}(data::AbstractArray) where {Syms,D,T,N} 
        (D <: AbstractArray{T,N}) || throw(ArgumentError("Data type parameter $(D) must be of type AbstractArray{$(T),$(N)}"))
        _check_labels(Syms, :LArray)
        _check_lengths(Syms, data)
        return new{Syms,D,T,N}(data)
    end
    function LArray{Syms,D}(data::AbstractArray) where {Syms,T,N,D<:AbstractArray{T,N}} 
        _check_labels(Syms, :LArray)
        _check_lengths(Syms, data)
        return new{Syms,D,T,N}(data)
    end
    function LArray{Syms}(data::D) where {Syms,T,N,D<:AbstractArray{T,N}} 
        _check_labels(Syms, :LArray)
        _check_lengths(Syms, data)
        return new{Syms,D,T,N}(data)
    end
end

const LVector{Syms,D,T} = LArray{Syms,D,T,1}
LVector{Syms,D}(data::AbstractVector) where {Syms,D<:AbstractVector} = LArray{Syms,D}(data)
LVector{Syms,D}(data::AbstractArray) where {Syms,D<:AbstractVector} = LArray{Syms,D}(reshape(data,:))
LVector{Syms}(data::AbstractVector) where {Syms} = LArray{Syms}(data)
LVector{Syms}(data::AbstractArray) where {Syms} = LArray{Syms}(reshape(data,:))

const LMatrix{Syms,D,T} = LArray{Syms,D,T,2}
LMatrix{Syms,D}(data::AbstractMatrix) where {Syms,D<:AbstractMatrix} = LArray{Syms,D}(data)
LMatrix{Syms}(data::AbstractMatrix) where {Syms} = LArray{Syms}(data)


"""
    SLVector{Syms,T,L}

A special type of "LArray" that is designed to mimic the NamedTuple API. Since the underlying data
is actually a Tuple, it is essentially a NamedTuple with a uniform type and vector-like behaviour
"""
const SLVector{Syms,T,L} = LArray{Syms,SVector{L,T},T,1}
SLVector{Syms,T,L}(data::AbstractArray) where {Syms,T,L} = LArray{Syms}(SVector{L,T}(data))
SLVector{Syms,T}(data::AbstractArray) where {Syms,T} = LArray{Syms}(SVector{length(Syms),T}(data))
SLVector{Syms}(data::AbstractArray) where {Syms} = LArray{Syms}(SVector{length(Syms)}(data))

#If the argument is not an array, use the SLVector constructor which is the most specific and efficient
LArray(x::Any) = SLVector(x)
LVector(x::Any) = SLVector(x)
LArray(;kwargs...) = SLVector(;kwargs...)
LVector(;kwargs...) = SLVector(;kwargs...)
LArray{Syms}(x::Any) where {Syms} = SLVector{Syms}(x)
LVector{Syms}(x::Any) where {Syms} = SLVector{Syms}(x)
LArray{Syms,D}(x::Any) where {Syms,D<:AbstractArray} = LArray{Syms}(convert(D,values(SLVector{Syms}(x))))
LVector{Syms,D}(x::Any) where {Syms,D<:AbstractVector} = LArray{Syms}(convert(D,values(SLVector{Syms}(x))))

#Interop with Tuple/NamedTuple
(::Type{SLV})(data::Tuple) where {Syms, SLV<:SLVector{Syms}} = SLV(SVector(data))
SLVector(data::NamedTuple{Syms}) where Syms = SLVector{Syms}(SVector(values(data)))
SLVector{Syms}(data::NamedTuple) where Syms = SLVector(data[Syms])
SLVector(;kwargs...) = SLVector(values(kwargs))
Base.NamedTuple(data::AbstractLabelledArray{Syms}) where Syms = NamedTuple{Syms}(convert(NTuple{length(syms)}, values(data)))

#Interop with dictionaries
(::Type{SLV})(data::Dict{Symbol}) where {Syms, SLV<:SLVector{Syms}} = SLV(map(Base.Fix1(getindex, data), Syms))
(::Type{SLV})(data::Dict{String}) where {Syms, SLV<:SLVector{Syms}} = SLV(map(Base.Fix1(getindex, data), map(string, Syms)))

#Cross-conversion
SLVector{Syms}(data::AbstractLabelledArray) where {Syms} = data[Syms]
SLVector{Syms,T}(data::AbstractLabelledArray) where {Syms,T} = SLVector{Syms,T}(values(data, Syms))
SLVector{Syms,T,L}(data::AbstractLabelledArray) where {Syms,T,L} = SLVector{Syms,T,L}(values(data, Syms))

"""
    SymbolicIndexer{Syms}

A type that returns the numeric index of the list of symbols in Syms. It essentially functions as a 
NamedTuple{Syms}(OneTo(length(Syms)))[sym], or findfirst(i->i==sym, Syms).

For example, SymbolicIndexer((:a,:b,:c))[:b] will return "2"
"""
struct SymbolicIndexer{Syms} 
    SymbolicIndexer{Syms}() where Syms = new{_check_labels(Syms, :SymbolicIndexer)}()
end
SymbolicIndexer(x::NTuple{N,Symbol}) where N = SymbolicIndexer{x}()
Base.getproperty(x::SymbolicIndexer, name::Symbol) = getindex(x, name)


#===================================================================================================
NamedTuple API duplication
===================================================================================================#
Base.values(x::AbstractLabelledArray) = getfield(x, :data)
Base.propertynames(x::AbstractLabelledArray{Syms}) where Syms = Syms
Base.keys(x::AbstractLabelledArray{Syms}) where Syms = Syms
function Base.pairs(x::AbstractLabelledArray{Syms}) where Syms
    (Syms[i] => xi for (i, xi) in enumerate(x))
end
Base.size(x::AbstractLabelledArray) = size(values(x))

function Base.:(==)(x1::AbstractLabelledArray{Syms1}, x2::AbstractLabelledArray{Syms2}) where {Syms1,Syms2} 
    return (Syms1==Syms2) && (values(x1)==values(x2))
end

#===================================================================================================
Checking utility functions
===================================================================================================#
function _check_labels(Syms, func) 
    (Syms isa NTuple{L,Symbol} where L) || throw(TypeError(func, "Syms", NTuple{N,Symbol} where N, Syms))
    allunique(Syms) || throw(ArgumentError("Duplicate name in $(func){$(Syms)}"))
    return Syms
end
_check_lengths(Syms, data) = (length(Syms) == length(data)) || throw(ArgumentError("Number of elements must match the number of names"))


#=
struct SLVector{Syms,T,L} <: AbstractLabelledVector{Syms,T}
    data::SVector{L,T}
    function SLVector{Syms,T,L}(data::AbstractArray) where {Syms,T,L}
        _check_labels(Syms, :SLVector)
        (L == length(Syms) == length(data)) || "Lenth parameter must match the number of elements must match the number of names"
        return new{Syms,T,L}(data)
    end
    function SLVector{Syms,T}(data::AbstractArray) where {Syms,T}
        _check_labels(Syms, :SLVector)
        _check_lengths(Syms, data)
        L = length(Syms)
        return new{Syms,T,L}(data)
    end
    function SLVector{Syms}(data::AbstractArray{T}) where {Syms,T}
        _check_labels(Syms, :SLVector)
        _check_lengths(Syms, data)
        L = length(Syms)
        return new{Syms,T,L}(data)
    end
end
=#