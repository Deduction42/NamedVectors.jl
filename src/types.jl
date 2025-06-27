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
const LMatrix{Syms,D,T} = LArray{Syms,D,T,2}


"""
    SLVector{Syms,T,L}

A special type of "LArray" that is designed to mimic the NamedTuple API. Since the underlying data
is actually a Tuple, it is essentially a NamedTuple with a uniform type and vector-like behaviour
"""
const SLVector{Syms,T,L} = LArray{Syms,SVector{L,T},T,1}
SLVector{Syms,T,L}(data::AbstractArray) where {Syms,T,L} = LArray{Syms}(SVector{L,T}(data))
SLVector{Syms,T}(data::AbstractArray) where {Syms,T} = LArray{Syms}(SVector{length(Syms),T}(data))
SLVector{Syms}(data::AbstractArray) where {Syms} = LArray{Syms}(SVector{length(Syms)}(data))

#===================================================================================================
Various constructors
===================================================================================================#

#Generic constructors
LArray{Syms,D}(x::Any) where {Syms,D} = LArray{Syms}(convert(D, getsvec(x, Syms)))
SLVector{Syms}(x::Any) where {Syms} = SLVector{Syms}(getsvec(x, Syms))
LArray{Syms}(x::Any) where {Syms} = SLVector{Syms}(x)
LArray(x::Any) = SLVector(x)
LArray(;kwargs...) = SLVector(;kwargs...)

#Interop with Tuple/NamedTuple
SLVector{Syms}(x::Tuple) where {Syms} = SLVector{Syms}(SVector(x))
Base.Tuple(x::AbstractLabelledArray{Syms,<:SArray}) where {Syms} = Tuple(values(x))
SLVector(x::NamedTuple{Syms}) where Syms = SLVector{Syms}(SVector(values(x)))
SLVector(;kwargs...) = SLVector(values(kwargs))
Base.NamedTuple(x::AbstractLabelledArray{Syms}) where Syms = NamedTuple{Syms}(NTuple{length(syms),eltype(x)}(values(x)))

#Cross-conversion
LArray{Syms,D,T,N}(data::AbstractLabelledArray) where {Syms,D,T,N} = LArray{Syms,D,T,N}(getsvec(data, Syms))
LArray{Syms,D}(data::AbstractLabelledArray) where {Syms,T,N,D<:AbstractArray{T,N}} = LArray{Syms,D,T,N}(getsvec(data, Syms))
LArray{Syms}(data::AbstractLabelledArray) where {Syms} = LArray{Syms}(getsvec(data, Syms))

SLVector{Syms,T,L}(data::AbstractLabelledArray) where {Syms,T,L} = SLVector{Syms,T,L}(getsvec(data, Syms))
SLVector{Syms,T}(data::AbstractLabelledArray) where {Syms,T} = SLVector{Syms,T}(getsvec(data, Syms))
SLVector{Syms}(data::AbstractLabelledArray) where {Syms} = SLVector{Syms}(getsvec(data, Syms))

#Converting from LArray to other array types
Base.convert(::Type{AT}, x::AT) where {AT<:LArray} = x
Base.convert(::Type{AT}, x::LArray) where {AT<:Array} = convert(AT, values(x))
Base.convert(::Type{AT}, x::LArray) where {AT<:StaticArray} = convert(AT, values(x))


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
NamedTuple API compliance
===================================================================================================#
Base.values(x::AbstractLabelledArray) = getfield(x, :data)
Base.propertynames(x::AbstractLabelledArray{Syms}) where Syms = Syms
Base.keys(x::AbstractLabelledArray{Syms}) where Syms = Syms
@inline symnames(::Type{<:LArray{Syms}}) where {Syms} = Syms
function Base.pairs(x::AbstractLabelledArray{Syms}) where Syms
    (Syms[i] => xi for (i, xi) in enumerate(x))
end
Base.size(x::AbstractLabelledArray) = size(values(x))

function Base.:(==)(x1::AbstractLabelledArray{Syms1}, x2::AbstractLabelledArray{Syms2}) where {Syms1,Syms2} 
    return (Syms1==Syms2) && (values(x1)==values(x2))
end


#===================================================================================================
Display utilities
===================================================================================================#
struct PrintWrapper{T, N, F, X <: AbstractArray{T, N}} <: AbstractArray{T, N}
    f::F
    x::X
end

for f in (:eltype, :length, :ndims, :size, :axes, :eachindex, :stride, :strides)
    @eval Base.$f(wrapper::PrintWrapper) = $f(wrapper.x)
end
Base.getindex(A::PrintWrapper, idxs...) = A.f(A.x, A.x[idxs...], idxs)
Base.getindex(A::PrintWrapper, idxs::LArray{Syms}) where Syms =  A.f(A.x, A.x[values(idxs)], values(idxs))

function lazypair(A, x, idxs)
    syms = symnames(typeof(A))
    II = LinearIndices(A)
    key = eltype(syms) <: Symbol ? syms[II[idxs...]] : findfirst(syms) do sym
        ii = idxs isa Tuple ? II[idxs...] : II[idxs]
        sym isa Tuple ? ii in II[sym...] : ii in II[sym]
    end
    key => x
end

Base.show(io::IO, ::MIME"text/plain", x::LArray) = show(io, x)
function Base.show(io::IO, x::LArray)
    syms = symnames(typeof(x))
    n = length(syms)
    pwrapper = PrintWrapper(lazypair, x)
    if io isa IOContext && get(io, :limit, false) &&
       displaysize(io) isa Tuple{Integer, Integer}
        io = IOContext(io, :limit => true, :displaysize => cld.(2 .* displaysize(io), 3))
    end
    println(io, summary(x), ':')
    Base.print_array(io, pwrapper)
end


#===================================================================================================
Argument checking
===================================================================================================#
function _check_labels(Syms, func) 
    (Syms isa NTuple{L,Symbol} where L) || throw(TypeError(func, "Syms", NTuple{N,Symbol} where N, Syms))
    allunique(Syms) || throw(ArgumentError("Duplicate field names found in '$(func)': $(collect(_find_duplicates(Syms)))"))
    return Syms
end
_check_lengths(Syms, data) = (length(Syms) == length(data)) || throw(ArgumentError("Number of elements must match the number of names"))

function _find_duplicates(symbols::NTuple{N,Symbol}) where N
    deja_vu = Set{Symbol}()
    duplicated = Set{Symbol}()
    for s in symbols
        if s in deja_vu
            push!(duplicated, s)
        else
            push!(deja_vu, s)
        end
    end
    return duplicated 
end