#===================================================================================================
Basic array functions
===================================================================================================#
function Base.similar(x::LArray{Syms,D}, ::Type{S}, dims::NTuple{N,Int}) where {Syms,D,S,N}
    tmp = similar(values(x), S, dims)
    return LArray{Syms}(tmp)
end

# Allow copying LArray of uninitialized data, as with regular Array
Base.copy(x::LArray) = typeof(x)(copy(values(x)))
Base.copyto!(x::LArray, y::LArray) = copyto!(values(x), values(y))

# enable the usage of LAPACK
function Base.unsafe_convert(::Type{Ptr{T}}, x::LArray{Syms,D,T}) where {Syms,D,T}
    return Base.unsafe_convert(Ptr{T}, values(x))
end

Base.convert(::Type{T}, x) where {T <: LArray} = T(x)
Base.convert(::Type{T}, x::T) where {T <: LArray} = x
Base.convert(::Type{<:Array}, x::LArray) = convert(Array, values(x))
Base.convert(::Type{<:AA}, x::LArray) where {AA<:AbstractArray} = convert(AA, values(x))

#=
#This function doesn't convert to the actual return type, I might want to verify this one in testing (see original)
#I think they should simply call AA(values(x)) where {T,N,AA<:AbstractArray{T,N}}
#https://github.com/SciML/LabelledArrays.jl/blob/79b4ff429e6264e93c18551c779b25888d1468bf/src/larray.jl#L146
function Base.convert(::Type{AbstractArray{T,N}}, x::LArray{Syms,D,<:Any,N}) where {Syms,D,T,N}
    LArray{Syms}(convert(AbstractArray{T, N}, values(x)))
end
Base.convert(::Type{AbstractArray{T, N}}, x::LArray{T, N}) where {T, N} = x
=#


#===================================================================================================
Broadcast utilities
===================================================================================================#
struct LAStyle{Syms,T,N} <: Broadcast.AbstractArrayStyle{N} end
LAStyle{Syms,T,N}(x::Val{i}) where {Syms,T,N,i} = LAStyle{Syms,T,N}()

Base.BroadcastStyle(::Type{LArray{Syms,D,T,N}}) where {Syms,D,T,N} = LAStyle{Syms,T,N}()
function Base.BroadcastStyle(::LabelledArrays.LAStyle{Syms,T,N}, ::LabelledArrays.LAStyle{Syms,E,N}) where {Syms,T,E,N}
    return LAStyle{Syms,promote_type(T,E),N}()
end

@generated function labels2axes(::Val{t}) where {t}
    if (t isa NamedTuple) && all(x -> x isa Union{Integer, UnitRange}, values(t)) # range labelling
        (Base.OneTo(maximum(Iterators.flatten(v for v in values(t)))),)
    elseif t isa NTuple{<:Any, Symbol}
        axes(t)
    else
        error("$t label isn't supported for broadcasting. Try to formulate it in terms of linear indexing.")
    end
end

function Base.similar(bc::Broadcast.Broadcasted{LAStyle{Syms,T,N}}, ::Type{ElType}) where {Syms,T,N,ElType}
    tmp = similar(Array{ElType}, axes(bc))
    if axes(bc) != labels2axes(Val(Syms))
        return tmp
    else
        return LArray{Syms, typeof(tmp), ElType, N}(tmp)
    end
end