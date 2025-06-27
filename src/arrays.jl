#===================================================================================================
Basic array functions
===================================================================================================#
Base.strides(A::LArray) = strides(values(A))
Base.stride(A::LArray, k::Integer) = stride(values(A), k)

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
function Base.BroadcastStyle(::LAStyle{Syms,T,N}, ::LAStyle{Syms,E,N}) where {Syms,T,E,N}
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

#===================================================================================================
SciML interfaces (may want this as extensions)
===================================================================================================#
#=
import ArrayInterface
function ArrayInterface.ismutable(::Type{<:LArray{Syms,D,T,N}}) where {Syms,D,T,N}
    return ArrayInterface.ismutable(D)
end
ArrayInterface.can_setindex(::Type{<:LArray{Syms,D}}) where {Syms,D} = ArrayInterface.can_setindex(D)


function ArrayInterface.undefmatrix(x::LArray{Syms,D,T,N}) where {Syms,D,T,N}
    lenfun(x) = length(x)
    lenfun(::Symbol) = 1
    n = sum(lenfun, Syms)
    return similar(values(x), n, n)
end

function PreallocationTools.get_tmp(dc::PreallocationTools.DiffCache, 
    u::LArray{Syms,D,T,N}) where {Syms, D, T<:ForwardDiff.Dual, N}
    nelem = div(sizeof(T), sizeof(eltype(dc.dual_du))) * length(dc.du)
    if nelem > length(dc.dual_du)
        PreallocationTools.enlargedualcache!(dc, nelem)
    end
    _x = ArrayInterface.restructure(dc.du, reinterpret(T, view(dc.dual_du, 1:nelem)))
    return LabelledArrays.LArray{T, N, D, Syms}(_x)
end
=#

#===================================================================================================
Static array interface
===================================================================================================#
function StaticArrays.similar_type(::Type{<:LArray{Syms,SArray{S,T,N,L}}}, 
    ::Type{NewElType}, ::Size{NewSize}) where {S, T, N, L, Syms, NewElType, NewSize}
    
    NewLen = prod(NewSize)
    NewDim = length(NewSize)
    NewSA  = SArray{Tuple{NewSize...}, NewElType, NewDim, NewLen}

    if NewLen == length(Syms) #Return a labelled version only if the lengths match
        return LArray{Syms, NewSA, NewElType, NewDim}
    else
        return NewSA
    end
end

function Base.similar(::Type{<:LArray{Syms, SArray{S,T,N,L}}}, 
    ::Type{NewElType}, ::Size{NewSize}) where {S, T, N, L, Syms, NewElType, NewSize}

    NewLen = prod(NewSize)
    NewDim = length(NewSize)

    if NewLen == length(Syms) #Return a labelled version only if the lengths match
        return LArray{Syms}(Array{NewElType}(undef, NewSize))
    else
        return MArray{Tuple{NewSize...}, NewElType, NewDim, NewLen}(undef)
    end
end

@inline Base.reshape(a::LArray{Syms,<:SArray}, s::Size) where Syms = StaticArrays.similar_type(a, s)(Tuple(a))
