#===================================================================================================
SymbolicIndexer indexing
===================================================================================================#
Base.getindex(::Type{SymbolicIndexer{Syms}}, ind) where Syms = findsymbols(SymbolicIndexer{Syms}, Val(ind))
Base.getindex(idxr::SymbolicIndexer, ind) = findsymbols(typeof(idxr), Val(ind))

@generated function findsymbols(::Type{SymbolicIndexer{Syms}}, ::Val{ind}) where {Syms, ind}
    indexer = NamedTuple{Syms}(Base.OneTo(length(Syms))) 
    idx = indexer[ind]
    if idx isa Integer
        :($idx)
    elseif idx isa NamedTuple
        inds = SVector(values(idx))
        :($inds)
    end
end

#===================================================================================================
AbstractLabelledArray indexing
===================================================================================================#

#Single value indexing through getproperty
Base.getproperty(x::AbstractLabelledArray, name::Symbol) = getindex(x, name)
Base.setproperty!(x::AbstractLabelledArray, name::Symbol, y) = setindex!(x, y, name)

#passthrough indexing removes names
@propagate_inbounds Base.getindex(x::AbstractLabelledArray, inds...) = getindex(values(x), inds...)
@propagate_inbounds Base.setindex!(x::AbstractLabelledArray, y, inds...) = setindex!(values(x), y, inds...)

#Single value indexing
@propagate_inbounds function Base.getindex(x::AbstractLabelledArray{Syms}, ind::Symbol) where Syms
    num_ind = SymbolicIndexer(Syms)[ind]
    data = values(x)
    return data[lin_offset(num_ind, data)]
end

@propagate_inbounds function Base.setindex!(x::AbstractLabelledArray{Syms}, y, ind::Symbol) where Syms
    num_ind = SymbolicIndexer(Syms)[ind]
    data = values(x)
    return setindex!(data, y, lin_offset(num_ind, data))
end

#Multi value indexing always returns SLVector{Syms}
@propagate_inbounds function Base.getindex(x::AbstractLabelledArray{Syms}, ind::NTuple{N,Symbol}) where {Syms,N}
    return LArray{ind}(symcollect(x, ind))
end

@propagate_inbounds function Base.setindex!(x::AbstractLabelledArray{Syms}, y, ind::NTuple{N,Symbol}) where {Syms,N}
    num_ind = SymbolicIndexer(Syms)[ind]
    data = values(x)
    return setindex!(data, y, lin_offset(num_ind, data))
end

@propagate_inbounds function symcollect(x::AbstractLabelledArray{Syms}, ind::NTuple{N,Symbol}) where {Syms,N}
    vec_ind = SymbolicIndexer(Syms)[ind]
    data = values(x)
    return data[lin_offset(vec_ind, data)]
end
symcollect(d::AbstractDict{Symbol}, ind::NTuple{N,Symbol}) where N = map(k->d[k], SVector{N}(ind))
symcollect(d::AbstractDict{String}, ind::NTuple{N,Symbol}) where N = map(k->d[string(k)], SVector{N}(ind))
symcollect(nt::NamedTuple, ind::NTuple{N,Symbol}) where N = SVector{N}(values(nt[ind]))


#No need for branching for integer or range index
lin_offset(ind::Union{Integer, AbstractRange}, data::AbstractArray) = ind + lin_offset(data)

#Avoid adding offsets for array indices if the offset is zero (avoids work)
function lin_offset(ind, data::AbstractArray)
    Δ = lin_offset(data)
    return iszero(Δ) ? ind : (ind + Δ)
end

lin_offset(data::AbstractArray) = firstindex(data) - 1
lin_offset(data::Array) = 0
lin_offset(data::StaticArray) = 0

