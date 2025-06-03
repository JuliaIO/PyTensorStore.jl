struct TensorStoreWrapper
    parent::Py
end
Base.parent(w::TensorStoreWrapper) = getfield(w, :parent)
PythonCall.Py(w::TensorStoreWrapper) = parent(w)
function Base.propertynames(w::TensorStoreWrapper)
    propertynames(parent(w))
end
function Base.getproperty(w::TensorStoreWrapper, s::Symbol)
    result = s == :read ? (x...,) -> FutureWrapper{pyarray_nocopy}(parent(w).read(x...)) :
                          getproperty(parent(w), s)
    return pyconvert(Any, result)
end
function Base.getindex(w::TensorStoreWrapper, indices...)
    @debug "getindex" indices
    py_indices = tsindex.(indices)
    @debug "py_indices" py_indices
    TensorStoreWrapper(parent(w)[py_indices...])
end
Base.size(w::TensorStoreWrapper) = pyconvert(Tuple, w.shape)

# Convert indicies into TensorStoreWrapper into Python indexes
tsindex(x) = x
tsindex(::Colon) = pyslice(nothing)
tsindex(i::Integer) = i-1
tsindex(r::UnitRange) = pyslice(first(r)-1, last(r))
tsindex(r::AbstractRange) = pyslice(first(r)-1, last(r), step(r))
tsindex(ci::CartesianIndex) = Tuple(ci - CartesianIndex(1,1))

