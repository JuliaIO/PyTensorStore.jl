"""
    TensorStoreWrapper(parent::Py)

A wrapper around a Python `tensorstore.TensorStore` object.
Implements much of the Julia `AbstractArray` interface.
"""
struct TensorStoreWrapper
    parent::Py
end

Base.parent(w::TensorStoreWrapper) = getfield(w, :parent)
PythonCall.Py(w::TensorStoreWrapper) = parent(w)

"""
    IndexDomainWrapper(parent::Py)

A wrapper around a Python `tensorstore.IndexDomain` object.
Represents the coordinate system and labels of a TensorStore.
"""
struct IndexDomainWrapper
    parent::Py
end
Base.parent(w::IndexDomainWrapper) = getfield(w, :parent)
PythonCall.Py(w::IndexDomainWrapper) = parent(w)

"""
    TransactionWrapper(parent::Py)

A wrapper around a Python `tensorstore.Transaction` object.
Used for atomic multi-write operations.
"""
struct TransactionWrapper
    parent::Py
end
Base.parent(w::TransactionWrapper) = getfield(w, :parent)
PythonCall.Py(w::TransactionWrapper) = parent(w)

"""
    SpecWrapper(parent::Py)

A wrapper around a Python `tensorstore.Spec` object.
Represents a serializable specification of a TensorStore.
"""
struct SpecWrapper
    parent::Py
end
Base.parent(w::SpecWrapper) = getfield(w, :parent)
PythonCall.Py(w::SpecWrapper) = parent(w)

"""
    SchemaWrapper(parent::Py)

A wrapper around a Python `tensorstore.Schema` object.
Describes the data type, domain, and layout of a TensorStore.
"""
struct SchemaWrapper
    parent::Py
end
Base.parent(w::SchemaWrapper) = getfield(w, :parent)
PythonCall.Py(w::SchemaWrapper) = parent(w)

"""
    ChunkLayoutWrapper(parent::Py)

A wrapper around a Python `tensorstore.ChunkLayout` object.
Describes how data is chunked on disk.
"""
struct ChunkLayoutWrapper
    parent::Py
end
Base.parent(w::ChunkLayoutWrapper) = getfield(w, :parent)
PythonCall.Py(w::ChunkLayoutWrapper) = parent(w)

"""
    ContextWrapper(parent::Py)

A wrapper around a Python `tensorstore.Context` object.
Used for sharing resources (like cache pools) between multiple TensorStore handles.
"""
struct ContextWrapper
    parent::Py
end
Base.parent(w::ContextWrapper) = getfield(w, :parent)
PythonCall.Py(w::ContextWrapper) = parent(w)

"""
    ContextSpecWrapper(parent::Py)

A wrapper around a Python `tensorstore.Context.Spec` object.
Represents the configuration for a `Context`.
"""
struct ContextSpecWrapper
    parent::Py
end
Base.parent(w::ContextSpecWrapper) = getfield(w, :parent)
PythonCall.Py(w::ContextSpecWrapper) = parent(w)

function Base.propertynames(w::TensorStoreWrapper)
    propertynames(parent(w))
end

function Base.getproperty(w::TensorStoreWrapper, s::Symbol)
    # Intercept read/write to return FutureWrapper
    if s == :read
        return (x...,) -> FutureWrapper{pyarray_nocopy}(parent(w).read(x...))
    elseif s == :write
        return (v, x...) -> FutureWrapper{ReturnsNothing}(parent(w).write(v, x...))
    elseif s == :domain
        return IndexDomainWrapper(parent(w).domain)
    elseif s == :spec
        return SpecWrapper(parent(w).spec())
    elseif s == :schema
        return SchemaWrapper(parent(w).schema)
    elseif s == :with_transaction
        return (txn) -> TensorStoreWrapper(parent(w).with_transaction(Py(txn)))
    else
        result = getproperty(parent(w), s)
        # Try converting to Any, but don't convert TensorStore objects that we want to keep wrapped
        return pyconvert(Any, result)
    end
end

ReturnsNothing(x) = nothing

function Base.getindex(w::TensorStoreWrapper, indices...; kwargs...)
    @debug "getindex" indices kwargs
    current = w
    if !isempty(indices)
        py_indices = map(tsindex, indices)
        current = TensorStoreWrapper(parent(current)[py_indices...])
    end
    if !isempty(kwargs)
        ts = @pyconst(pyimport("tensorstore"))
        keys = string.(Base.keys(kwargs))
        vals = [tsindex(v) for v in Base.values(kwargs)]
        current = TensorStoreWrapper(parent(current)[ts.d[keys...][vals...]])
    end
    return current
end

function Base.setindex!(w::TensorStoreWrapper, v, indices...; kwargs...)
    @debug "setindex!" indices kwargs
    target = w
    if !isempty(indices) || !isempty(kwargs)
        # For setindex!, we first get the sub-view we want to write into
        target = getindex(w, indices...; kwargs...)
    end
    parent(target)[()] = v
    return v
end

Base.size(w::TensorStoreWrapper) = pyconvert(Tuple, parent(w).shape)
function Base.size(w::TensorStoreWrapper, d::Integer)
    d < 1 && throw(ArgumentError("dimension must be ≥ 1"))
    di = Base.to_index(d)
    return di <= ndims(w) ? size(w)[di] : 1
end
Base.ndims(w::TensorStoreWrapper) = pyconvert(Int, parent(w).rank)

const TS_TYPE_MAP = Dict(
    "int8" => Int8,
    "int16" => Int16,
    "int32" => Int32,
    "int64" => Int64,
    "uint8" => UInt8,
    "uint16" => UInt16,
    "uint32" => UInt32,
    "uint64" => UInt64,
    "float16" => Float16,
    "float32" => Float32,
    "float64" => Float64,
    "bool" => Bool,
    "string" => String,
)

function Base.eltype(w::TensorStoreWrapper)
    dtype = parent(w).dtype
    if hasproperty(dtype, :numpy_dtype)
        try
            return pyconvert(Type, dtype.numpy_dtype)
        catch
        end
    end
    name = pyconvert(String, dtype.name)
    return get(TS_TYPE_MAP, name, Any)
end

function Base.axes(w::TensorStoreWrapper)
    domain = parent(w).domain
    rank = ndims(w)
    min_indices = pyconvert(Vector{Int}, domain.inclusive_min)
    max_indices = pyconvert(Vector{Int}, domain.exclusive_max)
    return Tuple((min_indices[i]+1):max_indices[i] for i in 1:rank)
end

function Base.show(io::IO, w::TensorStoreWrapper)
    print(io, "TensorStore(", eltype(w), ", rank=", ndims(w), ", shape=", size(w), ")")
end

# Domain operations

"""
    translate_by(w::TensorStoreWrapper, offsets...) -> TensorStoreWrapper

Shift the coordinate system of the TensorStore by the given relative offsets.
Offsets are relative to the current coordinate system.
"""
function translate_by(w::TensorStoreWrapper, offsets...)
    # translate_by offsets are relative, so we don't use tsindex (no -1)
    TensorStoreWrapper(parent(w).translate_by[offsets...])
end

"""
    translate_to(w::TensorStoreWrapper, coords...) -> TensorStoreWrapper

Shift the coordinate system of the TensorStore so that its origin starts at the specified coordinates.
Coordinates are 1-based Julia indices.
"""
function translate_to(w::TensorStoreWrapper, offsets...)
    # translate_to offsets are absolute 1-based Julia indices, so we use tsindex
    TensorStoreWrapper(parent(w).translate_to[map(tsindex, offsets)...])
end

"""
    label(w::TensorStoreWrapper, labels...) -> TensorStoreWrapper

Assign new labels to the dimensions of the TensorStore.
"""
function label(w::TensorStoreWrapper, labels...)
    # TensorStore's label property is indexed
    TensorStoreWrapper(parent(w).label[labels...])
end

# IndexDomainWrapper methods
Base.size(w::IndexDomainWrapper) = pyconvert(Tuple, parent(w).shape)
function Base.size(w::IndexDomainWrapper, d::Integer)
    d < 1 && throw(ArgumentError("dimension must be ≥ 1"))
    di = Base.to_index(d)
    return d <= ndims(w) ? size(w)[di] : 1
end
Base.ndims(w::IndexDomainWrapper) = pyconvert(Int, parent(w).rank)
function Base.axes(w::IndexDomainWrapper)
    rank = ndims(w)
    min_indices = pyconvert(Vector{Int}, parent(w).inclusive_min)
    max_indices = pyconvert(Vector{Int}, parent(w).exclusive_max)
    return Tuple((min_indices[i]+1):max_indices[i] for i in 1:rank)
end

"""
    labels(w::IndexDomainWrapper) -> Vector{String}

Return the labels of the dimensions in the index domain.
"""
labels(w::IndexDomainWrapper) = pyconvert(Vector{String}, parent(w).labels)

function Base.show(io::IO, w::IndexDomainWrapper)
    print(io, "IndexDomain(rank=", ndims(w), ", shape=", size(w), ", labels=", labels(w), ")")
end

# TransactionWrapper methods

"""
    commit_sync(txn::TransactionWrapper)

Commit the transaction synchronously and wait for completion.
"""
commit_sync(txn::TransactionWrapper) = parent(txn).commit_sync()

"""
    commit_async(txn::TransactionWrapper) -> FutureWrapper{ReturnsNothing}

Start committing the transaction asynchronously.
"""
commit_async(txn::TransactionWrapper) = FutureWrapper{ReturnsNothing}(parent(txn).commit_async())

"""
    abort(txn::TransactionWrapper)

Abort the transaction, discarding any changes.
"""
abort(txn::TransactionWrapper) = parent(txn).abort()

function Base.show(io::IO, txn::TransactionWrapper)
    print(io, "Transaction(...)")
end

# Spec, Schema, ChunkLayout methods
function Base.show(io::IO, s::SpecWrapper)
    print(io, "Spec(", pyconvert(Any, parent(s).to_json()), ")")
end

function Base.show(io::IO, s::SchemaWrapper)
    print(io, "Schema(dtype=", pyconvert(Any, parent(s).dtype.name), ", rank=", pyconvert(Any, parent(s).rank), ")")
end

function Base.getproperty(s::SchemaWrapper, sym::Symbol)
    if sym == :domain
        return IndexDomainWrapper(parent(s).domain)
    elseif sym == :chunk_layout
        return ChunkLayoutWrapper(parent(s).chunk_layout)
    else
        return pyconvert(Any, getproperty(parent(s), sym))
    end
end

function Base.show(io::IO, l::ChunkLayoutWrapper)
    print(io, "ChunkLayout(...)")
end

"""
    transaction(f::Function)

Run a function `f(txn)` within a transaction. 
The transaction is automatically committed if `f` returns normally, 
and aborted if an exception is thrown.
"""
function transaction(f::Function)
    txn = TransactionWrapper(Python.transaction())
    try
        result = f(txn)
        commit_sync(txn)
        return result
    catch
        abort(txn)
        rethrow()
    end
end

# ContextWrapper methods
function Base.getproperty(ctx::ContextWrapper, sym::Symbol)
    if sym == :spec
        return ContextSpecWrapper(parent(ctx).spec)
    else
        return pyconvert(Any, getproperty(parent(ctx), sym))
    end
end

function Base.show(io::IO, ctx::ContextWrapper)
    print(io, "Context(...)")
end

function Base.show(io::IO, s::ContextSpecWrapper)
    print(io, "ContextSpec(", pyconvert(Any, parent(s).to_json()), ")")
end

# Convert indices into TensorStoreWrapper into Python indexes
tsindex(x) = x
tsindex(::Colon) = pyslice(nothing)
tsindex(i::Integer) = i-1
tsindex(r::UnitRange) = pyslice(first(r)-1, last(r))
tsindex(r::AbstractRange) = pyslice(first(r)-1, last(r), step(r))
tsindex(ci::CartesianIndex) = Tuple(ci - CartesianIndex(1,1))
