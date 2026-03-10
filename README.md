# PyTensorStore

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaIO.github.io/PyTensorStore.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaIO.github.io/PyTensorStore.jl/dev/)
[![Build Status](https://github.com/JuliaIO/PyTensorStore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaIO/PyTensorStore.jl/actions/workflows/CI.yml?query=branch%3Amain)

PyTensorStore.jl provides a wrapper around the Python package `tensorstore`. A future TensorStore.jl may wrap the C++ API directly.

This package is being primarily developed to test Zarr.jl.

## Features

- **Julia Array Interface**: Support for `size`, `ndims`, `eltype`, `axes`, and 1-based indexing.
- **Read/Write Operations**: Synchronous and asynchronous read/write support.
- **Indexing & Domain Manipulation**:
    - Labeled indexing (e.g., `w[lat=1:10, lon=1:5]`).
    - Domain operations: `translate_by`, `translate_to`, and `label`.
- **Transactions**: Atomic multi-write operations with an idiomatic Julia context manager.
- **Specs & Schemas**: Programmatic access to TensorStore `Spec`, `Schema`, and `ChunkLayout`.

## Usage

```julia-repl
julia> using PyTensorStore
Precompiling PyTensorStore...
  1 dependency successfully precompiled in 2 seconds. 50 already precompiled.

julia> d = Dict(
           "driver" => "n5",
           "kvstore" => Dict(
               "driver" => "file",
               "path" => "tmp/dataset/",
           ),
           "metadata" => Dict(
               "compression" => Dict(
                   "type" => "gzip"
               ),
               "dataType" => "uint32",
               "dimensions" => [1000, 20000],
               "blockSize" => [100, 100],
           ),
           "create" => true,
           "delete_existing" => true
       )
Dict{String, Any} with 5 entries:
  "driver"          => "n5"
  "create"          => true
  "kvstore"         => Dict("driver"=>"file", "path"=>"tmp/dataset/")
  "metadata"        => Dict{String, Any}("blockSize"=>[100, 100], "dataType"=>"…
  "delete_existing" => true

julia> A = PyTensorStore.open(d).result()
PyTensorStore.TensorStoreWrapper(<py TensorStore({
  'context': {
    'cache_pool': {},
    'data_copy_concurrency': {},
    'file_io_concurrency': {},
    'file_io_sync': True,
  },
  'driver': 'n5',
  'dtype': 'uint32',
  'kvstore': {'driver': 'file', 'path': 'tmp/dataset/'},
  'metadata': {
    'blockSize': [100, 100],
    'compression': {'level': -1, 'type': 'gzip', 'useZlib': False},
    'dataType': 'uint32',
    'dimensions': [1000, 20000],
  },
  'transform': {
    'input_exclusive_max': [[1000], [20000]],
    'input_inclusive_min': [0, 0],
  },
})>)

julia> A[1:100, 1:100]
PyTensorStore.TensorStoreWrapper(<py TensorStore({
  'context': {
    'cache_pool': {},
    'data_copy_concurrency': {},
    'file_io_concurrency': {},
    'file_io_sync': True,
  },
  'driver': 'n5',
  'dtype': 'uint32',
  'kvstore': {'driver': 'file', 'path': 'tmp/dataset/'},
  'metadata': {
    'blockSize': [100, 100],
    'compression': {'level': -1, 'type': 'gzip', 'useZlib': False},
    'dataType': 'uint32',
    'dimensions': [1000, 20000],
  },
  'transform': {
    'input_exclusive_max': [100, 100],
    'input_inclusive_min': [0, 0],
  },
})>)

julia> A[1:100, 1:100].write(ones(UInt32, 100, 100)*UInt32(5)).result()
Python: None

julia> A[1:100, 1:100].read().result()
100×100 PyArray{UInt32, 2}:
 0x00000005  0x00000005  0x00000005  …  0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005  …  0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005  …  0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
          ⋮                          ⋱                          
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005  …  0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005  …  0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005
 0x00000005  0x00000005  0x00000005     0x00000005  0x00000005  0x00000005

julia> A[1,1].write(9).result()
Python: None

julia> A[1,1].read().result()
0-dimensional PyArray{UInt32, 0}:
0x00000009
```

### Advanced Features

#### Labeled Indexing

If your TensorStore has dimension labels, you can index using keywords:

```julia
# Open with labels in schema
spec["schema"] = Dict("domain" => Dict("labels" => ["x", "y"]))
w = PyTensorStore.open(spec).result()

# Index by dimension label
sub_w = w[x=1:5, y=10:15]
```

#### Transactions

Atomic multi-write operations can be performed using an idiomatic Julia context manager:

```julia
PyTensorStore.transaction() do txn
    w_txn = w.with_transaction(txn)
    w_txn[1, 1] = 42
    w_txn[2, 2] = 100
    # Changes are committed automatically when the block exits successfully.
    # If an error occurs, the transaction is aborted.
end
```

#### Domain Manipulation

```julia
# Shift the domain coordinate system
shifted_w = PyTensorStore.translate_by(w, 10, 20)

# Move the origin to a specific coordinate
centered_w = PyTensorStore.translate_to(w, 1, 1)

# Re-label dimensions
labeled_w = PyTensorStore.label(w, "lat", "lon")
```
