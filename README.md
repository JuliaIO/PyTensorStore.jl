# PyTensorStore

[![Build Status](https://github.com/mkitti/PyTensorStore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mkitti/PyTensorStore.jl/actions/workflows/CI.yml?query=branch%3Amain)

PyTensorStore.jl provides a wrapper around the Python package `tensorstore`. A future TensorStore.jl may wrap the C++ API directly.

This package is being primarily developed to test Zarr.jl.

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
