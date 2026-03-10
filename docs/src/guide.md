# User Guide

This guide explains how to use `PyTensorStore.jl` to read and write multi-dimensional arrays.

## Opening a Store

To open a TensorStore, use the `PyTensorStore.open` function with a configuration dictionary (Spec). This returns a `FutureWrapper`, and you can use `.result()` to wait for the store to open.

```julia
using PyTensorStore

spec = Dict(
    "driver" => "zarr",
    "kvstore" => "memory://test_array",
    "dtype" => "int32",
    "metadata" => Dict("shape" => [100, 200]),
    "create" => true
)

w = PyTensorStore.open(spec).result()
```

## Basic Operations

`PyTensorStore.jl` supports standard Julia 1-based indexing:

```julia
# Write data (synchronously via setindex!)
w[1:10, 1:10] = rand(Int32, 10, 10)

# Read data (returns a FutureWrapper)
read_future = w[1:5, 1:5].read()
data = read_future.result() # Returns a PyArray
```

## Labeled Indexing

TensorStore supports labeling dimensions, allowing you to index by name:

```julia
# Create with labels
spec["schema"] = Dict("domain" => Dict("labels" => ["lat", "lon"]))
w = PyTensorStore.open(spec).result()

# Use keywords for labeled dimensions
sub_w = w[lat=1:5, lon=10:15]
```

## Transactions

Atomic multi-write operations can be performed by using a transaction block. All changes made within the block are committed together when the block finishes.

```julia
PyTensorStore.transaction() do txn
    w_txn = w.with_transaction(txn)
    w_txn[1, 1] = 42
    w_txn[2, 2] = 100
end
```

## Domain Manipulation

You can transform the coordinate system of a TensorStore without moving the underlying data.

*   `translate_by(w, offsets...)`: Shift the coordinate system by a relative amount.
*   `translate_to(w, coords...)`: Shift the coordinate system so that the origin is at the specified coordinates (1-based Julia coordinates).
*   `label(w, labels...)`: Assign new labels to the dimensions.

```julia
shifted = translate_by(w, 10, 20)
@assert axes(shifted) == (11:110, 21:220)
```
