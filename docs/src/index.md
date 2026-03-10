```@meta
CurrentModule = PyTensorStore
```

# PyTensorStore

[PyTensorStore.jl](https://github.com/JuliaIO/PyTensorStore.jl) provides a Julia wrapper around the Google [TensorStore](https://google.github.io/tensorstore/) Python API. TensorStore is a library for efficiently reading and writing large multi-dimensional arrays, supporting various storage backends like Zarr, N5, and Google Cloud Storage.

## Contents

```@contents
Pages = ["index.md", "guide.md", "reference.md"]
Depth = 2
```

## Key Features

*   **Familiar Interface**: Implements much of the Julia `AbstractArray` interface, including 1-based indexing, `size`, `axes`, and `eltype`.
*   **Asynchronous Support**: Built on TensorStore's asynchronous core, allowing for non-blocking I/O.
*   **Powerful Indexing**: Support for labeled indexing, coordinate translation, and complex domain transformations.
*   **Transactions**: Support for atomic multi-write operations.

## Installation

```julia
using Pkg
Pkg.add("PyTensorStore")
```

Note: This package requires a working Python installation with the `tensorstore` package. `CondaPkg.jl` is used to manage this dependency automatically.
