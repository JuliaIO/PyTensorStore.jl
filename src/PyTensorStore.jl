module PyTensorStore

using PythonCall

public open, transaction, context

include("Python/Python.jl")
include("TensorStoreWrapper.jl")
include("FutureWrapper.jl")

"""
    open(spec; kwargs...) -> FutureWrapper{TensorStoreWrapper}

Open a TensorStore from a configuration dictionary or Spec.
Returns a `FutureWrapper` that resolves to a `TensorStoreWrapper`.

Common keyword arguments:
- `transaction`: A `TransactionWrapper` to use for this open operation.
- `context`: A `ContextWrapper` or dictionary for shared resources.
- `create`: Boolean, whether to create the store if it doesn't exist.
- `open`: Boolean, whether to open an existing store.
- `delete_existing`: Boolean, whether to delete an existing store.
"""
open(spec; kwargs...) = FutureWrapper{TensorStoreWrapper}(Python.open(spec; kwargs...))

"""
    transaction() -> TransactionWrapper

Create a new TensorStore transaction.
"""
transaction() = TransactionWrapper(Python.transaction())

"""
    context(spec::Dict) -> ContextWrapper
    context() -> ContextWrapper

Create a TensorStore context for shared resources like cache pools.
"""
context(ctx...) = ContextWrapper(Python.context(ctx...))

end
