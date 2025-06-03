module PyTensorStore

using PythonCall

public open

include("Python/Python.jl")
include("TensorStoreWrapper.jl")
include("FutureWrapper.jl")

open(spec; kwargs...) = FutureWrapper{TensorStoreWrapper}(Python.open(spec; kwargs...))

end
