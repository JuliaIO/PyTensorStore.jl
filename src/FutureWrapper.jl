"""
    FutureWrapper{Result}(parent::Py)

A wrapper around a Python `tensorstore.Future` object.
`Result` is the Julia type that the future will resolve to when `.result()` is called.
"""
struct FutureWrapper{Result}
    parent::Py
end

Base.parent(w::FutureWrapper) = getfield(w, :parent)
PythonCall.Py(w::FutureWrapper) = parent(w)

function Base.getproperty(w::FutureWrapper, s::Symbol)
    s == :result ? (x...,) -> result(w, x...) : 
                   getproperty(parent(w), s)
end

"""
    result(w::FutureWrapper{Result}) -> Result

Wait for the future to complete and return the result, converted to the Julia type `Result`.
"""
result(w::FutureWrapper{Result}, x...) where Result = Result(parent(w).result(x...))
pyarray_nocopy(x) = PyArray(x; copy=false)

function Base.show(io::IO, w::FutureWrapper{Result}) where Result
    print(io, "FutureWrapper{", Result, "}(...)")
end
