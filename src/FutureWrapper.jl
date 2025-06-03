struct FutureWrapper{Result}
    parent::Py
end
Base.parent(w::FutureWrapper) = getfield(w, :parent)
PythonCall.Py(w::FutureWrapper) = parent(w)
function Base.getproperty(w::FutureWrapper, s::Symbol)
    s == :result ? (x...,) -> result(w, x...) : 
                   getproperty(parent(w), s)
end
result(w::FutureWrapper{Result}, x...) where Result = Result(parent(w).result(x...))
pyarray_nocopy(x) = PyArray(x; copy=false)
