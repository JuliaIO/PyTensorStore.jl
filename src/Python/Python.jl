module Python
    using PythonCall

    pyts() = @pyconst(pyimport("tensorstore"))

    open(spec::Py; kwargs...) = pyts().open(spec; kwargs...)
    open(spec::AbstractDict) = pyts().open(pydict_recursive(spec))
    pydict_recursive(x) = x
    pydict_recursive(x::AbstractArray) = pylist(x)
    function pydict_recursive(x::AbstractDict)
        return pydict(Dict(
            string.(keys(x)) .=> pydict_recursive.(values(x))
        ))
    end
end
