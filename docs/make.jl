using PyTensorStore
using Documenter

DocMeta.setdocmeta!(PyTensorStore, :DocTestSetup, :(using PyTensorStore); recursive=true)

makedocs(;
    modules=[PyTensorStore],
    authors="Mark Kittisopikul <markkitt@gmail.com> and contributors",
    sitename="PyTensorStore.jl",
    format=Documenter.HTML(;
        canonical="https://mkitti.github.io/PyTensorStore.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "User Guide" => "guide.md",
        "API Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/mkitti/PyTensorStore.jl",
    devbranch="main",
)
