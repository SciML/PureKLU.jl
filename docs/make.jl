using Documenter
using PureKLU
using SparseArrays

makedocs(
    sitename = "PureKLU.jl",
    authors = "Chris Rackauckas and contributors",
    modules = [PureKLU],
    clean = true,
    doctest = true,
    checkdocs = :exports,
    format = Documenter.HTML(
        canonical = "https://docs.sciml.ai/PureKLU/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Developer API" => "developer_api.md",
    ]
)

deploydocs(
    repo = "github.com/SciML/PureKLU.jl.git";
    push_preview = true
)
