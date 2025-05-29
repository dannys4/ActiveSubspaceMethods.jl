using ActiveSubspaceMethods
using Documenter

DocMeta.setdocmeta!(
    ActiveSubspaceMethods, :DocTestSetup, :(using ActiveSubspaceMethods); recursive=true
)

makedocs(;
    modules=[ActiveSubspaceMethods],
    authors="Daniel Sharp <dannys4@mit.edu> and contributors",
    sitename="ActiveSubspaceMethods.jl",
    format=Documenter.HTML(;
        canonical="https://dannys4.github.io/ActiveSubspaceMethods.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/dannys4/ActiveSubspaceMethods.jl", devbranch="main")
