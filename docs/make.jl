using BlockSparseMatrices
using Documenter

DocMeta.setdocmeta!(
    ButterflyFactorization, :DocTestSetup, :(using ButterflyFactorization); recursive=true
)

makedocs(;
    modules=[ButterflyFactorization],
    authors="Joshua M. Tetzner <joshua.tetzner@uni-rostock.de> and contributors",
    sitename="ButterflyFactorization.jl",
    format=Documenter.HTML(;
        canonical="https://joshuatetzner.github.io/ButterflyFactorization.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Introduction" => "index.md",
        "Contributing" => "contributing.md",
        "API Reference" => "apiref.md",
    ],
)

deploydocs(;
    repo="github.com/joshuatetzner/ButterflyFactorization.jl",
    target="build",
    devbranch="main",
    push_preview=true,
    forcepush=true,
    versions=["stable" => "v^", "dev" => "dev"],
)
