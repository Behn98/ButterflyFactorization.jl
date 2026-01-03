using Test, TestItems, TestItemRunner

@testitem "ButterflyFactorization" begin end

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(ButterflyFactorization; unbound_args=false)
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter, ButterflyFactorization
    @test JuliaFormatter.format(pkgdir(ButterflyFactorization), overwrite=false)
end

@run_package_tests verbose = true
