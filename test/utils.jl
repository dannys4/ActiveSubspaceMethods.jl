@testset "Create Ur" begin
    d, r = 10, 7
    rng = Xoshiro(28402)
    A = randn(rng, d, d)
    Q = collect(qr(A).Q)
    Qr = Q[:,1:r]
    Qr_create = create_Ur(Q[:,r+1:end])
    @test norm((I - Qr*Qr')*Qr_create)/norm(Qr) < 1e-14
end

@testset "Estimating variance" begin
    
end