using HPAT

#HPAT.DomainPass.set_debug_level(3)
#HPAT.DistributedPass.set_debug_level(3)

@acc hpat function cumsum_test1(n)
    A = ones(n)
    B = cumsum(A)
    return sum(B)
end

using Base.Test
@test_approx_eq cumsum_test1(5) 15.0


