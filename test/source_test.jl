module SourceTest

using HPAT
#HPAT.DomainPass.set_debug_level(3)
#HPAT.DataTablePass.set_debug_level(3)

@acc hpat function stest(file_name)
    A = DataSource(Vector{Float64},HDF5,"/A", file_name)
    return sum(A)
end

@acc hpat function stest2(file_name)
    A = DataSource(Matrix{Float64},HDF5,"/A2", file_name)
    return sum(A)
end

using HDF5
using MPI
using Base.Test

function main()

    if MPI.Comm_rank(MPI.COMM_WORLD)==0 
        h5write("A.hdf5", "/A", [1.0,2.0,3.0,4.0])
        h5write("A.hdf5", "/A2", [1.0 2.0 3.0; 4.0 5.0 6.0])
    end

    @test_approx_eq SourceTest.stest("A.hdf5") 10.0
    @test_approx_eq SourceTest.stest2("A.hdf5") 21.0
    if MPI.Comm_rank(MPI.COMM_WORLD)==0 rm("A.hdf5") end
end

end

SourceTest.main()
