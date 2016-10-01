using HPAT

#HPAT.CaptureAPI.set_debug_level(3)
#HPAT.DomainPass.set_debug_level(3)
#HPAT.DistributedPass.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)
#using CompilerTools
#CompilerTools.OptFramework.set_debug_level(3)

@acc hpat function test1(file_name)
    t1 = DataSource(DataTable{:userid=Int64, :val2=Float64}, HDF5, file_name)
    avg = stencil(x->(x[-1]+2*x[0]+x[1])/4.0, t1[:val2])
    return avg
end

using HDF5
using MPI
if MPI.Comm_rank(MPI.COMM_WORLD)==0 
    h5write("test1_1.hdf5", "/userid", [1, 2, 3, 1, 2])
    h5write("test1_1.hdf5", "/val2", [1.2, 2.1, 3.1, 3.2, 1.9])
end
a = test1("test1_1.hdf5")
println(a)


using Base.Test
@test_approx_eq a [1.2, 2.125, 2.875, 2.85, 1.9]

if MPI.Comm_rank(MPI.COMM_WORLD)==0 
    rm("test1_1.hdf5")
end
