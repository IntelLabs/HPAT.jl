module PushFilterTest1

using HPAT

#HPAT.CaptureAPI.set_debug_level(3)
HPAT.DomainPass.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)
#using CompilerTools
#CompilerTools.OptFramework.set_debug_level(3)

@acc hpat function test1(file_name, file_name2)
    t1 = DataSource(DataTable{:userid=Int64, :val2=Float64}, HDF5, file_name)
    t2 = DataSource(DataTable{:userid=Int64, :val3=Float64}, HDF5, file_name2)
    t3 = join(t1,t2, :userid==:userid, :userid)
    t4 = t3[:val2>2.0]
    return t4[:userid], t4[:val3]
end

using HDF5
using MPI
using Base.Test

function main()

if MPI.Comm_rank(MPI.COMM_WORLD)==0 
    h5write("test1_1.hdf5", "/userid", [1,2,3,1,2])
    h5write("test1_1.hdf5", "/val2", [1.1,2.1,3.1,3.2,1.9])
    h5write("test1_2.hdf5", "/userid", [1,3])
    h5write("test1_2.hdf5", "/val3", [7.1,8.3])
end
a,b = test1("test1_1.hdf5","test1_2.hdf5")
println(a)
println(b)


@test a==[1,3]
@test_approx_eq b [7.1,8.3]

if MPI.Comm_rank(MPI.COMM_WORLD)==0 
    rm("test1_1.hdf5")
    rm("test1_2.hdf5")
end

end

end

PushFilterTest1.main()
