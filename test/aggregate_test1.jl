using HPAT

#HPAT.CaptureAPI.set_debug_level(3)
#HPAT.DomainPass.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)
#ParallelAccelerator.CGen.setCreateMain(true)
#using CompilerTools
#CompilerTools.OptFramework.set_debug_level(3)

@acc hpat function test1(file_name)
    t1 = DataSource(DataTable{:userid=Int64, :val2=Float64}, HDF5, file_name)
    t2 = aggregate(t1,:userid, :ss=sum(:val2))
    #t2 = aggregate(t1,:userid, :ss=sum(:val2), :aa=length(:val2>2.0))
    #t2 = aggregate(t1,:userid, :ss=sum(:val2), :aa=length(:val2>2.0), :bb=length(:val2==1.0))
    return t2[:userid], t2[:ss]
end

@acc hpat function test2(file_name)
    t1 = DataSource(DataTable{:userid=Int64, :val2=Float64}, HDF5, file_name)
    #t2 = aggregate(t1,:userid, :ss=sum(:val2))
    #t2 = aggregate(t1,:userid, :ss=sum(:val2), :aa=length(:val2>2.0))
    t2 = aggregate(t1,:userid, :ss=sum(:val2), :aa=length(:val2>2.0), :bb=length(:val2==1.0))
    return t2[:userid], t2[:ss]
end

using HDF5
using MPI
if MPI.Comm_rank(MPI.COMM_WORLD)==0 
    h5write("test1_1.hdf5", "/userid", [1, 2, 3, 1, 2])
    h5write("test1_1.hdf5", "/val2", [1.1, 2.1, 3.1, 3.2, 1.9])
end
a,b = test1("test1_1.hdf5")
println(a)
println(b)

using Base.Test
@test a==[1,2,3]
@test_approx_eq b [4.3, 4.0, 3.1]

a,b = test2("test1_1.hdf5")
@test a==[1,2,3]
@test_approx_eq b [4.3, 4.0, 3.1]

if MPI.Comm_rank(MPI.COMM_WORLD)==0 
    rm("test1_1.hdf5")
end
