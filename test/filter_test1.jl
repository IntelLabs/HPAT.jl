module FilterTest1

using HPAT

#HPAT.CaptureAPI.set_debug_level(3)
#HPAT.DomainPass.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)
#ParallelAccelerator.CGen.setCreateMain(true)
#using CompilerTools
#CompilerTools.OptFramework.set_debug_level(3)

@acc hpat function test1(file_name)
    t1 = DataSource(DataTable{:userid=Int64, :val2=Float64}, HDF5, file_name)
    t1 = t1[:userid>2]
    return t1[:userid], t1[:val2]
end

using HDF5
using MPI
using Base.Test

function main()

    if MPI.Comm_rank(MPI.COMM_WORLD)==0 
        h5write("test1_1.hdf5", "/userid", [1, 2, 3, 1, 2])
        h5write("test1_1.hdf5", "/val2", [1.1, 2.1, 3.1, 3.2, 1.9])
    end
    a,b = test1("test1_1.hdf5")
    println(a)
    println(b)


    @test a==[3]
    @test_approx_eq b [3.1]

    if MPI.Comm_rank(MPI.COMM_WORLD)==0 
        rm("test1_1.hdf5")
    end

end

end

FilterTest1.main()
