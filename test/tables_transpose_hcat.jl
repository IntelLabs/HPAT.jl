module TablesCatTest

using HPAT

#HPAT.CaptureAPI.set_debug_level(3)
#HPAT.DomainPass.set_debug_level(3)
#HPAT.DistributedPass.set_debug_level(3)
#ParallelAccelerator.set_debug_level(3)
#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.CGen.setCreateMain(true)
#ParallelAccelerator.CGen.set_debug_level(3)
#using CompilerTools
#CompilerTools.OptFramework.set_debug_level(3)
#CompilerTools.CFGs.set_debug_level(3)

@acc hpat function test1(file_name)
    t1 = DataSource(DataTable{:userid=Int64, :val2=Float64}, HDF5, file_name)
    t2 = aggregate(t1,:uid = :userid, :ss=sum(:val2))
    t3 = aggregate(t1, :uid = :userid, :ss=sum(:val2>2.0))
    data = transpose(typed_hcat(Float64, t3[:uid], t3[:ss], t2[:ss]))
    return sum(data)
end

using HDF5
using MPI
using Base.Test

function main()

    if MPI.Comm_rank(MPI.COMM_WORLD)==0 
        h5write("test1_1.hdf5", "/userid", [1, 2, 3, 1, 2])
        h5write("test1_1.hdf5", "/val2", [1.1, 2.1, 3.1, 3.2, 1.9])
    end
    s = test1("test1_1.hdf5")
    println(s)

    @test_approx_eq s 20.4 

    if MPI.Comm_rank(MPI.COMM_WORLD)==0 
        rm("test1_1.hdf5")
    end

end

end

TablesCatTest.main()
