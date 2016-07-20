using HPAT

#HPAT.CaptureAPI.set_debug_level(3)
ParallelAccelerator.set_debug_level(3)
using CompilerTools
CompilerTools.OptFramework.set_debug_level(3)

@acc hpat function test1(file_name, file_name2)
    t1 = DataSource(DataTable{:userid=Int64, :val2=Float64}, HDF5, file_name)
    t2 = DataSource(DataTable{:userid=Int64, :val3=Float64}, HDF5, file_name2)
    t3 = join(t1,t2, :userid==:userid, :userid)
    t4 = aggregate(t3, :userid, :sumo2 = sum(:val2==1.1), :size_val3 = length(:val3))
    return t4[:sumo2], t4[:size_val3]
end


a,b = test1("test1_1.hdf5","test1_2.hdf5")

println(a)
println(b)

