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
    return t3[:val3]
end


a,b = test1("test1_1.hdf5","test1_2.hdf5")

println(a)
println(b)

