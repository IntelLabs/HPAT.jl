using Base.Test

println("testing MPI...")
run(`julia mpi_test.jl`)
run(`mpirun -np 2 julia mpi_test.jl`)
println("done testing MPI.")

println("testing data source/sink...")
include("source_test.jl")
include("sink_test.jl")
run(`mpirun -np 2 julia sink_test.jl`)
println("done testing data source/sink.")

println("testing filter...")
include("filter_test1.jl")
println("done testing filter.")

println("testing join...")
include("join_test1.jl")
println("done testing join.")

println("testing aggregate...")
include("aggregate_test1.jl")
include("aggregate_rename_test1.jl")
println("done testing aggregate...")

println("testing push filter up...")
include("push_filter_test1.jl")
println("done testing push filter up...")

include("q26_test.jl")
include("q05_test.jl")
include("length_unique_test1.jl")
include("tables_cat.jl")
include("stencil_test1.jl")
run(`mpirun -np 2 julia stencil_test2.jl`)
run(`mpirun -np 2 julia cumsum_test.jl`)
include("index_test.jl")

# commented out since MKL BLACS and SCALAPACK libraries are not linked automatically currently
#run(`mpirun -np 4 julia --depwarn=no 2d_test.jl`)
# mpiicpc -O0 -g -std=c++11  -DMKL_ILP64 -qopenmp -I${MKLROOT}/include -I/usr/local/hdf5/include  -g -fpic  -o main1 main1.cc -L/usr/local/hdf5/lib -lhdf5  -lm  -Wl,--start-group -L${MKLROOT}/lib/intel64 -lmkl_scalapack_ilp64 -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -lmkl_blacs_intelmpi_ilp64 -lpthread -lm -ldl -Wl,--end-group
# mpirun -np 4 ./main1
