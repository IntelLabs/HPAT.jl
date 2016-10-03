using Base.Test

run(`julia mpi_test.jl`)
run(`mpirun -np 2 julia mpi_test.jl`)
run(`julia source_test.jl`)
run(`julia sink_test.jl`)
run(`mpirun -np 2 julia sink_test.jl`)
run(`julia q26_test.jl`)
run(`julia q05_test.jl`)
run(`mpirun -np 2 julia cumsum_test.jl`)

# commented out since MKL BLACS and SCALAPACK libraries are not linked automatically currently
#run(`mpirun -np 4 julia --depwarn=no 2d_test.jl`)
# mpiicpc -O0 -g -std=c++11  -DMKL_ILP64 -qopenmp -I${MKLROOT}/include -I/usr/local/hdf5/include  -g -fpic  -o main1 main1.cc -L/usr/local/hdf5/lib -lhdf5  -lm  -Wl,--start-group -L${MKLROOT}/lib/intel64 -lmkl_scalapack_ilp64 -lmkl_intel_ilp64 -lmkl_core -lmkl_intel_thread -lmkl_blacs_intelmpi_ilp64 -lpthread -lm -ldl -Wl,--end-group
# mpirun -np 4 ./main1
