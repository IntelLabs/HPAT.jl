using Base.Test

run(`mpirun -np 2 julia mpi-test.jl`)
run(`mpirun -np 2 julia sink_test.jl`)
