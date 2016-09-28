using Base.Test

run(`julia mpi_test.jl`)
run(`mpirun -np 2 julia mpi_test.jl`)
run(`julia source_test.jl`)
run(`julia sink_test.jl`)
run(`mpirun -np 2 julia sink_test.jl`)
run(`julia q26_test.jl`)
run(`julia q05_test.jl`)
run(`mpirun -np 2 julia cumsum_test.jl`)
#run(`mpirun -np 4 julia --depwarn=no 2d_test.jl`)
