using Base.Test

run(`julia --depwarn=no mpi_test.jl`)
run(`mpirun -np 2 julia --depwarn=no mpi_test.jl`)
run(`julia --depwarn=no sink_test.jl`)
run(`mpirun -np 2 julia --depwarn=no sink_test.jl`)
run(`julia --depwarn=no q26_test.jl`)
run(`julia --depwarn=no q05_test.jl`)
#run(`mpirun -np 4 julia --depwarn=no 2d_test.jl`)
