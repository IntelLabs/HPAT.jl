using HPAT 
using MPI

@acc hpat function calcPi(n::Int64)
    x = rand(n) .* 2.0 .- 1.0
    y = rand(n) .* 2.0 .- 1.0
    return 4.0*sum(x.^2 .+ y.^2 .< 1.0)/n
end

n = 10^9

if (length(ARGS) > 0)
	n = parse(Int, ARGS[1])
end

rank = MPI.Comm_rank(MPI.COMM_WORLD)
pes = MPI.Comm_size(MPI.COMM_WORLD)


p1 = calcPi(1000)
MPI.Barrier(MPI.COMM_WORLD)
t1 = time_ns()
p2 = calcPi(n)
t2 = time_ns()

if rank==0
	println("nodes: ", pes, "\npi exec time: ", (t2-t1)/1.0e9, "\npi: ", p2)
end

MPI.Barrier(MPI.COMM_WORLD)
