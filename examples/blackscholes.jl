using HPAT 
using MPI

@acc @inline function cndf2( In::Array{Float64,1} )
    Out = 0.5 .+ 0.5 .* erf(0.707106781 .* In);
    return Out;
end

@acc hpat function blackscholes(iterations) 

    sptprice   = Float64[ 42.0 for i=1:iterations ]
    strike     = Float64[ 40.0 + (i / iterations) for i=1:iterations ]
    rate       = Float64[ 0.5 for i=1:iterations ]
    volatility = Float64[ 0.2 for i=1:iterations ]
    time       = Float64[ 0.5 for i=1:iterations ]
    
    logterm = log10(sptprice ./ strike);
    powterm = .5 .* volatility .* volatility;
    den = volatility .* sqrt(time);
    d1 = (((rate .+ powterm) .* time) .+ logterm) ./ den;
    d2 = d1 .- den;		       
    NofXd1 = cndf2(d1);
    NofXd2 = cndf2(d2);	
    futureValue = strike .* exp(- rate .* time);
    c1 = futureValue .* NofXd2
    Call = sptprice .* NofXd1 .- c1;
    Put  = Call .- futureValue .+ sptprice;
    sum(Put)
end

n = 10^8

if (length(ARGS) > 0)
	n = parse(Int, ARGS[1])
end

checksum = blackscholes(n)

if MPI.Comm_rank(MPI.COMM_WORLD)==0	println("checksum: ", checksum) end

