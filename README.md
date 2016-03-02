# HPAT

[![Build Status](https://travis-ci.org/IntelLabs/HPAT.jl.svg?branch=master)](https://travis-ci.org/IntelLabs/HPAT.jl)

*High Performance Analytics Toolkit (HPAT)* is a Julia-based framework for big data analytics on clusters that
is both easy to use and extremely fast; it is orders of magnitude faster than alternatives 
like [Apache Spark&trade;](http://spark.apache.org/).  

HPAT automatically parallelizes analytics tasks written in Julia, generates efficient MPI/C++ code, 
and uses existing high performance libraries such as [HDF5](https://www.hdfgroup.org/HDF5/)
and [Intel&reg; Data Analytics Acceleration Library (Intel&reg; DAAL)](https://software.intel.com/en-us/daal). 
HPAT is based on [ParallelAccelerator](https://github.com/IntelLabs/ParallelAccelerator.jl) 
and [CompilerTools](https://github.com/IntelLabs/CompilerTools.jl) packages. 

HPAT is in early development and therefore feedback is highly appreciated.

## Quick Start
```shell
$ julia -e 'Pkg.add("HPAT")'
$ mpirun -np 2 julia ~/.julia/v0.4/HPAT/examples/pi.jl 
```

If there were any issues, make sure [MPI.jl](https://github.com/JuliaParallel/MPI.jl) 
and [ParallelAccelerator.jl](https://github.com/IntelLabs/ParallelAccelerator.jl)
are installed correctly.

On Ubuntu, these commands resolve some MPI.jl issues:
```shell
$ sudo apt-get install cmake mpich
$ julia ~/.julia/v0.4/MPI/deps/build.jl
```


