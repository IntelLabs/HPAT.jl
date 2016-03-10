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
$ sudo apt-get install -y gcc gfortran cmake openmpi-bin openmpi-common libopenmpi-dev 
$ julia ~/.julia/v0.4/MPI/deps/build.jl
```

[HDF5](https://www.hdfgroup.org/HDF5/) is the recommended library for parallel file operations. On Ubuntu:
```shell
$ sudo apt-get install -y libhdf5-dev hdf5-tools libhdf5-openmpi-dev
```

## Performance Comparison with Spark\*
### Logistic Regression

Front page of [Spark\* website](http://spark.apache.org/) demonstrates over two orders of manitude
speedup over Hadoop\* on [Logistic Regression](https://github.com/apache/spark/blob/master/examples/src/main/python/logistic_regression.py) example. Simply put, Spark\* keeps data
in memory while Hadoop\* reads and writes to disks frequently.

HPAT is two orders of magnitude faster than Spark\*!
Data is kept in processor registers as much as possible with HPAT, which is necessary for best performance.
In addition, HPAT doesn't have Spark\*'s TCP/IP and Java Virtual Machine (JVM) overheads since it generates "bare-metal" MPI/C++ code.

Here is how one can compare the performance of HPAT and Spark\* for Logistic Regression example.

Generate input data:
```shell
# generate data with 2 billion labeled instances
$ julia $HOME/.julia/v0.4/HPAT/generate_data/generate_logistic_regression.jl --instances=2000000000 --path=/tmp/
```

Run Logistic Regression example of HPAT:
```shell
# run on 64 MPI processes
$ mpirun -np 64 julia $HOME/.julia/v0.4/examples/logistic_regression.jl --iterations=200 --file=/tmp/logistic_regression.hdf5 &> lr_hpat.out
```

Run Logistic Regression example of Spark\*:
```shell
# assuming spark is configured properly (with driver and executor memory set)
$ spark-submit $SPARK_HOME/examples/src/main/python/logistic_regression.py /tmp/logistic_regression.csv 200 &> lr_spark.out
```
### Monte Carlo Pi Estimation
Monte Carlo Pi estimation is another interesting [example of Spark\*](https://github.com/apache/spark/blob/master/examples/src/main/python/pi.py).
HPAT is over 1000 times faster for this case for various reasons!
First, HPAT can divide computation statically since it generates code rather than executing the program operation-by-operation.
Spark\* uses a dynamic scheduler with high overhead. But more importantly, Spark\* generates an enormous
array for the *map* operation, then executes the *reduce* operation. However, HPAT uses ParallelAccelerator
which removes unnecessary arrays. Therefore, HPAT doesn't create any arrays and the generated code is just a loop.
Hence, the computation is performed in registers and there is no extra memory access.

Run Pi example of HPAT:
```shell
$ mpirun -np 64 julia ~/.julia/v0.4/HPAT/examples/pi.jl 10000000
```

Run Pi example of Spark\*:
```shell
$ spark-submit $SPARK_HOME/examples/src/main/python/pi.py /tmp/logistic_regression.csv 100 &> pi_spark.out
```







