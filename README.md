# HPAT

[![Build Status](https://travis-ci.org/ehsantn/HPAT.jl.svg?branch=master)](https://travis-ci.org/ehsantn/HPAT.jl)

*High Performance Analytics Toolkit (HPAT)* is a Julia-based framework for big data analytics on clusters that
is both easy to use and extremely fast; it is orders of magnitude faster than alternatives 
like [Apache Spark&trade;](http://spark.apache.org/).  

HPAT automatically parallelizes analytics tasks written in Julia, generates efficient MPI/C++ code, 
and uses existing high performance libraries such as [HDF5](https://www.hdfgroup.org/HDF5/)
and [Intel&reg; Data Analytics Acceleration Library (Intel&reg; DAAL)](https://software.intel.com/en-us/daal). 
HPAT is based on [ParallelAccelerator](https://github.com/IntelLabs/ParallelAccelerator.jl) 
and [CompilerTools](https://github.com/IntelLabs/CompilerTools.jl) packages. 

HPAT is in early development and therfore feedback is highly appreciated.
