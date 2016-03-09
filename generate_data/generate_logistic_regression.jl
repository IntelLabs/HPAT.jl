using HDF5
using DocOpt
using HPAT


function generate_files(path,N)
    D = 10
    A = rand(Float32,D+1,N)
    hdf5_file = path*"logistic_regression.hdf5"
    csv_file = path*"logistic_regression.csv"
    if isfile(hdf5_file)
        rm(hdf5_file)
    end
    if isfile(csv_file)
        rm(csv_file)
    end
    h5write(hdf5_file,"/points", A[2:end,:])
    h5write(hdf5_file,"/responses", reshape(A[1,:],N))
    writecsv(csv_file,A')
end


function main()
    doc = """generate_logistic_regression.jl

generate random input for logistic regression example.

Usage:
  generate_logistic_regression.jl -h | --help
  generate_logistic_regression.jl [--instances=<instances>] [--path=<path>]

Options:
  -h --help                  Show this screen.
  --instances=<instances>    Specify number of instances; defaults to 2000000.
  --path=<path>              Specify output path for generated files; defaults to HPAT's default data path.
"""

    arguments = docopt(doc)

    if (arguments["--instances"] != nothing)
        instances = parse(Int, arguments["--instances"])
    else
        instances = 2*10^6
    end

    if (arguments["--path"] != nothing)
        path = arguments["--path"]
    else
        path = HPAT.getDefaultDataPath()
    end

    generate_files(path, instances)
end

main()
