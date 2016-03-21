using HDF5
using DocOpt
using HPAT


function generate_files(path,N)
    A = rand(Float64,N)
    hdf5_file = path*"1D_large.hdf5"
    csv_file = path*"1D_large.csv"
    if isfile(hdf5_file)
        rm(hdf5_file)
    end
    if isfile(csv_file)
        rm(csv_file)
    end
    h5write(hdf5_file,"/labels", A)
    writecsv(csv_file,A)
end

function generate_files_small(path)
    N=1000
    A = rand(Float64,N)
    hdf5_file = path*"1D_small.hdf5"
    csv_file = path*"1D_small.csv"
    if isfile(hdf5_file)
        rm(hdf5_file)
    end
    if isfile(csv_file)
        rm(csv_file)
    end
    h5write(hdf5_file,"/labels", A)
    writecsv(csv_file,A)
end

function main()
    doc = """generate random 1D array data.

Usage:
  generate_1D_array.jl -h | --help
  generate_1D_array.jl [--instances=<instances>] [--path=<path>]

Options:
  -h --help                  Show this screen.
  --instances=<instances>    Specify number of instances; defaults to 2000000.
  --path=<path>              Specify output path for generated files; defaults to HPAT's default data path.
"""

    arguments = docopt(doc)

    if (arguments["--instances"] != nothing)
        instances = parse(Int, arguments["--instances"])
    else
        instances = 10^6
    end

    if (arguments["--path"] != nothing)
        path = arguments["--path"]
    else
        path = HPAT.getDefaultDataPath()
    end

    generate_files_small(path)
    generate_files(path, instances)
end

main()
