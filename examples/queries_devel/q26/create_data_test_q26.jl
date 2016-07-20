using HDF5

# Arguments:
# ARGS[1] = option
# ARGS[2] = table store_sales path e.g."/home/whassan/tmp/csv/store_sales_sanitized.csv"
# ARGS[3] = table item path e.g. "/home/whassan/tmp/csv/item_sanitized.csv"
option = parse(Int,ARGS[1])
if  option == 1 && length(ARGS) >= 3
    table1_path=ARGS[2]
    table2_path=ARGS[3]
end

customer = []
sale = []
item = []
class = []
category = []

num_rows = 20000
if option == 0
    #=
    customer    iterm
    1       1
    1       2
    2       1
    1       3
    2       3
    3       1
    =#
    customer = [1,1,2,1,2,3]
    sale = [1,2,1,3,3,1]

    #= item class category
    1   3   1
    2   1   2
    3   2   1
    =#
    item = [1,2,3]
    class = [3,1,2]
    category = [1,2,1]
end

if option == 1
    store_sales_file = open(table1_path)
    item_file = open(table2_path)
    counter = 0
    for line in readlines(item_file)
        line = chomp(line)
        line_splited = split(line,",")
        push!(item,parse(Int64,line_splited[1]))
        push!(class,parse(Int64,line_splited[2]))
        push!(category,parse(Int64,line_splited[3]))
        counter = counter + 1
        if counter >= num_rows
            break
        end
    end
    counter = 0
    for line in readlines(store_sales_file)
        line = chomp(line)
        line_splited = split(line,",")
        push!(sale,parse(Int64,line_splited[1]))
        push!(customer,parse(Int64,line_splited[2]))
        counter = counter + 1
        if counter >= num_rows
            break
        end

    end
    # Explicit conversion of types. Idk how to declare them
    customer = convert(Array{Int64,1},customer)
    sale = convert(Array{Int64,1},sale)
    item = convert(Array{Int64,1},item)
    class = convert(Array{Int64,1},class)
    category = convert(Array{Int64,1},category)
end

file_name = "test_q26.hdf5"

if isfile(file_name)
    rm(file_name)
end

h5write(file_name ,"/ss_customer_sk",customer)
h5write(file_name, "/ss_item_sk",sale)

h5write(file_name, "/i_item_sk", item)
h5write(file_name, "/i_class_id", class)
h5write(file_name, "/i_category", category)
