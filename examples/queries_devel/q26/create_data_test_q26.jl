using HDF5

customer = []
sale = []
item = []
class = []
category = []

option = 1
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
    item_file = open("/home/whassan/tmp/csv/item_sanitized.csv")
    store_sales_file = open("/home/whassan/tmp/csv/store_sales_sanitized.csv")
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

h5write(file_name ,"/ss_customer_sk",customer)
h5write(file_name, "/ss_item_sk",sale)

h5write(file_name, "/i_item_sk", item)
h5write(file_name, "/i_class_id", class)
h5write(file_name, "/i_category", category)
