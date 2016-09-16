using HDF5
using DataFrames
# Arguments:
# ARGS[1] = option
# ARGS[2] = table store_sales path e.g."/home/whassan/tmp/csv/q25/store_sales_sanitized.csv"
# ARGS[5] = table web_sales path e.g. "/home/whassan/tmp/csv/q25/web_sales_sanitized.csv"


function example_dataset(ss_customer_sk, ss_ticket_number, ss_sold_date_sk, ss_net_paid,
       ws_bill_customer_sk, ws_order_number, ws_sold_date_sk, ws_net_paid)
    # ss_customer_sk   ss_ticket_number  ss_sold_date_sk  ss_net_paid
    # 1                0                 37580            101.0
    # 1                0                 37600            3.2
    # 1                3                 34000            24.0
    # 2                5                 35000            3.5
    # 2                6                 36000            50.0
    append!(ss_customer_sk, [1,1,1,2,2])
    append!(ss_ticket_number, [0,0,3,5,6])
    append!(ss_sold_date_sk, [37580,37600,34000,35000,36000])
    append!(ss_net_paid, [101.0,3.2,24.0,3.5,50.0])

    # ws_bill_customer_sk ws_order_number ws_sold_date_sk ss_net_paid
    # 1                1                 37581            100.0
    # 1                1                 32610            30.2
    # 1                3                 34000            240.0
    # 2                5                 35000            35.0
    # 2                5                 36000            50.3
    # 3                4                 37620            30.1

    append!(ws_bill_customer_sk, [1,1,1,2,2,3])
    append!(ws_order_number, [0,0,3,5,5,4])
    append!(ws_sold_date_sk, [37581,37610,34000,35000,36000,37620])
    append!(ws_net_paid, [100.0,30.2,240.0,35.0,50.3,30.1])

    # cid   recency   frequency  totalspend
    # 1     1.0       4          468.2
    # 2     0.0       3          138.8
    # 3     1.0       1          30.1
end

function generate_dataset(ss_customer_sk, ss_ticket_number, ss_sold_date_sk, ss_net_paid, 
        ws_bill_customer_sk, ws_order_number, ws_sold_date_sk, ws_net_paid, 
        table_ss_path, table_ws_path)

    ss_df = readtable(open(table_ss_path))
    ws_df = readtable(open(table_ws_path))
    # order of columns in generate-dataset.sh
    #replace NA values in Dataframes with 2147483648
    append!(ss_customer_sk, convert(Array, ss_df[2], typemax(Int32)))
    append!(ss_ticket_number, convert(Array, ss_df[3], typemax(Int32)))
    append!(ss_sold_date_sk, convert(Array, ss_df[1], typemax(Int32)))
    append!(ss_net_paid, convert(Array, ss_df[4], typemax(Float32)))

    append!(ws_bill_customer_sk, ws_df[2])
    append!(ws_order_number, ws_df[3])
    append!(ws_sold_date_sk, ws_df[1])
    append!(ws_net_paid, ws_df[4])
end

function main()

    option = parse(Int,ARGS[1])
    if  option == 1 && length(ARGS) >= 3
        table_ss_path=ARGS[2]
        table_ws_path=ARGS[3]
    end

    ss_customer_sk = Int64[]
    ss_ticket_number = Int64[]
    ss_sold_date_sk = Int64[]
    ss_net_paid = Float64[]

    ws_bill_customer_sk = Int64[]
    ws_order_number = Int64[]
    ws_sold_date_sk = Int64[]
    ws_net_paid = Float64[]

    if option == 0
        example_dataset(ss_customer_sk, ss_ticket_number, ss_sold_date_sk, ss_net_paid, ws_bill_customer_sk, ws_order_number, ws_sold_date_sk, ws_net_paid)
    end

    if option == 1
        generate_dataset(ss_customer_sk, ss_ticket_number, ss_sold_date_sk, ss_net_paid, ws_bill_customer_sk, ws_order_number, ws_sold_date_sk, ws_net_paid, table_ss_path, table_ws_path)
    end
    println(":D Done reading into arrays whole dataset")
    file_name = "test_q25.hdf5"

    if isfile(file_name)
        rm(file_name)
    end
    h5write(file_name ,"/ss_customer_sk", ss_customer_sk)
    h5write(file_name ,"/ss_ticket_number", ss_ticket_number)
    h5write(file_name ,"/ss_sold_date_sk", ss_sold_date_sk)
    h5write(file_name ,"/ss_net_paid", ss_net_paid)
    h5write(file_name ,"/ws_bill_customer_sk", ws_bill_customer_sk)
    h5write(file_name ,"/ws_order_number", ws_order_number)
    h5write(file_name ,"/ws_sold_date_sk", ws_sold_date_sk)
    h5write(file_name ,"/ws_net_paid", ws_net_paid)
end
main()
