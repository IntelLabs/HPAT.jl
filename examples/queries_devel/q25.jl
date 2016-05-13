using HPAT

# DataTable
#     list column names and types can specify new column names as a 4th parameter
#     store_sales = DataSource(DataTable{:ss_customer_sk=Int64, :ss_ticket_number=Int64, :ss_sold_date_sk=Int64, :ss_net_paid=Float64}, HDF5, file_name)
#     table_name = DataSource(DataTable{:column1=<typeof_column1>, :column2=<typeof_column2>, ...}, HDF5, filename, [:column1_name_in_file, :column2_name_in_file,...])
#
# join
#     join <table1> <table2> <column_from_table1>==<column_from_table2> [joined_column_name]
#     The joined column name is optional but if present reduces the two columns from the original tables down to one column.
#
# table[conditions] - returns a new table filtered by the conditions
# table[:new_column_name1 = :old_column_name1 && :new_column_name2 = :old_column_name2 ...]
#     Use this form to return a new table with renamed columns.  Columns not listed retain their names.
#     This can be combined with conditions.  Effectively, renaming is always "true."
# table[:column_name1 && :column_name2]
#     Returns a new table with only the specified columns from the given table.
#
# DataTable + DataTable
#     Add the contents of two DataTables together.  This requires all column names to be identical.
#
# aggregate
#     aggregate(<table>, :new_column_name1 = :group_by_key_name,
#                        :new_column_name2 = aggregation_function(<column_expression>), ...)
#     where aggregation_function is count, max, or sum.

@acc hpat function q25(d_date, file_name)
    store_sales = DataSource(DataTable{:ss_customer_sk=Int64, :ss_ticket_number=Int64, :ss_sold_date_sk=Int64, :ss_net_paid=Float64}, HDF5, file_name)
    web_sales = DataSource(DataTable{:ws_bill_customer_sk=Int64, :ws_order_number=Int64, :ws_sold_date_sk=Int64, :ws_net_paid=Float64}, HDF5, file_name)

    store_sales = store_sales[:ss_sold_date_sk > d_date]
    web_sales   = web_sales[:ws_sold_date_sk > d_date]

    web_store_agg = aggregate(store_sales, :cid = :ss_customer_sk, 
                                           :frequency = count(union(:ss_ticket_number)),
                                           :most_recent_date = max(:ss_sold_date_sk),
                                           :amount = sum(:ss_net_paid))
    web_store_agg += aggregate(web_sales,  :cid = :ws_bill_customer_sk,
                                           :frequency = count(union(:ws_order_number)),
                                           :most_recent_date = max(:ws_sold_date_sk),
                                           :amount = sum(:ws_net_paid))

    result = aggregate(web_store_agg, :cid,
                                      :recency = (37621 - max(:most_recent_date) < 60 ? 1.0 : 0.0),
                                      :frequency = sum(:frequency),
                                      :totalspend = sum(:amount))

    sort!(result, by=:cid)
end

println(q25("1-1-2015", "data.hdf5"))
