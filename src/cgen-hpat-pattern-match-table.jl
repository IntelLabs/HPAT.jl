#=
Copyright (c) 2016, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
=#

function pattern_match_call_filter(linfo,f::GlobalRef, id, cond_e, num_cols,table_cols...)
    s = ""
    if f.name!=:__hpat_filter
        return s
    end
    # its an array of array. array[2:end] and table_cols... notation does that
    all_table_cols = table_cols[1]
    out_table_cols = all_table_cols[1:num_cols]
    in_table_cols = all_table_cols[(num_cols + 1):end]

    # For unique counter variables of filter
    unique_id = string(id)
    # assuming that all columns are of same size in a table
    column1_name = ParallelAccelerator.CGen.from_expr(in_table_cols[1],linfo)
    array_length = column1_name*"_array_length_filter" * unique_id
    s *= "int $array_length = " * column1_name * ".ARRAYLEN();\n"
    # Calculate final filtered array length
    write_index = "write_index_filter" * unique_id
    s *= "int $write_index = 1;\n"
    cond_e_arr = ParallelAccelerator.CGen.from_expr(cond_e, linfo)
    s *= "for (int index = 1 ; index < $array_length + 1 ; index++) { \n"
    s *= "if ( $cond_e_arr.ARRAYELEM(index) ){\n"
    # If condition satisfy copy all columns values
    for col_name in in_table_cols
        arr_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "$arr_col_name.ARRAYELEM($write_index) =  $arr_col_name.ARRAYELEM(index); \n"
    end
    s *= "$write_index = $write_index + 1;\n"
    s *= "};\n" # if condition
    s *= "};\n" # for loop
    # After filtering we need to change the size of each array
    # And assign to output filter column tables
    for (index, col_name) in enumerate(in_table_cols)
        arr_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        out_arr_col_name = ParallelAccelerator.CGen.from_expr(out_table_cols[index],linfo)
        s *= "$arr_col_name.dims[0] =  $write_index - 1; \n"
        s *= "$out_arr_col_name = $arr_col_name ; \n"
    end
    return s
end

function pattern_match_call_filter(linfo,f::Any, id, cond_e, num_cols,table_cols...)
    return ""
end

function pattern_match_call_join_seq(linfo, f::GlobalRef, id, table_new_cols_len, table1_cols_len, table2_cols_len, table_columns...)
    s = ""
    if f.name!=:__hpat_join
        return s
    end
    # its an array of array. array[2:end] and table_cols... notation does that
    table_columns = table_columns[1]
    # extract columns of each table
    table_new_cols = table_columns[1:table_new_cols_len]
    table1_cols = table_columns[table_new_cols_len+1:table_new_cols_len+table1_cols_len]
    table2_cols = table_columns[table_new_cols_len+table1_cols_len+1:end]

    # assuming that all columns are of same size in a table
    # Also output table's length would be sum of both table length
    t1c1_length_join = "t1c1_length_join_$id"
    t2c1_length_join = "t2c1_length_join_$id"
    joined_table_length = "joined_table_length_$id"
    t1_c1_join = ParallelAccelerator.CGen.from_expr(table1_cols[1],linfo)
    t2_c1_join = ParallelAccelerator.CGen.from_expr(table2_cols[1],linfo)
    s *= "int $t1c1_length_join = $t1_c1_join.ARRAYLEN() ;\n "
    s *= "int $t2c1_length_join = $t2_c1_join.ARRAYLEN() ;\n "
    s *= "int $joined_table_length = $t2c1_length_join * $t2c1_length_join ;\n "
    # Instantiation of columns for  output table
    for col_name in table_new_cols
        arr_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "$arr_col_name = j2c_array<int64_t>::new_j2c_array_1d(NULL, $joined_table_length);\n"
    end
    # Assuming that join is always on the first column of tables
    # Nested for loop implementation of join
    c_cond_sym = "=="
    table_new_counter_join = "table_new_counter_join_$id"
    s *= "int $table_new_counter_join = 1 ; \n"
    s *= "for (int table1_index = 1 ; table1_index < $t1c1_length_join+1 ; table1_index++) { \n"
    s *= "for (int table2_index = 1 ; table2_index < $t2c1_length_join+1 ; table2_index++) { \n"
    s *= "if ( $t1_c1_join.ARRAYELEM(table1_index) $c_cond_sym  $t2_c1_join.ARRAYELEM(table2_index) ){\n"
    count = 0;
    for (index, col_name) in enumerate(table1_cols)
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index],linfo)
        table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "$table_new_col_name.ARRAYELEM($table_new_counter_join) = $table1_col_name.ARRAYELEM(table1_index); \n"
        count = count + 1
    end
    for (index, col_name) in enumerate(table2_cols)
        if index == 1
            continue
        end
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index+count-1],linfo)
        table2_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "$table_new_col_name.ARRAYELEM($table_new_counter_join) =  $table2_col_name.ARRAYELEM(table2_index); \n"
    end
    s *= "$table_new_counter_join++;\n"
    s *= "};\n" # join if condition
    s *= "};\n" # inner for loop
    s *= "};\n" # outer for loop
    # Change the size of each output array
    for col_name in table_new_cols
        arr_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "$arr_col_name.dims[0] =  $table_new_counter_join - 1; \n"
    end
    # For debugging
    #s *= "for (int i = 1 ; i < $table_new_counter_join ; i++){ std::cout << psale_itemspss_item_sk.ARRAYELEM(i) << std::endl;}\n"
    return s
end

function pattern_match_call_join_seq(linfo, f::Any, id, table_new_len, table1_len, table2_len, table_columns...)
    return ""
end

function pattern_match_call_join(linfo, f::GlobalRef, id, table_new_cols_len, table1_cols_len, table2_cols_len, table_columns...)
    s = ""
    if f.name!=:__hpat_join
        return s
    end
    # TODO remove join random. Use join id/counter in domain pass and pass to this function
    HPAT_path = joinpath(dirname(@__FILE__), "..")
    HPAT_includes = string("\n#include \"", HPAT_path, "/deps/include/hpat_sort.h\"\n")
    ParallelAccelerator.CGen.addCgenUserOptions(ParallelAccelerator.CGen.CgenUserOptions(HPAT_includes,"",""))

    # its an array of array. array[2:end] and table_cols... notation does that
    table_columns = table_columns[1]
    # extract columns of each table
    table_new_cols = table_columns[1:table_new_cols_len]
    table1_cols = table_columns[table_new_cols_len+1:table_new_cols_len+table1_cols_len]
    table2_cols = table_columns[table_new_cols_len+table1_cols_len+1:end]

    s *= "int join_num_pes_$id;\n"
    s *= "MPI_Comm join_comm_$id;\n"

    if haskey(ENV, "ENABLE_GAAS")
        s *= " join_num_pes_$id = " * ENV["ENABLE_GAAS"] * ";\n"
        s *= " join_comm_$id = __hpat_bridge_comm;\n"
    else
        s *= " join_num_pes_$id = __hpat_num_pes ;\n"
        s *= " join_comm_$id = MPI_COMM_WORLD ;\n"
    end

    if haskey(ENV, "ENABLE_GAAS")
        s *= gatherv_part_of_gass(table1_cols, table2_cols, string(id), linfo)
    end

    # Sending counts for both tables
    s *= "int * scount_t1_$id;\n"
    s *= "int * scount_t2_$id;\n"

    s *= "int * scount_t1_tmp_$id;\n"
    s *= "int * scount_t2_tmp_$id;\n"

    s *= "scount_t1_$id = (int*)malloc(sizeof(int)* join_num_pes_$id);\n"
    s *= "memset (scount_t1_$id, 0, sizeof(int)* join_num_pes_$id);\n"

    s *= "scount_t1_tmp_$id = (int*)malloc(sizeof(int)* join_num_pes_$id);\n"
    s *= "memset (scount_t1_tmp_$id, 0, sizeof(int)* join_num_pes_$id);\n"

    s *= "scount_t2_$id = (int*)malloc(sizeof(int)* join_num_pes_$id);\n"
    s *= "memset (scount_t2_$id, 0, sizeof(int)* join_num_pes_$id);\n"

    s *= "scount_t2_tmp_$id = (int*)malloc(sizeof(int)* join_num_pes_$id);\n"
    s *= "memset (scount_t2_tmp_$id, 0, sizeof(int)* join_num_pes_$id);\n"

    # Receiving counts for both tables
    s *= "int rsize_t1_$id = 0;\n"
    s *= "int rsize_t2_$id = 0;\n"

    s *= "int * rcount_t1_$id;\n"
    s *= "int * rcount_t2_$id;\n"
    s *= "rcount_t1_$id = (int*)malloc(sizeof(int)* join_num_pes_$id);\n"
    s *= "rcount_t2_$id = (int*)malloc(sizeof(int)* join_num_pes_$id);\n"

    # Displacement arrays for both tables
    s *= "int * sdis_t1_$id;\n"
    s *= "int * rdis_t1_$id;\n"
    s *= "sdis_t1_$id = (int*)malloc(sizeof(int)* join_num_pes_$id);\n"
    s *= "rdis_t1_$id = (int*)malloc(sizeof(int)* join_num_pes_$id);\n"

    s *= "int *sdis_t2_$id;\n"
    s *= "int *rdis_t2_$id;\n"
    s *= "sdis_t2_$id = (int*)malloc(sizeof(int)* join_num_pes_$id);\n"
    s *= "rdis_t2_$id = (int*)malloc(sizeof(int)* join_num_pes_$id);\n"

    t1c1_length_join = "t1c1_length_join_$id"
    t2c1_length_join = "t2c1_length_join_$id"

    t1_c1_join = ParallelAccelerator.CGen.from_expr(table1_cols[1],linfo)
    t2_c1_join = ParallelAccelerator.CGen.from_expr(table2_cols[1],linfo)
    s *= "int $t1c1_length_join = $t1_c1_join.ARRAYLEN() ;\n "
    s *= "int $t2c1_length_join = $t2_c1_join.ARRAYLEN() ;\n "

    # Starting for table 1
    s *= "for (int i = 1 ; i <  $t1c1_length_join + 1 ; i++){\n"
    s *= "int node_id = $t1_c1_join.ARRAYELEM(i) % join_num_pes_$id ;\n"
    s *= "scount_t1_$id[node_id]++;"
    s *= "}\n"

    s *= "sdis_t1_$id[0]=0;\n"
    s *= "for(int i=1;i < join_num_pes_$id ;i++){\n"
    s *= "sdis_t1_$id[i]=scount_t1_$id[i-1] + sdis_t1_$id[i-1];\n"
    s *= "}\n"

    s *= "MPI_Alltoall(scount_t1_$id,1,MPI_INT,rcount_t1_$id,1,MPI_INT, join_comm_$id);\n"

    # Declaring temporary buffers
    # Assuming that all of them have same length
    # TODO do insertion sort like a combiner in Hadoop
    for (index, col_name) in enumerate(table1_cols)
        table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        j2c_typ = get_j2c_type_from_array(col_name,linfo)
        table1_col_name_tmp = table1_col_name * "_tmp_join_" * string(id)
        s *= "j2c_array< $j2c_typ > $table1_col_name_tmp = j2c_array<$j2c_typ>::new_j2c_array_1d(NULL, $t1c1_length_join );\n"
    end
    s *= "for (int i = 1 ; i <  $t1c1_length_join + 1 ; i++){\n"
    s *= "int node_id = $t1_c1_join.ARRAYELEM(i) % join_num_pes_$id;\n"
    for (index, col_name) in enumerate(table1_cols)
        table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        table1_col_name_tmp = table1_col_name * "_tmp_join_" * string(id)
        s *= "$table1_col_name_tmp.ARRAYELEM(sdis_t1_$id[node_id]+scount_t1_tmp_$id[node_id]+1) = $table1_col_name.ARRAYELEM(i);\n"
    end
    s *= "scount_t1_tmp_$id[node_id]++;\n"
    s *= "}\n"

    # Starting for table 2
    s *= "for (int i = 1 ; i <  $t2c1_length_join + 1 ; i++){\n"
    s *= "int node_id = $t2_c1_join.ARRAYELEM(i) % join_num_pes_$id ;\n"
    s *= "scount_t2_$id[node_id]++;"
    s *= "}\n"

    s *= "sdis_t2_$id[0]=0;\n"
    s *= "for(int i=1;i < __hpat_num_pes;i++){\n"
    s *= "sdis_t2_$id[i]=scount_t2_$id[i-1] + sdis_t2_$id[i-1];\n"
    s *= "}\n"

    s *= "MPI_Alltoall(scount_t2_$id,1,MPI_INT, rcount_t2_$id,1,MPI_INT,MPI_COMM_WORLD);\n"

    # Declaring temporary buffers
    for (index, col_name) in enumerate(table2_cols)
        table2_col_name =ParallelAccelerator.CGen.from_expr(col_name,linfo)
        j2c_typ = get_j2c_type_from_array(col_name,linfo)
        table2_col_name_tmp =  table2_col_name * "_tmp_join_" * string(id)
        s *= "j2c_array< $j2c_typ > $table2_col_name_tmp = j2c_array<$j2c_typ>::new_j2c_array_1d(NULL, $t2c1_length_join);\n"
    end

    s *= "for (int i = 1 ; i <   $t2c1_length_join + 1 ; i++){\n"
    s *= "int node_id = $t2_c1_join.ARRAYELEM(i) % join_num_pes_$id ;\n"
    for (index, col_name) in enumerate(table2_cols)
        table2_col_name =ParallelAccelerator.CGen.from_expr(col_name,linfo)
        table2_col_name_tmp =  table2_col_name * "_tmp_join_$id"
        s *= "$table2_col_name_tmp.ARRAYELEM(sdis_t2_$id[node_id]+scount_t2_tmp_$id[node_id]+1) = $table2_col_name.ARRAYELEM(i);\n"
    end
    s *= "scount_t2_tmp_$id[node_id]++;\n"
    s *= "}\n"

    # Caculating displacements for both tables
    s *= """
              rdis_t1_$id[0]=0;
              rdis_t2_$id[0]=0;
              for(int i=1;i < join_num_pes_$id ;i++){
                  rdis_t1_$id[i] = rcount_t1_$id[i-1] + rdis_t1_$id[i-1];
                  rdis_t2_$id[i] = rcount_t2_$id[i-1] + rdis_t2_$id[i-1];
              }
        """

    # Summing up receiving counts
    s *= """
            for(int i=0;i< join_num_pes_$id ;i++){
                rsize_t1_$id = rsize_t1_$id + rcount_t1_$id[i];
                rsize_t2_$id = rsize_t2_$id + rcount_t2_$id[i];
              }
        """
    for (index, col_name) in enumerate(table1_cols)
        table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        j2c_typ = get_j2c_type_from_array(col_name,linfo)
        table1_col_name_tmp = table1_col_name *"_tmp_join_" * string(id)
        s *= " j2c_array< $j2c_typ > rbuf_$table1_col_name = j2c_array<$j2c_typ>::new_j2c_array_1d(NULL, rsize_t1_$id);\n"
        s *= """ MPI_Alltoallv($table1_col_name_tmp.getData(), scount_t1_$id, sdis_t1_$id, MPI_INT64_T,
                                     rbuf_$table1_col_name.getData(), rcount_t1_$id, rdis_t1_$id, MPI_INT64_T, join_comm_$id);
                     """
        s *= " $table1_col_name = rbuf_$table1_col_name; \n"
    end
    # delete [] tmp_table1_col_name

    for (index, col_name) in enumerate(table2_cols)
        table2_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        j2c_typ = get_j2c_type_from_array(col_name,linfo)
        table2_col_name_tmp = table2_col_name * "_tmp_join_" * string(id)
        s *= " j2c_array< $j2c_typ > rbuf_$table2_col_name = j2c_array<$j2c_typ>::new_j2c_array_1d(NULL, rsize_t2_$id);\n"
        s *= """ MPI_Alltoallv($table2_col_name_tmp.getData(), scount_t2_$id, sdis_t2_$id, MPI_INT64_T,
                                     rbuf_$table2_col_name.getData(), rcount_t2_$id, rdis_t2_$id, MPI_INT64_T, join_comm_$id);
                     """
        s *= " $table2_col_name = rbuf_$table2_col_name; \n"
    end
    # delete [] tmp_table2_col_name

    table_new_counter_join = "table_new_counter_join_$id"
    s *= "int $table_new_counter_join = 1 ; \n"
    count = 0;
    # Initiatilizing new table(output table) arrays
    for (index, col_name) in enumerate(table1_cols)
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index],linfo)
        j2c_typ = get_j2c_type_from_array(col_name,linfo)
        s *= "std::vector< $j2c_typ> *vec_$(id)_$table_new_col_name = new std::vector< $j2c_typ>();\n"
        count = count + 1
    end
    for (index, col_name) in enumerate(table2_cols)
        if index == 1
            continue
        end
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index+count-1],linfo)
        j2c_typ = get_j2c_type_from_array(col_name,linfo)
        s *= "std::vector< $j2c_typ> *vec_$(id)_$table_new_col_name = new std::vector< $j2c_typ>();\n"
    end

    if haskey(ENV, "ENABLE_GAAS")
        s *= scatterv_part_of_gass(table1_cols, table2_cols, string(id), linfo)
    end

    # Use any sorting algorithm here before merging
    # Right now using simple bubble sort
    # TODO add tim sort here too
    j2c_type_t1 = get_j2c_type_from_array(table1_cols[1],linfo)
    j2c_type_t2 = get_j2c_type_from_array(table2_cols[1],linfo)
    t1_length = length(table1_cols)
    t2_length = length(table2_cols)
    t1_all_arrays = "t1_all_arrays_$id"
    t2_all_arrays = "t2_all_arrays_$id"
    s *= "$j2c_type_t1 * $t1_all_arrays[$t1_length - 1];\n"
    s *= "$j2c_type_t2 * $t2_all_arrays[$t2_length - 1];\n"
    for (index, col_name) in enumerate(table1_cols)
        if index == 1
            continue
        end
        arr_index = index - 2
        table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "$t1_all_arrays [$arr_index] = ( $j2c_type_t1 *) $table1_col_name.getData();\n"
    end

    for (index, col_name) in enumerate(table2_cols)
        if index == 1
            continue
        end
        arr_index = index - 2
        table2_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "$t2_all_arrays [$arr_index] = ( $j2c_type_t2 *) $table2_col_name.getData();\n"
    end
    s *= "__hpat_timsort(( $j2c_type_t1 *) $t1_c1_join.getData(), rsize_t1_$id , $t1_all_arrays, $t1_length - 1);\n"
    s *= "__hpat_timsort(( $j2c_type_t2 *) $t2_c1_join.getData(), rsize_t2_$id , $t2_all_arrays, $t2_length - 1);\n"
    # s *= "__hpat_quicksort($t1_all_arrays,$t1_length - 1, ( $j2c_type_t1 *) $t1_c1_join.getData(), 0, $rsize_t1 - 1);\n"
    # s *= "__hpat_quicksort($t2_all_arrays,$t2_length - 1, ( $j2c_type_t2 *) $t2_c1_join.getData(), 0, $rsize_t2 - 1);\n"

    #s *= "qsort($t2_c1_join.getData(),$rsize_t2, sizeof( $j2c_type_t2 ), __hpat_compare_qsort_$j2c_type_t2);\n"
    # after the arrays has been sorted merge them
    # I used algorithm from here www.dcs.ed.ac.uk/home/tz/phd/thesis/node20.htm
    left = "left_join_table_$id"
    right = "right_join_table_$id"
    s *= "int $left = 1;\n"
    s *= "int $right = 1;\n"
    s *= "while ( ($left < rsize_t1_$id + 1) && ($right < rsize_t2_$id + 1) ){\n"
    s *= "if($t1_c1_join.ARRAYELEM($left) == $t2_c1_join.ARRAYELEM($right)){\n"
    count = 0
    for (index, col_name) in enumerate(table1_cols)
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index],linfo)
        table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "vec_$(id)_$table_new_col_name->push_back( $table1_col_name.ARRAYELEM($left) );\n"
        count = count + 1
    end
    for (index, col_name) in enumerate(table2_cols)
        if index == 1
            continue
        end
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index+count-1],linfo)
        table2_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "vec_$(id)_$table_new_col_name->push_back( $table2_col_name.ARRAYELEM($right) ); \n"
    end
    s *= "$table_new_counter_join++;\n"

    s *= "int tmp_$left = $left + 1 ;\n"
    s *= "while((tmp_$left < rsize_t1_$id + 1) && ($t1_c1_join.ARRAYELEM(tmp_$left) == $t2_c1_join.ARRAYELEM($right))){\n"
    count = 0
    for (index, col_name) in enumerate(table1_cols)
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index],linfo)
        table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "vec_$(id)_$table_new_col_name->push_back( $table1_col_name.ARRAYELEM(tmp_$left) ); \n"
        count = count + 1
    end
    for (index, col_name) in enumerate(table2_cols)
        if index == 1
            continue
        end
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index+count-1],linfo)
        table2_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "vec_$(id)_$table_new_col_name->push_back( $table2_col_name.ARRAYELEM($right) ); \n"
    end
    s *= "tmp_$left++;\n"
    s *= "$table_new_counter_join++;\n"
    s *= "}\n"

    s *= "int tmp_$right = $right + 1 ;\n"
    s *= "while((tmp_$right < rsize_t2_$id + 1) && ($t1_c1_join.ARRAYELEM($left) == $t2_c1_join.ARRAYELEM(tmp_$right))){\n"
    count = 0
    for (index, col_name) in enumerate(table1_cols)
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index],linfo)
        table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "vec_$(id)_$table_new_col_name->push_back( $table1_col_name.ARRAYELEM($left) ); \n"
        count = count + 1
    end
    for (index, col_name) in enumerate(table2_cols)
        if index == 1
            continue
        end
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index+count-1],linfo)
        table2_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        s *= "vec_$(id)_$table_new_col_name->push_back( $table2_col_name.ARRAYELEM(tmp_$right) ); \n"
    end
    s *= "tmp_$right++;\n"
    s *= "$table_new_counter_join++;\n"
    s *= "}\n"

    s *= "$left++;\n"
    s *= "$right++;\n"
    s *= "}\n" # if condition
    s *= "else if ($t1_c1_join.ARRAYELEM($left) < $t2_c1_join.ARRAYELEM($right))\n"
    s *= "$left++;\n"
    s *= "else\n"
    s *= "$right++;\n"
    s *= "}\n" # while condition

    count = 0
    # Initializing new table(output table) arrays and copy from values from vectors
    for (index, col_name) in enumerate(table1_cols)
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index],linfo)
        j2c_typ = get_j2c_type_from_array(col_name,linfo)
        s *= "$table_new_col_name = j2c_array<$j2c_typ>::new_j2c_array_1d(vec_$(id)_$table_new_col_name->data(), vec_$(id)_$table_new_col_name->size() );\n"
        count = count + 1
    end
    for (index, col_name) in enumerate(table2_cols)
        if index == 1
            continue
        end
        table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index+count-1],linfo)
        j2c_typ = get_j2c_type_from_array(col_name,linfo)
        s *= "$table_new_col_name = j2c_array<$j2c_typ>::new_j2c_array_1d(vec_$(id)_$table_new_col_name->data(), vec_$(id)_$table_new_col_name->size() );\n"
    end
    return s
end

function pattern_match_call_join(linfo, f::Any, id, table_new_len, table1_len, table2_len, table_columns...)
    return ""
end

function pattern_match_call_agg_seq(linfo, f::GlobalRef,  id, groupby_key, num_exprs, expr_func_output_list...)
    s = ""
    if f.name!=:__hpat_aggregate
        return s
    end
    HPAT_path = joinpath(dirname(@__FILE__), "..")
    HPAT_includes = string("\n#include <unordered_map>\n")
    ParallelAccelerator.CGen.addCgenUserOptions(ParallelAccelerator.CGen.CgenUserOptions(HPAT_includes,"",""))

    expr_func_output_list = expr_func_output_list[1]
    exprs_list = expr_func_output_list[1:num_exprs]
    funcs_list = expr_func_output_list[num_exprs+1:(2*num_exprs)]

    # first element of output list is the groupbykey column
    output_cols_list = expr_func_output_list[(2*num_exprs)+1 : end]
    agg_key_col_input = ParallelAccelerator.CGen.from_expr(groupby_key, linfo)
    agg_key_col_output = ParallelAccelerator.CGen.from_expr(output_cols_list[1], linfo)
    # Temporaty map for each column
    for (index, value) in enumerate(output_cols_list)
        table_new_col_name = ParallelAccelerator.CGen.from_expr(value,linfo)
        s *= "std::unordered_map<int,int> temp_map_$table_new_col_name ;\n"
    end
    agg_key_map_temp = "temp_map_$agg_key_col_output"
    s *= "for(int i = 1 ; i < $agg_key_col_input.ARRAYLEN() + 1 ; i++){\n"
    s *= "$agg_key_map_temp[$agg_key_col_input.ARRAYELEM(i)] = $agg_key_col_input.ARRAYELEM(i);\n"
    for (index, func) in enumerate(funcs_list)
        column_name = ""
        expr_name = ParallelAccelerator.CGen.from_expr(exprs_list[index],linfo)
        map_name = "temp_map_" * ParallelAccelerator.CGen.from_expr(output_cols_list[index+1],linfo)
        s *= return_reduction_string_with_closure(agg_key_col_input, expr_name, map_name, func)
    end
    s *= "}\n"
    # Initializing new columns
    for col_name in output_cols_list
        arr_col_name = ParallelAccelerator.CGen.from_expr(col_name, linfo)
        j2c_typ = get_j2c_type_from_array(col_name,linfo)
        s *= "$arr_col_name = j2c_array<$j2c_typ>::new_j2c_array_1d(NULL, $agg_key_map_temp.size());\n"
    end
    # copy back the values from map into arrays
    counter_agg = "counter_agg$id"
    s *= "int $counter_agg = 1;\n"
    s *= "for(auto i : $agg_key_map_temp){\n"
    for (index, value) in enumerate(output_cols_list)
        map_name = ParallelAccelerator.CGen.from_expr(value, linfo)
        s *= "$map_name.ARRAYELEM($counter_agg) = temp_map_$map_name[i.first];\n"
    end
    s *= "$counter_agg++;\n"
    s *= "}\n"
    # Debugging
    # s *= "for (int i = 1 ; i < $counter_agg ; i++){ std::cout << pcustomer_i_classpid3.ARRAYELEM(i) << std::endl;}\n"
    return s
end

function pattern_match_call_agg_seq(linfo, f::Any, id,  groupby_key, num_exprs, exprs_func_list...)
    return ""
end

function pattern_match_call_agg(linfo, f::GlobalRef,  id, groupby_key, num_exprs, expr_func_output_list...)
    s = ""
    if f.name!=:__hpat_aggregate
        return s
    end

    HPAT_path = joinpath(dirname(@__FILE__), "..")
    HPAT_includes = string("\n#include <unordered_map>\n")
    ParallelAccelerator.CGen.addCgenUserOptions(ParallelAccelerator.CGen.CgenUserOptions(HPAT_includes,"",""))

    expr_func_output_list = expr_func_output_list[1]
    exprs_list = expr_func_output_list[1:num_exprs]
    expr_arrs = map(x->ParallelAccelerator.CGen.from_expr(x, linfo), exprs_list)
    funcs_list = expr_func_output_list[num_exprs+1:(2*num_exprs)]

    # first element of output list is the groupbykey column
    output_cols_list = expr_func_output_list[(2*num_exprs)+1 : end]
    agg_key_col_input = ParallelAccelerator.CGen.from_expr(groupby_key, linfo)
    agg_key_col_output = ParallelAccelerator.CGen.from_expr(output_cols_list[1], linfo)

    agg_key_map_temp = "temp_map_$(id)_$agg_key_col_output"
    s *= "std::unordered_map<int,int> $agg_key_map_temp;\n"
    # generate code to get number of elements to send and receive in alltoallv
    # scount_id, rcount_id sdis_id, rdis_id, rsize_id are generated
    s *= gen_alltoall_counts(id, agg_key_col_input, agg_key_map_temp)

    # allocate buffer arrays, first column is groupbykey which is handled separately
    agg_key_col_input_tmp = agg_key_col_input * "_tmp_agg_$id"
    key_ctype = get_j2c_type_from_array(groupby_key, linfo)
    s *= "j2c_array< $key_ctype > $agg_key_col_input_tmp = j2c_array< $key_ctype >::new_j2c_array_1d(NULL, agg_total_unique_keys_$id);\n"
    for (index, expr_arr) in enumerate(expr_arrs)
        expr_arr_tmp = expr_arr * "_tmp_agg_$id"
        j2c_type = get_j2c_type_from_array(output_cols_list[index+1],linfo)
        # allocate aggregate array to accumulate locally
        s *= alloc_agg_arr(funcs_list[index], expr_arr_tmp, key_ctype, j2c_type, "agg_total_unique_keys_$id")
    end

    # aggregate locally and write to send buffer (proper index for each node)
    s *= "for(int i = 1; i <= $agg_key_col_input.ARRAYLEN(); i++){\n"
    s *= "  $key_ctype key = $agg_key_col_input.ARRAYELEM(i);\n"
    s *= "  int node_id = key % __hpat_num_pes ;\n"
    s *= "  if ($agg_key_map_temp.find(key) == $agg_key_map_temp.end()){\n"
    agg_write_index = "agg_write_index_$id"
    s *= "    int $agg_write_index = sdis_$id[node_id]+s_ind_tmp_$id[node_id]+1 ;\n"
    s *= "    $agg_key_map_temp[key] = $agg_write_index;\n"
    s *= "    $agg_key_col_input_tmp.ARRAYELEM($agg_write_index) = key;\n"
    for (index, func) in enumerate(funcs_list)
        expr_arr = expr_arrs[index]
        expr_arr_tmp = expr_arr * "_tmp_agg_$id"
        j2c_type = get_j2c_type_from_array(output_cols_list[index+1],linfo)
        s *= return_combiner_string_with_closure_first_elem(expr_arr_tmp, expr_arr, func, agg_write_index, "key", key_ctype, j2c_type)
    end
    s *= "    s_ind_tmp_$id[node_id]++;\n"
    s *= "}\n"
    s *= "else{\n"
    current_write_index = "current_write_index$id"
    s *= "int $current_write_index = $agg_key_map_temp[key];\n"
    for (index, func) in enumerate(funcs_list)
        expr_arr = expr_arrs[index]
        expr_arr_tmp = expr_arr * "_tmp_agg_$id"
        j2c_type = get_j2c_type_from_array(output_cols_list[index+1],linfo)
        s *= return_combiner_string_with_closure_second_elem(expr_arr_tmp, expr_arr, func, current_write_index, "key", key_ctype, j2c_type)
    end
    s *= "}\n"
    s *= "}\n"
    s *= "$agg_key_map_temp.clear();\n"

    # generate communication for arrays
    # First column is groupbykey which is handled separately
    # After mpi_alltoallv the length of agg_key_col_input is changed. Don't use agg_key_col_input_len
    mpi_type = get_mpi_type_from_array(groupby_key,linfo)
    s *= " j2c_array< $key_ctype > rbuf_$(id)_$agg_key_col_input = j2c_array< $key_ctype >::new_j2c_array_1d(NULL, rsize_$id);\n"
    s *= """ MPI_Alltoallv($agg_key_col_input_tmp.getData(), scount_$id, sdis_$id, $mpi_type,
                                         rbuf_$(id)_$agg_key_col_input.getData(), rcount_$id, rdis_$id, $mpi_type, MPI_COMM_WORLD);
                         """

    for (index, expr_arr) in enumerate(expr_arrs)
        mpi_type = get_mpi_type_from_array(output_cols_list[index+1], linfo)
        j2c_type = get_j2c_type_from_array(output_cols_list[index+1], linfo)
        s *= gen_expr_arr_comm(id, funcs_list[index], expr_arr, j2c_type, mpi_type, key_ctype)
    end
    # delete [] expr_name_tmp

    for col_name in output_cols_list
        j2c_type = get_j2c_type_from_array(col_name,linfo)
        arr_col_name = ParallelAccelerator.CGen.from_expr(col_name, linfo)
        s *= "$arr_col_name = j2c_array< $j2c_type >::new_j2c_array_1d(NULL, rbuf_$(id)_$agg_key_col_input.ARRAYLEN());\n"
    end

    agg_write_index = "agg_write_index_$id"
    s *= "int $agg_write_index = 1;\n"
    s *= "for(int i = 1 ; i < rbuf_$(id)_$agg_key_col_input.ARRAYLEN() + 1 ; i++){\n"
    s *= "  $key_ctype key = rbuf_$(id)_$agg_key_col_input.ARRAYELEM(i);\n"
    s *= "  if ($agg_key_map_temp.find(key) == $agg_key_map_temp.end()){"
    s *= "    $agg_key_map_temp[key] = $agg_write_index ;\n"
    col_name = ParallelAccelerator.CGen.from_expr(output_cols_list[1], linfo)
    s *= "    $col_name.ARRAYELEM($agg_write_index) = key;\n"
    for (index, func) in enumerate(funcs_list)
        expr_arr = expr_arrs[index]
        rbuf_expr_arr = "rbuf_$(id)_" * expr_arr
        new_col_name = ParallelAccelerator.CGen.from_expr(output_cols_list[index + 1], linfo)
        s *= return_reduction_string_with_closure_first_elem(new_col_name, rbuf_expr_arr, func, agg_write_index, "key", expr_arr * "_tmp_agg_$id")
    end
    s *= "    $agg_write_index++;\n"
    s *= "}\n"
    s *= "else{\n"
    current_write_index = "current_write_index$id"
    s *= "int $current_write_index = $agg_key_map_temp[key];\n"
    for (index, func) in enumerate(funcs_list)
        expr_arr = expr_arrs[index]
        rbuf_expr_arr = "rbuf_$(id)_" * expr_arr
        new_col_name = ParallelAccelerator.CGen.from_expr(output_cols_list[index + 1], linfo)
        s *= return_reduction_string_with_closure_second_elem(new_col_name, rbuf_expr_arr, func, current_write_index, "key", expr_arr * "_tmp_agg_$id")
    end
    s *= "}\n"
    s *= "}\n"
    counter_agg = "counter_agg_$id"
    s *= "int $counter_agg = $agg_key_map_temp.size();\n"
    for col_name in output_cols_list
        j2c_type = get_j2c_type_from_array(col_name,linfo)
        arr_col_name = ParallelAccelerator.CGen.from_expr(col_name, linfo)
        s *= "$arr_col_name.dims[0] = $counter_agg;\n"
    end
    return s
end

function pattern_match_call_agg(linfo, f::Any, groupby_key, num_exprs, exprs_func_list...)
    return ""
end

function alloc_agg_arr(func, expr_arr_tmp, key_ctype, arr_ctyp, num_total_keys)
    s = ""
    if func==GlobalRef(HPAT.DomainPass,:length_unique)
        HPAT_includes = """
                            #include <boost/functional/hash.hpp>
                            #include <utility>
                            #include <unordered_set>
                            """
        ParallelAccelerator.CGen.addCgenUserOptions(ParallelAccelerator.CGen.CgenUserOptions(HPAT_includes,"",""))
        # a map for each key
        s = """std::unordered_set<std::pair<$key_ctype,$arr_ctyp>, boost::hash<std::pair<$key_ctype,$arr_ctyp> > > *unique_set_$expr_arr_tmp =
                 new std::unordered_set<std::pair<$key_ctype,$arr_ctyp>, boost::hash<std::pair<$key_ctype,$arr_ctyp> > >[__hpat_num_pes];
               std::unordered_set<std::pair<$key_ctype,$arr_ctyp>, boost::hash<std::pair<$key_ctype,$arr_ctyp> > > recv_unique_set_$expr_arr_tmp;
               std::unordered_map<$key_ctype,$arr_ctyp> unique_map_$expr_arr_tmp;
        """
    else
        s = "j2c_array< $arr_ctyp > $expr_arr_tmp = j2c_array< $arr_ctyp >::new_j2c_array_1d(NULL, $num_total_keys);\n"
    end
    return s
end

# TODO Combine all below five functions into one.
function return_reduction_string_with_closure(agg_key_col_input,expr_arr,agg_map, func::GlobalRef)
    s = ""
    if func.name==:length
        s *= "if ($agg_map.find($agg_key_col_input.ARRAYELEM(i)) == $agg_map.end())\n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] = 1;\n"
        s *= "else \n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] += 1;\n\n"
    elseif func.name==:sum
        s *= "if ($agg_map.find($agg_key_col_input.ARRAYELEM(i)) == $agg_map.end())\n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] = $expr_arr.ARRAYELEM(i) ;\n"
        s *= "else \n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] +=  $expr_arr.ARRAYELEM(i)  ;\n\n"
    elseif func.name==:maximum
        s *= "if ($agg_map.find($agg_key_col_input.ARRAYELEM(i)) == $agg_map.end())){\n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] = $expr_arr.ARRAYELEM(i) ;}\n"
        s *= "else{ \n"
        s *= "if (agg_map_count[$agg_key_col_input.ARRAYELEM(i)] < $expr_arr.ARRAYELEM(i) ) \n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] = $expr_arr.ARRAYELEM(i) ;}\n\n"
    else
        throw("aggregate function not supported in CGen $func")
    end
    return s
end

function return_combiner_string_with_closure_first_elem(new_column_name, expr_arr, func::GlobalRef, write_index, key,key_ctype, j2c_type)
    s = ""
    if func.name==:length
        s *= "$new_column_name.ARRAYELEM($write_index) = 1;\n"
    elseif func.name==:sum
        s *= "$new_column_name.ARRAYELEM($write_index) = $expr_arr.ARRAYELEM(i);\n"
    elseif func.name==:maximum
        s *= "$new_column_name.ARRAYELEM($write_index) = $expr_arr.ARRAYELEM(i);\n"
    elseif func.name==:length_unique
        # s *= "$new_column_name.ARRAYELEM($write_index) = 1;\n"
        s *= "unique_set_$new_column_name[node_id].insert(std::make_pair($key,$expr_arr.ARRAYELEM(i)));\n"
    else
        throw("aggregate function not found $func")
    end
    return s
end

function return_combiner_string_with_closure_second_elem(new_column_name, expr_arr, func::GlobalRef, current_index, key,key_ctype, j2c_type)
    s = ""
    if func.name==:length
        s *= "$new_column_name.ARRAYELEM($current_index) += 1;\n"
    elseif func.name==:sum
        s *= "$new_column_name.ARRAYELEM($current_index) += $expr_arr.ARRAYELEM(i);\n"
    elseif func.name==:maximum
        s *= "if ($new_column_name.ARRAYELEM($current_index) < $expr_arr.ARRAYELEM(i))\n"
        s *= "$new_column_name.ARRAYELEM($current_index) = $expr_arr.ARRAYELEM(i);\n"
    elseif func.name==:length_unique
        #s *= "if(unique_map_$expr_arr[$key].find($expr_arr.ARRAYELEM(i)) == unique_map_$expr_arr[$key].end()){\n"
        #s *= "unique_map_$expr_arr[$key][$expr_arr.ARRAYELEM(i)] = true;\n"
        #s *= "$new_column_name.ARRAYELEM($current_index) += 1;\n"
        #s *= "}\n"
        s *= "unique_set_$new_column_name[node_id].insert(std::make_pair($key,$expr_arr.ARRAYELEM(i)));\n"
    else
        throw("aggregate function not found $func")
    end
    return s
end

# reduction is after alltoallv so data is already partially aggregated
function return_reduction_string_with_closure_first_elem(new_column_name,expr_arr,func::GlobalRef,write_index, key, old_expr_arr)
    s = ""
    if func.name==:length
        s *= "$new_column_name.ARRAYELEM($write_index) = $expr_arr.ARRAYELEM(i);\n"
    elseif func.name==:sum
        s *= "$new_column_name.ARRAYELEM($write_index) = $expr_arr.ARRAYELEM(i);\n"
    elseif func.name==:maximum
        s *= "$new_column_name.ARRAYELEM($write_index) = $expr_arr.ARRAYELEM(i);\n"
    elseif func.name==:length_unique
        #s *= "$new_column_name.ARRAYELEM($write_index) = $expr_arr.ARRAYELEM(i);\n"
        #s *= "unique_map_$old_expr_arr[$key].clear();\n"
        #s *= "unique_map_$old_expr_arr[$key][$expr_arr.ARRAYELEM(i)] = true;\n"
        s *= "$new_column_name.ARRAYELEM($write_index) = unique_map_$old_expr_arr[$key];\n"
    else
        throw("aggregate reduction function not found $func")
    end
    return s
end

function return_reduction_string_with_closure_second_elem(new_column_name,expr_arr,func::GlobalRef, current_index, key, old_expr_arr)
    s = ""
    if func.name==:length
        s *= "$new_column_name.ARRAYELEM($current_index) += $expr_arr.ARRAYELEM(i);\n"
    elseif func.name==:sum
        s *= "$new_column_name.ARRAYELEM($current_index) +=  $expr_arr.ARRAYELEM(i);\n"
    elseif func.name==:maximum
        s *= "if ($new_column_name.ARRAYELEM($current_index) < $expr_arr.ARRAYELEM(i))\n"
        s *= "$new_column_name.ARRAYELEM($current_index) = $expr_arr.ARRAYELEM(i);\n"
    elseif func.name==:length_unique
        #s *= "if(unique_map_$old_expr_arr[$key].find($expr_arr.ARRAYELEM(i)) != unique_map_$old_expr_arr[$key].end()){\n"
        #s *= "unique_map_$old_expr_arr[$key][$expr_arr.ARRAYELEM(i)] = true;\n"
        #s *= "$new_column_name.ARRAYELEM($current_index) += $expr_arr.ARRAYELEM(i);\n"
        #s *= "}\n"
    else
        throw("aggregate reduction function not found $func")
    end
    return s
end

function gen_alltoall_counts(id, agg_key_col_input, agg_key_map_temp)
    s = ""
    # number of elements to send to each node in alltoallv
    s *= "int *scount_$id = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
    s *= "memset(scount_$id, 0, sizeof(int)*__hpat_num_pes);\n"

    # running write index for send buffer of each node
    s *= "int *s_ind_tmp_$id = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
    s *= "memset(s_ind_tmp_$id, 0, sizeof(int)*__hpat_num_pes);\n"

    # Receiving counts
    s *= "int rsize_$id = 0;\n"
    s *= "int *rcount_$id = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"

    # send and receive indices for data sent to and received from each node
    s *= "int *sdis_$id = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
    s *= "int *rdis_$id = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"

    s *= "int agg_total_unique_keys_$id = 0;\n"
    ### Counting displacements for table
    # count unique keys of each node and total unique keys
    s *= "for (int i=1; i <= $agg_key_col_input.ARRAYLEN(); i++){\n"
    s *= "  if ($agg_key_map_temp.find($agg_key_col_input.ARRAYELEM(i)) == $agg_key_map_temp.end()){\n"
    s *= "     $agg_key_map_temp[$agg_key_col_input.ARRAYELEM(i)] = 1;\n"
    s *= "     int node_id = $agg_key_col_input.ARRAYELEM(i) % __hpat_num_pes;\n"
    s *= "     scount_$id[node_id]++;\n"
    s *= "     agg_total_unique_keys_$id++;\n"
    s *= "  }\n"
    s *= "}\n"

    s *= "sdis_$id[0]=0;\n"
    s *= "for(int i=1;i < __hpat_num_pes;i++){\n"
    s *= "sdis_$id[i] = scount_$id[i-1] + sdis_$id[i-1];\n"
    s *= "}\n"

    s *= "MPI_Alltoall(scount_$id,1,MPI_INT,rcount_$id,1,MPI_INT,MPI_COMM_WORLD);\n"

    s *= "$agg_key_map_temp.clear();\n"
    # Caculating displacements
    s *= """
           rdis_$id[0]=0;
           for(int i=1;i < __hpat_num_pes;i++){
               rdis_$id[i] = rcount_$id[i-1] + rdis_$id[i-1];
           }
        """

    # Summing receiving counts
    s *= """
            for(int i=0;i<__hpat_num_pes;i++){
                rsize_$id = rsize_$id + rcount_$id[i];
            }
        """
    return s
end

function gen_expr_arr_comm(id, func, expr_arr, j2c_type, mpi_type, key_ctype)
    s = ""
    if func==GlobalRef(HPAT.DomainPass,:length_unique)
        # sending elements as pairs of (key,value)
        arr_id = "$(id)_$expr_arr"
        sets = "unique_set_$(expr_arr)_tmp_agg_$id"
        # send and recv sizes, in bytes
        s *= "int el_size_$arr_id = sizeof($key_ctype)+sizeof($j2c_type);\n"
        s *= "int total_send_count_$arr_id = 0;\n"
        s *= "int total_recv_count_$arr_id = 0;\n"
        s *= "int *send_sizes_$arr_id = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
        s *= "memset(send_sizes_$arr_id, 0, sizeof(int)*__hpat_num_pes);\n"
        s *= "int *recv_sizes_$arr_id = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
        s *= "memset(recv_sizes_$arr_id, 0, sizeof(int)*__hpat_num_pes);\n"
        s *= "int *send_dis_$arr_id = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
        s *= "int *recv_dis_$arr_id = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
        s *= """
              for(int i=0; i<__hpat_num_pes; i++) {
                send_sizes_$arr_id[i] = $sets[i].size() * el_size_$arr_id;
                total_send_count_$arr_id += $sets[i].size();
              }
        """
        s *= "MPI_Alltoall(send_sizes_$arr_id, 1, MPI_INT, recv_sizes_$arr_id, 1, MPI_INT, MPI_COMM_WORLD);\n"
        s *= """
              for(int i=0; i<__hpat_num_pes; i++) {
                total_recv_count_$arr_id += recv_sizes_$arr_id[i];
              }
              total_recv_count_$arr_id /= el_size_$arr_id;
        """
        # send and recv buffers
        s *= "char *send_buf_$arr_id = new char[total_send_count_$arr_id * el_size_$arr_id];\n"
        s *= "char *recv_buf_$arr_id = new char[total_recv_count_$arr_id * el_size_$arr_id];\n"
        # pack data in send buffer
        s *= """
             int curr_buff_loc_$arr_id = 0;
             for(int i=0; i<__hpat_num_pes; i++) {
               send_dis_$arr_id[i] = curr_buff_loc_$arr_id;
               for(auto& x: $sets[i]) {
                 $key_ctype *ptr1 = ($key_ctype*) &send_buf_$arr_id[curr_buff_loc_$arr_id];
                 *ptr1 = x.first;
                 curr_buff_loc_$arr_id += sizeof($key_ctype);
                 $j2c_type *ptr2 = ($j2c_type*) &send_buf_$arr_id[curr_buff_loc_$arr_id];
                 *ptr2 = x.second;
                 curr_buff_loc_$arr_id += sizeof($j2c_type);
               }
              }
              recv_dis_$arr_id[0] = 0;
              for(int i=1; i<__hpat_num_pes; i++) {
                recv_dis_$arr_id[i] = recv_sizes_$arr_id[i-1] + recv_dis_$arr_id[i-1];
              }
        """
        s *= """ MPI_Alltoallv(send_buf_$arr_id, send_sizes_$arr_id, send_dis_$arr_id, MPI_CHAR,
                                             recv_buf_$arr_id, recv_sizes_$arr_id, recv_dis_$arr_id, MPI_CHAR, MPI_COMM_WORLD);
                delete[] send_buf_$arr_id;
             """
        s *= """
              curr_buff_loc_$arr_id = 0;
              for(int i=0; i<total_recv_count_$arr_id; i++) {
                  $key_ctype key = *(($key_ctype*)&recv_buf_$arr_id[curr_buff_loc_$arr_id]);
                  curr_buff_loc_$arr_id += sizeof($key_ctype);
                  $j2c_type val = *(($j2c_type*)&recv_buf_$arr_id[curr_buff_loc_$arr_id]);
                  curr_buff_loc_$arr_id += sizeof($j2c_type);
                  recv_unique_set_$(expr_arr)_tmp_agg_$id.insert(std::make_pair(key, val));
              }
              delete[] recv_buf_$arr_id;
              for(auto& x:recv_unique_set_$(expr_arr)_tmp_agg_$id) {
                  if(unique_map_$(expr_arr)_tmp_agg_$id.find(x.first)==unique_map_$(expr_arr)_tmp_agg_$id.end())
                    unique_map_$(expr_arr)_tmp_agg_$id[x.first] = 1;
                  else
                    unique_map_$(expr_arr)_tmp_agg_$id[x.first] += 1;
              }
              recv_unique_set_$(expr_arr)_tmp_agg_$id.clear();
        """
    else
        s *= " j2c_array< $j2c_type > rbuf_$(id)_$expr_arr = j2c_array< $j2c_type >::new_j2c_array_1d(NULL, rsize_$id);\n"
        s *= """ MPI_Alltoallv($(expr_arr)_tmp_agg_$id.getData(), scount_$id, sdis_$id, $mpi_type,
                                             rbuf_$(id)_$expr_arr.getData(), rcount_$id, rdis_$id, $mpi_type, MPI_COMM_WORLD);
             """
    end
end

function pattern_match_call_rebalance(func::GlobalRef, arr::LHSVar, count::LHSVar, linfo)
    s = ""
    if func==GlobalRef(HPAT.API, :__hpat_arr_rebalance)
        typ = ParallelAccelerator.CGen.getSymType(arr, linfo)
        num_dims = ndims(typ)
        typ = eltype(typ)
        c_typ = ParallelAccelerator.CGen.toCtype(typ)
        c_arr = ParallelAccelerator.CGen.from_expr(arr, linfo)
        c_count = ParallelAccelerator.CGen.from_expr(count, linfo)
        mpi_typ = get_mpi_type_from_var_type(typ)
        s *= "int64_t __hpat_old_size_$c_arr = $c_arr.dims[$num_dims-1];\n"
        # get size of each multidim array row (e.g. row size of matrix)
        s *= "int64_t __hpat_row_size_$c_arr = 1;\n"
        for i in 0:num_dims-2
            s *= "__hpat_row_size_$c_arr *= $c_arr.dims[$i];\n"
        end
        # allocate new array
        s *= "$c_typ *__hpat_tmp_$c_arr = new $c_typ[__hpat_row_size_$c_arr*$c_count];\n"
        # copy old data
        s *= "int64_t __hpat_new_data_ind_$c_arr = 0;\n"
        s *= "#define MIN(x, y) (((x) < (y)) ? (x) : (y))\n"
        s *= "for(int64_t i=0; i<MIN(__hpat_old_size_$c_arr, $c_count); i++) {\n"
        s *= "   for(int64_t j=0; j<__hpat_row_size_$c_arr; j++){\n"
        s *= "    __hpat_tmp_$c_arr[__hpat_new_data_ind_$c_arr] = $c_arr.data[__hpat_new_data_ind_$c_arr];\n"
        s *= "    __hpat_new_data_ind_$c_arr++;\n"
        s *= "    }"
        s *= "}\n"
        # my diff, all diffs
        s *= "int64_t _my_diff_$c_arr = __hpat_old_size_$c_arr-$c_count;\n"
        s *= "int64_t *_all_diff_$c_arr = new int64_t[__hpat_num_pes];\n"
        # s *= "printf(\"__hpat_node_id:%d my_size:%d my_count:%d total_size:%d my_diff:%d\\n\", __hpat_node_id, my_size, my_count, total_size, my_diff);"
        s *= "MPI_Allgather(&_my_diff_$c_arr, 1, MPI_LONG_LONG_INT, _all_diff_$c_arr, 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);\n"
        # printf("__hpat_node_id:%d all_diff[0]:%d all_diff[1]:%d ... all_diff[n-1]:%d\n", __hpat_node_id, all_diff[0], all_diff[1], all_diff[num_pes-1]);
        s *= "MPI_Request *_all_reqs_$c_arr = new MPI_Request[__hpat_num_pes];\n"
        s *= "int _curr_req_$c_arr = 0;\n"
        #// for each potential receiver
        s *= "for(int i=0; i<__hpat_num_pes; i++) {\n"
        #// if receiver
        s *= "  if(_all_diff_$c_arr[i]<0) {\n"
        #// for each potential sender
        s *= "    for(int j=0; j<__hpat_num_pes; j++) {\n"
        #// if sender
        s *= "      if(_all_diff_$c_arr[j]>0) {\n"
        s *= "         int _send_size = MIN(_all_diff_$c_arr[j], -_all_diff_$c_arr[i]);\n"
        #// if I'm receiver
        s *= "         if(__hpat_node_id==i) {\n"
        #//printf("__hpat_node_id:%d receiving from:%d size:%d\n", __hpat_node_id, j, send_size);
        s *= "            MPI_Irecv(&__hpat_tmp_$c_arr[__hpat_new_data_ind_$c_arr], __hpat_row_size_$c_arr*_send_size, $mpi_typ, j, 0, MPI_COMM_WORLD, &_all_reqs_$c_arr[_curr_req_$c_arr++]);\n"
        s *= "            __hpat_new_data_ind_$c_arr += __hpat_row_size_$c_arr*_send_size;\n"
        s *= "         }\n"
        s *= "         if(__hpat_node_id==j) {\n"
        #s *= "            printf("rank:%d sending to:%d size:%d\n", __hpat_node_id, i, send_size);
        s *= "            MPI_Isend(&$c_arr.data[__hpat_new_data_ind_$c_arr], __hpat_row_size_$c_arr*_send_size, $mpi_typ, i, 0, MPI_COMM_WORLD, &_all_reqs_$c_arr[_curr_req_$c_arr++]);\n"
        s *= "            __hpat_new_data_ind_$c_arr += __hpat_row_size_$c_arr*_send_size;\n"
        s *= "         }\n"
        s *= "         _all_diff_$c_arr[i] += _send_size;\n"
        s *= "         _all_diff_$c_arr[j] -= _send_size;\n"
        s *= "         if(_all_diff_$c_arr[i]==0) break;\n"
        s *= "    }\n"
        s *= "   }\n"
        s *= " }\n"
        s *= "}\n"
        s *= "MPI_Waitall(_curr_req_$c_arr, _all_reqs_$c_arr, MPI_STATUSES_IGNORE);\n"
        s *= "delete[] _all_diff_$c_arr;\n"
        s *= "delete[] _all_reqs_$c_arr;\n"
        # delete old array, assign new
        s *= "delete[] $c_arr.data;\n"
        s *= "$c_arr.data = __hpat_tmp_$c_arr;\n"
    end
    return s
end

function pattern_match_call_rebalance(func::ANY, arr::ANY, count::ANY, linfo)
    return ""
end

function gatherv_part_of_gass(table1_cols, table2_cols, id, linfo)
    s = "\n /* Starting Gatherv of gass */ \n"
    table_cols =  [table1_cols; table2_cols]
    for (index, col_name) in enumerate(table_cols)
        table_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        table_gather_displs =  table_col_name * "_gather_displs_t_" * id
        table_gather_rcounts = table_col_name * "_gather_rcounts_t_" * id
        table_gather_num = table_col_name * "_gather_num_t_" * id
        # initialize all the variable for tables
        s *= "int * $table_gather_displs , * $table_gather_rcounts , $table_gather_num;\n"
        s *= "$table_gather_num = $table_col_name.ARRAYLEN();\n"
        # For node 0
        s *= "$table_gather_rcounts = (int *)malloc(__hpat_num_pes_local*sizeof(int));\n"
        s *= "memset ($table_gather_rcounts, 0, sizeof(int)*__hpat_num_pes_local);\n"
        s *= "MPI_Gather( &$table_gather_num, 1, MPI_INT, $table_gather_rcounts, 1, MPI_INT, 0, __hpat_local_comm);\n"
        # For node 0
        s *= "$table_gather_displs = (int *)malloc(__hpat_num_pes_local*sizeof(int));\n"
        s *= "$table_gather_displs[0] = 0;\n"
        s *= "for (int i=1; i<__hpat_num_pes_local; ++i) {\n"
        s *= "$table_gather_displs[i] = $table_gather_displs[i-1]+ $table_gather_rcounts[i-1];\n"
        s *= "}\n"
        table_gather_rsize = table_col_name * "_gather_rsize_t_" * id
        s *= "int $table_gather_rsize = 0;\n"
        s *= "for(int i=0;i< __hpat_num_pes_local;i++){\n"
    	s *= "$table_gather_rsize = $table_gather_rsize + $table_gather_rcounts[i];\n"
        s *= "}\n"
        # Uncomment for debugging
        # s *= "if (__hpat_node_id_local == 0)\n"
        # s *= "std::cout <<\"table1 $table_col_name: At node 0 After gather recieved = \" << $table_gather_rsize << std::endl;\n"
        table_gather_rbuf = table_col_name * "_gather_rbuf_t_" * id
        s *= "j2c_array< int64_t > $table_gather_rbuf = j2c_array<int64_t>::new_j2c_array_1d(NULL, $table_gather_rsize);\n"
        s *= "MPI_Gatherv($table_col_name.getData(), $table_gather_num, MPI_INT64_T, $table_gather_rbuf.getData(), $table_gather_rcounts, $table_gather_displs, MPI_INT64_T, 0, __hpat_local_comm);\n"
        s *= "$table_col_name = $table_gather_rbuf;\n"
    end
    return s
end

function scatterv_part_of_gass(table1_cols, table2_cols, id, linfo)
    s = "\n /* Starting Scatterv of gass */ \n"
    table_cols =  [table1_cols; table2_cols]
    for (index, col_name) in enumerate(table_cols)
        table_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
        table_scatter_count =  table_col_name * "_scatter_count_t_" * id
        table_scatter_displs = table_col_name * "_scatter_displs_t_" * id
        s *= "int * $table_scatter_count, * $table_scatter_displs ;\n"
        s *= "$table_scatter_count = (int*)malloc(__hpat_num_pes_local * sizeof(int));\n"
        s *= "memset ($table_scatter_count, 0, sizeof(int)*__hpat_num_pes_local);\n"
        s *= "for (int i = 1 ; i <  $table_col_name.ARRAYLEN() + 1 ; i++){\n"
        s *= "int node_id = $table_col_name.ARRAYELEM(i) % __hpat_num_pes_local ;\n"
        s *= "$table_scatter_count[node_id]++;\n"
        s *= "}\n"
        table_scatter_input = table_col_name * "_scatter_input_t_" * id
        s *= "int $table_scatter_input = 0;\n"
        s *= "MPI_Scatter($table_scatter_count, 1, MPI_INT, &$table_scatter_input,1,MPI_INT,0, __hpat_local_comm);\n"
        s *= "$table_scatter_displs = (int *)malloc(__hpat_num_pes_local*sizeof(int));\n"
        s *= "$table_scatter_displs[0] = 0;\n"
        s *= "for (int i=1; i<__hpat_num_pes_local; ++i) {\n"
        s *= "$table_scatter_displs[i] = $table_scatter_displs[i-1]+$table_scatter_count[i-1];\n"
        s *= "}\n"
        table_scatter_rbuf = table_col_name * "_scatter_rbuf_t_" * id
        s *= "j2c_array< int64_t > $table_scatter_rbuf;\n"
        s *= "$table_scatter_rbuf = j2c_array<int64_t>::new_j2c_array_1d(NULL, $table_scatter_input);\n"
        s *= "MPI_Scatterv($table_col_name.getData(), $table_scatter_count , $table_scatter_displs, MPI_INT64_T, $table_scatter_rbuf.getData(), $table_scatter_input, MPI_INT64_T, 0, __hpat_local_comm);\n"
    end
    return s
end
