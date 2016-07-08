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

module CGenPatternMatch

import ParallelAccelerator
import ParallelAccelerator.ParallelIR.DelayedFunc
import CompilerTools.DebugMsg
using CompilerTools.LambdaHandling
using CompilerTools.Helper
DebugMsg.init()

import HPAT

#using Debug

include("cgen-hpat-pattern-match-daal.jl")

function pattern_match_call_dist_init(f::GlobalRef,linfo)
    if f.name==:hpat_dist_init
        return ";"#"MPI_Init(0,0);"
    else
        return ""
    end
end

function pattern_match_call_dist_init(f::Any,linfo)
    return ""
end

function pattern_match_call_dist_init2d(f::GlobalRef,linfo)
    if f.name==:hpat_dist_2d_init
        return """    blacs_setup_( &__hpat_node_id, &__hpat_num_pes);
                      // get default context
                      int i_zero=0, i_one=1, i_negone=-1, ictxt=-1;
                      blacs_get_( &i_negone, &i_zero, &ictxt );
                      int __hpat_2d_dims[2];
                      MPI_Dims_create(__hpat_num_pes, 2, __hpat_2d_dims)
              """
    else
        return ""
    end
end

function pattern_match_call_dist_init2d(f::Any,linfo)
    return ""
end

function pattern_match_call_get_sec_since_epoch(f::GlobalRef,linfo)
    if f.mod == HPAT.Checkpointing && f.name==:hpat_get_sec_since_epoch
        return "MPI_Wtime()"
    else
        return ""
    end
end

function pattern_match_call_get_sec_since_epoch(f::Expr,linfo)
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_get_sec_since_epoch f is call to getfield")
        s *= pattern_match_call_get_sec_since_epoch(GlobalRef(eval(f.args[2]), f.args[3].value))
    end
    return s
end

function pattern_match_call_get_sec_since_epoch(f::Any,linfo)
    return ""
end

function pattern_match_reduce_sum(reductionFunc::DelayedFunc,linfo)
    if reductionFunc.args[1][1].args[2].args[1].name==:add_float || reductionFunc.args[1][1].args[2].args[1].name==:add_int
        return true
    end
    return false
end

function pattern_match_reduce_sum(reductionFunc::GlobalRef,linfo)
    if reductionFunc.name==:add_float || reductionFunc.name==:add_int
        return true
    end
    return false
end

function pattern_match_call_dist_reduce(f::GlobalRef, var::TypedVar, reductionFunc::DelayedFunc, output::LHSVar,linfo)
    if f.name==:hpat_dist_reduce
        mpi_type = ""
        if var.typ==Float64
            mpi_type = "MPI_DOUBLE"
        elseif var.typ==Float32
            mpi_type = "MPI_FLOAT"
        elseif var.typ==Int32
            mpi_type = "MPI_INT"
        elseif var.typ==Int64
            mpi_type = "MPI_LONG_LONG_INT"
        else
            throw("CGen unsupported MPI reduction type")
        end

        mpi_func = ""
        if pattern_match_reduce_sum(reductionFunc, linfo)
            mpi_func = "MPI_SUM"
        else
            throw("CGen unsupported MPI reduction function")
        end

        s="MPI_Reduce(&$(var.name), &$output, 1, $mpi_type, $mpi_func, 0, MPI_COMM_WORLD);"
        # debug print for 1D_sum
        #s*="printf(\"len %d start %d end %d\\n\", parallel_ir_save_array_len_1_1, __hpat_loop_start_2, __hpat_loop_end_3);\n"
        return s
    else
        return ""
    end
end

function pattern_match_call_dist_reduce(f::Any, v::Any, rf::Any, o::Any,linfo)
    return ""
end

function pattern_match_call_dist_portion(f::GlobalRef, total::Union{RHSVar,Int}, div::Union{RHSVar,Int}, num_pes::RHSVar, node_id::LHSVar,linfo)
    s = ""
    if f.name==:__hpat_get_node_portion
        c_total = ParallelAccelerator.CGen.from_expr(total, linfo)
        c_div = ParallelAccelerator.CGen.from_expr(div, linfo)
        s = "(($node_id==$num_pes-1) ? $c_total-$node_id*$c_div : $c_div)"
    end
    return s
end

function pattern_match_call_dist_portion(f::ANY, total::ANY, div::ANY, num_pes::ANY, node_id::ANY,linfo)
    return ""
end

function pattern_match_call_dist_node_end(f::GlobalRef, total::RHSVar, div::RHSVar, num_pes::LHSVar, node_id::LHSVar,linfo)
    s = ""
    if f.name==:__hpat_get_node_end
        c_total = ParallelAccelerator.CGen.from_expr(total, linfo)
        c_div = ParallelAccelerator.CGen.from_expr(div, linfo)
        s = "(($node_id==$num_pes-1) ? $c_total : ($node_id+1)*$c_div)"
    end
    return s
end

function pattern_match_call_dist_node_end(f::ANY, total::ANY, div::ANY, num_pes::ANY, node_id::ANY,linfo)
    return ""
end

function pattern_match_call_dist_allreduce(f::GlobalRef, var::RHSVar, reductionFunc, output::RHSVar, size::Union{RHSVar,Int},linfo)
    if f.name==:hpat_dist_allreduce
        mpi_type = ""
        var = toLHSVar(var)
        c_var = ParallelAccelerator.CGen.from_expr(var, linfo)
        c_output = ParallelAccelerator.CGen.from_expr(output, linfo)
        var_typ = ParallelAccelerator.CGen.getSymType(var, linfo)
        is_array =  var_typ<:Array

        if is_array
            var_typ = eltype(var_typ)
            c_var *= ".data"
            c_output *= ".data"
        else
            c_var = "&"*c_var
            c_output = "&"*c_output
        end
        if var_typ==Float64
            mpi_type = "MPI_DOUBLE"
        elseif var_typ==Float32
            mpi_type = "MPI_FLOAT"
        elseif var_typ==Int32
            mpi_type = "MPI_INT"
        elseif var_typ==Int64
            mpi_type = "MPI_LONG_LONG_INT"
        else
            println("reduction type ", var_typ)
            throw("CGen unsupported MPI reduction type")
        end

        mpi_func = ""
        if pattern_match_reduce_sum(reductionFunc, linfo)
            mpi_func = "MPI_SUM"
        else
            throw("CGen unsupported MPI reduction function")
        end

        s="MPI_Allreduce($c_var, $c_output, $size, $mpi_type, $mpi_func, MPI_COMM_WORLD);"
        # debug print for 1D_sum
        #s*="printf(\"len %d start %d end %d\\n\", parallel_ir_save_array_len_1_1, __hpat_loop_start_2, __hpat_loop_end_3);\n"
        return s
    else
        return ""
    end
end

function pattern_match_call_dist_allreduce(f::Any, v::Any, rf::Any, o::Any, s::Any,linfo)
    return ""
end

function pattern_match_call_dist_bcast(f::GlobalRef, var::RHSVar, size::ANY,linfo)
    @dprintln(3, "pattern_match_call_dist_bcast f = ", f)
    c_size = ParallelAccelerator.CGen.from_expr(size, linfo)
    if f.name==:__hpat_dist_broadcast
        mpi_type = ""
        var = toLHSVar(var)
        c_var = ParallelAccelerator.CGen.from_expr(var, linfo)
        var_typ = ParallelAccelerator.CGen.getSymType(var, linfo)
        is_array =  var_typ<:Array
        if is_array
            var_typ = eltype(var_typ)
            c_var *= ".data"
        else
            c_var = "&"*c_var
        end
        if var_typ==Float64
            mpi_type = "MPI_DOUBLE"
        elseif var_typ==Float32
            mpi_type = "MPI_FLOAT"
        elseif var_typ==Int32
            mpi_type = "MPI_INT"
        elseif var_typ==Int64
            mpi_type = "MPI_LONG_LONG_INT"
        else
            println("reduction type ", var_typ)
            throw("CGen unsupported MPI broadcast type")
        end

        s="MPI_Bcast($c_var, $c_size, $mpi_type, 0, MPI_COMM_WORLD);"
        return s
    else
        return ""
    end
end


function pattern_match_call_dist_bcast(f::Any, v::Any, rf::Any,linfo)
    return ""
end

"""
Generate code for HDF5 file open
"""
function pattern_match_call_data_src_open(f::GlobalRef, id::Int, data_var::Union{RHSVar,AbstractString}, file_name::Union{RHSVar,AbstractString}, arr::RHSVar,linfo)
    s = ""
    if f.name==:__hpat_data_source_HDF5_open
        num::AbstractString = string(id)
        s = "hid_t plist_id_$num = H5Pcreate(H5P_FILE_ACCESS);\n"
        s *= "assert(plist_id_$num != -1);\n"
        s *= "herr_t ret_$num;\n"
        s *= "hid_t file_id_$num;\n"
        s *= "ret_$num = H5Pset_fapl_mpio(plist_id_$num, MPI_COMM_WORLD, MPI_INFO_NULL);\n"
        s *= "assert(ret_$num != -1);\n"
        s *= "file_id_$num = H5Fopen((const char*)"*ParallelAccelerator.CGen.from_expr(file_name, linfo)*".data.data, H5F_ACC_RDONLY, plist_id_$num);\n"
        s *= "assert(file_id_$num != -1);\n"
        s *= "ret_$num = H5Pclose(plist_id_$num);\n"
        s *= "assert(ret_$num != -1);\n"
        s *= "hid_t dataset_id_$num;\n"
        s *= "dataset_id_$num = H5Dopen2(file_id_$num, "*ParallelAccelerator.CGen.from_expr(data_var, linfo)*", H5P_DEFAULT);\n"
        s *= "assert(dataset_id_$num != -1);\n"
    elseif f.name==:__hpat_data_sink_HDF5_open
      num = string(id)
      s = "hid_t plist_id_$num = H5Pcreate(H5P_FILE_ACCESS);\n"
      s *= "assert(plist_id_$num != -1);\n"
      s *= "herr_t ret_$num;\n"
      s *= "hid_t file_id_$num;\n"
      s *= "ret_$num = H5Pset_fapl_mpio(plist_id_$num, MPI_COMM_WORLD, MPI_INFO_NULL);\n"
      s *= "assert(ret_$num != -1);\n"
      s *= "file_id_$num = H5Fcreate((const char*)"*ParallelAccelerator.CGen.from_expr(file_name, linfo)*".data.data, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_$num);\n"
      s *= "assert(file_id_$num != -1);\n"
      s *= "ret_$num = H5Pclose(plist_id_$num);\n"
      s *= "assert(ret_$num != -1);\n"
#      s *= "hid_t dataset_id_$num;\n"
      #s *= "dataset_id_$num = H5Dcreate(file_id_$num, "*ParallelAccelerator.CGen.from_expr(data_var, linfo)*", H5P_DEFAULT);\n"
      #s *= "assert(dataset_id_$num != -1);\n"
    end
    return s
end

function pattern_match_call_data_src_open(f::Any, v::Any, rf::Any, o::Any, arr::Any,linfo)
    return ""
end

"""
Generate code for HDF5 file close
"""
function pattern_match_call_data_src_close(f::GlobalRef, id::Int,linfo)
    s = ""
    if f.name==:__hpat_data_source_HDF5_close || f.name==:__hpat_data_sink_HDF5_close
        num::AbstractString = string(id)

        s *= "H5Dclose(dataset_id_$num);\n"
        s *= "H5Fclose(file_id_$num);\n"
    elseif f.name==:__hpat_data_source_TXT_close
        num = string(id)
        s *= "MPI_File_close(&dsrc_txt_file_$num);\n"
    end
    return s
end

function pattern_match_call_data_src_close(f::Any, v::Any,linfo)
    return ""
end

function pattern_match_call_filter_seq(linfo,f::GlobalRef,cond_e, num_cols,table_cols...)
  s = ""
  if f.name==:__hpat_filter
    # its an array of array. array[2:end] and table_cols... notation does that
    table_cols = table_cols[1]
    # For unique counter variables of filter
    filter_rand = string(convert(Int8, round(rand() * 100)))
    # assuming that all columns are of same size in a table
    array_length = "array_length" * filter_rand
    s *= "int $array_length = " * ParallelAccelerator.CGen.from_expr(table_cols[1],linfo) * ".ARRAYLEN();\n"
    # Calculate final filtered array length
    write_index = "write_index" * filter_rand
    s *= "int $write_index = 1;\n"
    cond_e_arr = ParallelAccelerator.CGen.from_expr(cond_e, linfo)
    s *= "for (int index = 1 ; index < $array_length + 1 ; index++) { \n"
    s *= "if ( $cond_e_arr.ARRAYELEM(index) ){\n"
    # If condition satisfy copy all columns values
    for col_name in table_cols
      arr_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
      s *= "$arr_col_name.ARRAYELEM($write_index) =  $arr_col_name.ARRAYELEM(index); \n"
    end
    s *= "$write_index = $write_index + 1;\n"
    s *= "};\n" # if condition
    s *= "};\n" # for loop
    # Change the size of each array
    for col_name in table_cols
      arr_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
      s *= "$arr_col_name.dims[0] =  $write_index - 1; \n"
    end
  end
  return s
end

function pattern_match_call_filter_seq(linfo,f::Any,cond_e, num_cols,table_cols...)
    return ""
end

function pattern_match_call_join_seq(linfo, f::GlobalRef,table_new_cols_len, table1_cols_len, table2_cols_len, table_columns...)
  s = ""
  if f.name==:__hpat_join
    # its an array of array. array[2:end] and table_cols... notation does that
    table_columns = table_columns[1]
    # extract columns of each table
    table_new_cols = table_columns[1:table_new_cols_len]
    table1_cols = table_columns[table_new_cols_len+1:table_new_cols_len+table1_cols_len]
    table2_cols = table_columns[table_new_cols_len+table1_cols_len+1:end]
    join_rand = string(convert(Int8, round(rand() * 100)))
    # assuming that all columns are of same size in a table
    # Also output table's length would be sum of both table length
    t1c1_length_join = "t1c1_length_join"*join_rand
    t2c1_length_join = "t2c1_length_join"*join_rand
    joined_table_length = "joined_table_length"*join_rand
    t1_c1_join = ParallelAccelerator.CGen.from_expr(table1_cols[1],linfo)
    t2_c1_join = ParallelAccelerator.CGen.from_expr(table2_cols[1],linfo)
    s *= "int $t1c1_length_join = $t1_c1_join.ARRAYLEN() ;\n "
    s *= "int $t2c1_length_join = $t2_c1_join.ARRAYLEN() ;\n "
    s *= "int $joined_table_length = $t2c1_length_join + $t2c1_length_join ;\n "
    # Instantiation of output table
    for col_name in table_new_cols
      arr_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
      s *= "$arr_col_name = j2c_array<int64_t>::new_j2c_array_1d(NULL, $joined_table_length);\n"
    end
    # Assuming that join is always on the first column of tables
    # Nested for loop implementation of join
    c_cond_sym = "=="
    table_new_counter_join = "table_new_counter_join" *join_rand
    s *= "int $table_new_counter_join = 1 ; \n"
    s *= "for (int table1_index = 1 ; table1_index < $t1c1_length_join+1 ; table1_index++) { \n"
    s *= "for (int table2_index = 1 ; table2_index < $t2c1_length_join+1 ; table2_index++) { \n" #
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
  end
  return s
end

function pattern_match_call_join_seq(linfo, f::Any,table_new_len, table1_len, table2_len, table_columns...)
    return ""
end

function pattern_match_call_join(linfo, f::GlobalRef,table_new_cols_len, table1_cols_len, table2_cols_len, table_columns...)
    s = ""
    if f.name==:__hpat_join
        # its an array of array. array[2:end] and table_cols... notation does that
        table_columns = table_columns[1]
        # extract columns of each table
        table_new_cols = table_columns[1:table_new_cols_len]
        table1_cols = table_columns[table_new_cols_len+1:table_new_cols_len+table1_cols_len]
        table2_cols = table_columns[table_new_cols_len+table1_cols_len+1:end]
        join_rand = string(convert(Int8, round(rand() * 100)))

        # Sending counts for both tables
        scount_t1 = "scount_t1_"*join_rand
        scount_t2 = "scount_t2_"*join_rand
        s *= "int * $scount_t1;\n"
        s *= "int * $scount_t2;\n"
        s *= "$scount_t1 = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
        s *= "memset ($scount_t1, 0, sizeof(int)*__hpat_num_pes);\n"
        s *= "$scount_t2 = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
        s *= "memset ($scount_t2, 0, sizeof(int)*__hpat_num_pes);\n"

        # Receiving counts for both tables
        rsize_t1 = "rsize_t1_"*join_rand
        rsize_t2 = "rsize_t2_"*join_rand
        s *= "int  $rsize_t1 = 0;\n"
        s *= "int  $rsize_t2 = 0;\n"

        rcount_t1 = "rcount_t1_"*join_rand
        rcount_t2 = "rcount_t2_"*join_rand
        s *= "int * $rcount_t1;\n"
        s *= "int * $rcount_t2;\n"
        s *= "$rcount_t1 = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
        s *= "$rcount_t2 = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"

        # Displacement arrays for both tables
        sdis_t1 = "sdis_t1_"*join_rand
        rdis_t1 = "rdis_t1_"*join_rand
        s *= "int * $sdis_t1;\n"
        s *= "int * $rdis_t1;\n"
        s *= "$sdis_t1 = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
        s *= "$rdis_t1 = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
        sdis_t2 = "sdis_t2_"*join_rand
        rdis_t2 = "rdis_t2_"*join_rand
        s *= "int * $sdis_t2;\n"
        s *= "int * $rdis_t2;\n"
        s *= "$sdis_t2 = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"
        s *= "$rdis_t2 = (int*)malloc(sizeof(int)*__hpat_num_pes);\n"

        t1c1_length_join = "t1c1_length_join"*join_rand
        t2c1_length_join = "t2c1_length_join"*join_rand

        t1_c1_join = ParallelAccelerator.CGen.from_expr(table1_cols[1],linfo)
        t2_c1_join = ParallelAccelerator.CGen.from_expr(table2_cols[1],linfo)
        s *= "int $t1c1_length_join = $t1_c1_join.ARRAYLEN() ;\n "
        s *= "int $t2c1_length_join = $t2_c1_join.ARRAYLEN() ;\n "

        # Sorting data in buffers so that we can distribute using mpi_alltoallv
        # Sorting is based on hash which will essentially do hash partitioning
        # TODO optimize this with fast sorting or hashtables
        s *= "for (int i = 1 ; i <  $t1c1_length_join + 1 ; i++){\n"
        s *= "for (int j = 1 ; j < $t1c1_length_join ; j++ ){\n"
        s *= "int hash1 = $t1_c1_join.ARRAYELEM(j) % __hpat_num_pes ;\n"
        s *= "int hash2 = $t1_c1_join.ARRAYELEM(j+1) % __hpat_num_pes;\n"
        s *= "if (hash1 > hash2){\n"
        for (index, col_name) in enumerate(table1_cols)
            table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
            s *= "int temp_int_$table1_col_name = $table1_col_name.ARRAYELEM(j); \n"
            s *= "$table1_col_name.ARRAYELEM(j) = $table1_col_name.ARRAYELEM(j+1);\n"
            s *= "$table1_col_name.ARRAYELEM(j+1) = temp_int_$table1_col_name;\n"
        end
        s *="}}};\n"

        s *= "for (int i = 1 ; i <  $t2c1_length_join + 1 ; i++){\n"
        s *= "for (int j = 1 ; j < $t2c1_length_join ; j++ ){\n"
        s *= "int hash1 = $t2_c1_join.ARRAYELEM(j) % __hpat_num_pes ;\n"
        s *= "int hash2 = $t2_c1_join.ARRAYELEM(j+1) % __hpat_num_pes;\n"
        s *= "if (hash1 > hash2){\n"
        for (index, col_name) in enumerate(table2_cols)
            table2_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
            s *= "int temp_int_$table2_col_name = $table2_col_name.ARRAYELEM(j); \n"
            s *= "$table2_col_name.ARRAYELEM(j) = $table2_col_name.ARRAYELEM(j+1);\n"
            s *= "$table2_col_name.ARRAYELEM(j+1) = temp_int_$table2_col_name;\n"
        end
        s *="}}};\n"

        s *= "for (int i = 1 ; i <  $t1c1_length_join + 1 ; i++){\n"
        s *= "int node_id = $t1_c1_join.ARRAYELEM(i) % __hpat_num_pes ;\n"
        s *= "$scount_t1[node_id]++;\n"
        s *= "}\n"

        s *= "for (int i = 1 ; i <  $t2c1_length_join + 1 ; i++){\n"
        s *= "int node_id = $t2_c1_join.ARRAYELEM(i) % __hpat_num_pes ;\n"
        s *= "$scount_t2[node_id]++;\n"
        s *= "}\n"

        s *= "MPI_Alltoall($scount_t1,1,MPI_INT,$rcount_t1,1,MPI_INT,MPI_COMM_WORLD);\n"
        s *= "MPI_Alltoall($scount_t2,1,MPI_INT,$rcount_t2,1,MPI_INT,MPI_COMM_WORLD);\n"

        # Caculating displacements for both tables
        s *= """
              $sdis_t1[0]=0;
              $rdis_t1[0]=0;
              $sdis_t2[0]=0;
              $rdis_t2[0]=0;
              for(int i=1;i < __hpat_num_pes;i++){
                  $sdis_t1[i]=$scount_t1[i-1] + $sdis_t1[i-1];
                  $rdis_t1[i]=$rcount_t1[i-1] + $rdis_t1[i-1];
                  $sdis_t2[i]=$scount_t2[i-1] + $sdis_t2[i-1];
                  $rdis_t2[i]=$rcount_t2[i-1] + $rdis_t2[i-1];
              }
        """

        # Summing receiving counts
        s *= """
            for(int i=0;i<__hpat_num_pes;i++){
                $rsize_t1=$rsize_t1 + $rcount_t1[i];
                $rsize_t2=$rsize_t2 + $rcount_t2[i];
              }
        """

        for (index, col_name) in enumerate(table1_cols)
            table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
            s *= " j2c_array< int64_t > rbuf_$table1_col_name = j2c_array<int64_t>::new_j2c_array_1d(NULL, $rsize_t1);\n"
            s *= """ MPI_Alltoallv($table1_col_name.getData(), $scount_t1, $sdis_t1, MPI_INT64_T,
                                 rbuf_$table1_col_name.getData(), $rcount_t1, $rdis_t1, MPI_INT64_T, MPI_COMM_WORLD);
                 """
            s *= " $table1_col_name = rbuf_$table1_col_name; \n"
        end

        for (index, col_name) in enumerate(table2_cols)
            table2_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
            s *= " j2c_array< int64_t > rbuf_$table2_col_name = j2c_array<int64_t>::new_j2c_array_1d(NULL, $rsize_t2);\n"
            s *= """ MPI_Alltoallv($table2_col_name.getData(), $scount_t2, $sdis_t2, MPI_INT64_T,
                                 rbuf_$table2_col_name.getData(), $rcount_t2, $rdis_t2, MPI_INT64_T, MPI_COMM_WORLD);
                 """
            s *= " $table2_col_name = rbuf_$table2_col_name; \n"
        end

        table_new_counter_join = "table_new_counter_join" *join_rand
        s *= "int $table_new_counter_join = 1 ; \n"
        count = 0;
        # Initiatilizing new table arrays
        for (index, col_name) in enumerate(table1_cols)
            table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index],linfo)
            s *= "$table_new_col_name = j2c_array<int64_t>::new_j2c_array_1d(NULL, $rsize_t1 + $rsize_t2);\n"
            count = count + 1
        end
        for (index, col_name) in enumerate(table2_cols)
            if index == 1
                continue
            end
            table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index+count-1],linfo)
            s *= "$table_new_col_name = j2c_array<int64_t>::new_j2c_array_1d(NULL, $rsize_t1 + $rsize_t2);\n"
        end
        s *= "for(int i=1 ;i< $rsize_t1 + 1; i++){\n"
        s *= "for(int j=1 ;j < $rsize_t2 + 1; j++){\n"
        s *= "if($t1_c1_join.ARRAYELEM(i) == $t2_c1_join.ARRAYELEM(j)){\n"
        # For debugging
        s *= "printf(\"MATCHED %ld \\n\", $t1_c1_join.ARRAYELEM(i));\n"
        count = 0
        for (index, col_name) in enumerate(table1_cols)
            table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index],linfo)
            table1_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
            s *= "$table_new_col_name.ARRAYELEM($table_new_counter_join) = $table1_col_name.ARRAYELEM(i); \n"
            count = count + 1
        end
        for (index, col_name) in enumerate(table2_cols)
            if index == 1
                continue
            end
            table_new_col_name = ParallelAccelerator.CGen.from_expr(table_new_cols[index+count-1],linfo)
            table2_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
            s *= "$table_new_col_name.ARRAYELEM($table_new_counter_join) =  $table2_col_name.ARRAYELEM(j); \n"
        end
        s *= "$table_new_counter_join++;\n"
        s *= "}}}\n"
        for col_name in table_new_cols
            table_new_col_name = ParallelAccelerator.CGen.from_expr(col_name,linfo)
            s *= "$table_new_col_name.dims[0] = $table_new_counter_join - 1;\n"
        end
    end
    return s
end

function pattern_match_call_join(linfo, f::Any,table_new_len, table1_len, table2_len, table_columns...)
    return ""
end

function pattern_match_call_agg_seq(linfo, f::GlobalRef, groupby_key, num_exprs, expr_func_output_list...)
  s = ""
  if f.name==:__hpat_aggregate
    expr_func_output_list = expr_func_output_list[1]
    exprs_list = expr_func_output_list[1:num_exprs]
    funcs_list = expr_func_output_list[num_exprs+1:(2*num_exprs)]
    agg_rand = string(convert(Int8, round(rand() * 100)))
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
      s *= "$arr_col_name = j2c_array<int64_t>::new_j2c_array_1d(NULL, $agg_key_map_temp.size());\n"
    end
    # copy back the values from map into arrays
    counter_agg = "counter_agg$agg_rand"
    s *= "int $counter_agg = 1;\n"
    s *= "for(auto i : $agg_key_map_temp){\n"
    for (index, value) in enumerate(output_cols_list)
      map_name = ParallelAccelerator.CGen.from_expr(value, linfo)
      s *= "$map_name.ARRAYELEM($counter_agg) = temp_map_$map_name[i.first];\n"
    end
    s *= "$counter_agg++;\n"
    s *= "}\n"
    # Debugging
    # s *= "for (int i = 1 ; i < counter_agg ; i++){ std::cout << pcustomer_i_classpid3.ARRAYELEM(i) << std::endl;}\n"
  end
  return s
end

function pattern_match_call_agg_seq(linfo, f::Any, groupby_key, num_exprs, exprs_func_list...)
    return ""
end

"""
Generate code for get checkpoint time.
"""
function pattern_match_call_get_checkpoint_time(f::GlobalRef, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_get_checkpoint_time f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_get_checkpoint_time
        s *= "__hpat_get_checkpoint_time(" * ParallelAccelerator.CGen.from_expr(id, linfo) * ")"
    end
    @dprintln(3, "pattern_match_call_get_checkpoint_time done s = ", s)
    return s
end

function pattern_match_call_get_checkpoint_time(f::Expr, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_get_checkpoint_time f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_get_checkpoint_time f is call to getfield")
        s *= pattern_match_call_get_checkpoint_time(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_get_checkpoint_time(f::Any, id::Any,linfo)
    @dprintln(3, "pattern_match_call_get_checkpoint_time f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end


"""
Generate code for start checkpoint.
"""
function pattern_match_call_start_checkpoint(f::GlobalRef, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_start_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_start_checkpoint
        @dprintln(3, "pattern_match_call_start_checkpoint doing replacement")
        s *= "__hpat_start_checkpoint(" * ParallelAccelerator.CGen.from_expr(id, linfo) * ")"
    end
    @dprintln(3, "pattern_match_call_start_checkpoint done s = ", s)
    return s
end

function pattern_match_call_start_checkpoint(f::Expr, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_start_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_start_checkpoint f is call to getfield")
        s *= pattern_match_call_start_checkpoint(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_start_checkpoint(f::Any, id::Any,linfo)
    @dprintln(3, "pattern_match_call_start_checkpoint f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end

"""
Generate code for finish checkpoint.
"""
function pattern_match_call_finish_checkpoint(f::GlobalRef, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_finish_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_finish_checkpoint_region
        s *= "__hpat_finish_checkpoint_region(" * ParallelAccelerator.CGen.from_expr(id, linfo) * ")"
    end
    @dprintln(3, "pattern_match_call_finish_checkpoint done s = ", s)
    return s
end

function pattern_match_call_finish_checkpoint(f::Expr, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_finish_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_finish_checkpoint f is call to getfield")
        s *= pattern_match_call_finish_checkpoint(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_finish_checkpoint(f::Any, id::Any,linfo)
    @dprintln(3, "pattern_match_call_finish_checkpoint f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end


"""
Generate code for end checkpoint.
"""
function pattern_match_call_end_checkpoint(f::GlobalRef, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_end_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_end_checkpoint
        s *= "__hpat_end_checkpoint(" * ParallelAccelerator.CGen.from_expr(id, linfo) * ")"
    end
    @dprintln(3, "pattern_match_call_end_checkpoint done s = ", s)
    return s
end

function pattern_match_call_end_checkpoint(f::Expr, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_end_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_end_checkpoint f is call to getfield")
        s *= pattern_match_call_end_checkpoint(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_end_checkpoint(f::Any, id::Any,linfo)
    @dprintln(3, "pattern_match_call_end_checkpoint f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end

"""
Generate code for checkpointing a single program element.
"""
function pattern_match_call_value_checkpoint(f::GlobalRef, id::Union{Int,RHSVar}, value::RHSVar,linfo)
    @dprintln(3, "pattern_match_call_value_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_value_checkpoint
        s *= "__hpat_value_checkpoint(" * ParallelAccelerator.CGen.from_expr(id, linfo) * "," * ParallelAccelerator.CGen.from_expr(value, linfo) * ")"
    end
    @dprintln(3, "pattern_match_call_value_checkpoint done s = ", s)
    return s
end

function pattern_match_call_value_checkpoint(f::Expr, id::Union{Int,RHSVar}, value::RHSVar,linfo)
    @dprintln(3, "pattern_match_call_value_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_value_checkpoint f is call to getfield")
        s *= pattern_match_call_value_checkpoint(GlobalRef(eval(f.args[2]), f.args[3].value), id, value)
    end
    return s
end

function pattern_match_call_value_checkpoint(f::Any, id::Any, value::Any,linfo)
    @dprintln(3, "pattern_match_call_value_checkpoint f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    return ""
end

function pattern_match_call_restore_checkpoint_start(f::GlobalRef, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_restore_checkpoint_start f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_checkpoint_restore_start
        s *= "__hpat_restore_checkpoint_start(" * ParallelAccelerator.CGen.from_expr(id, linfo) * ")"
    end
    @dprintln(3, "pattern_match_call_restore_checkpoint_start done s = ", s)
    return s
end

function pattern_match_call_restore_checkpoint_start(f::Expr, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_restore_checkpoint_start f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_restore_checkpoint_start f is call to getfield")
        s *= pattern_match_call_restore_checkpoint_start(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_restore_checkpoint_start(f::Any, id::Any,linfo)
    @dprintln(3, "pattern_match_call_restore_checkpoint_start f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end

"""
Generate code for end checkpoint.
"""
function pattern_match_call_restore_checkpoint_end(f::GlobalRef, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_restore_checkpoint_end f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_checkpoint_restore_end
        s *= "__hpat_restore_checkpoint_end(" * ParallelAccelerator.CGen.from_expr(id, linfo) * ")"
    end
    @dprintln(3, "pattern_match_call_restore_checkpoint_end done s = ", s)
    return s
end

function pattern_match_call_restore_checkpoint_end(f::Expr, id::Union{Int,RHSVar},linfo)
    @dprintln(3, "pattern_match_call_restore_checkpoint_end f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_restore_checkpoint_end f is call to getfield")
        s *= pattern_match_call_restore_checkpoint_end(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_restore_checkpoint_end(f::Any, id::Any,linfo)
    @dprintln(3, "pattern_match_call_restore_checkpoint_end f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end

"""
Generate code for checkpointing a single program element.
"""
function pattern_match_call_restore_checkpoint_value(f::GlobalRef, id::Union{Int,RHSVar}, value::RHSVar,linfo)
    @dprintln(3, "pattern_match_call_restore_checkpoint_value f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_checkpoint_restore_value
        s *= "__hpat_restore_checkpoint_value(" * ParallelAccelerator.CGen.from_expr(id, linfo) * "," * ParallelAccelerator.CGen.from_expr(value, linfo) * ")"
    end
    @dprintln(3, "pattern_match_call_restore_checkpoint_value done s = ", s)
    return s
end

function pattern_match_call_restore_checkpoint_value(f::Expr, id::Union{Int,RHSVar}, value::RHSVar,linfo)
    @dprintln(3, "pattern_match_call_restore_checkpoint_value f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_restore_checkpoint_value f is call to getfield")
        s *= pattern_match_call_restore_checkpoint_value(GlobalRef(eval(f.args[2]), f.args[3].value), id, value)
    end
    return s
end

function pattern_match_call_restore_checkpoint_value(f::Any, id::Any, value::Any,linfo)
    @dprintln(3, "pattern_match_call_restore_checkpoint_value f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    return ""
end

"""
Generate code for text file open (no variable name input)
"""
function pattern_match_call_data_src_open(f::GlobalRef, id::Int, file_name::Union{RHSVar,AbstractString}, arr,linfo)
    s = ""
    if f.name==:__hpat_data_source_TXT_open
        num::AbstractString = string(id)
        file_name_str::AbstractString = ParallelAccelerator.CGen.from_expr(file_name, linfo)
        s = """
            MPI_File dsrc_txt_file_$num;
            int ierr_$num = MPI_File_open(MPI_COMM_WORLD, $file_name_str.data.data, MPI_MODE_RDONLY, MPI_INFO_NULL, &dsrc_txt_file_$num);
            assert(ierr_$num==0);
            """
    end
    return s
end

function pattern_match_call_data_src_open(f::Any, rf::Any, o::Any, arr::Any,linfo)
    return ""
end

# sequential read for debugging
function pattern_match_call_data_src_read_seq(f::GlobalRef, id::Int, arr::RHSVar,linfo)
    s = ""
    num::AbstractString = string(id)

    if f.name==:__hpat_data_source_HDF5_read
        data_typ = eltype(ParallelAccelerator.CGen.getSymType(arr, linfo))
        h5_typ = ""
        carr = ParallelAccelerator.CGen.from_expr(toLHSVar(arr), linfo)
        if data_typ==Float64
            h5_typ = "H5T_NATIVE_DOUBLE"
        elseif data_typ==Float32
            h5_typ = "H5T_NATIVE_FLOAT"
        elseif data_typ==Int32
            h5_typ = "H5T_NATIVE_INT"
        elseif data_typ==Int64
            h5_typ = "H5T_NATIVE_LLONG"
        else
            println("g5 data type ", data_typ)
            throw("CGen unsupported HDF5 data type")
        end

        # assuming 1st dimension is partitined
        s =  "hsize_t CGen_HDF5_start_$num[data_ndim_$num];\n"
        s *= "hsize_t CGen_HDF5_count_$num[data_ndim_$num];\n"
        s *= "CGen_HDF5_start_$num[0] = 0;\n"
        s *= "CGen_HDF5_count_$num[0] = space_dims_$num[0];\n"
        s *= "for(int i_CGen_dim=1; i_CGen_dim<data_ndim_$num; i_CGen_dim++) {\n"
        s *= "CGen_HDF5_start_$num[i_CGen_dim] = 0;\n"
        s *= "CGen_HDF5_count_$num[i_CGen_dim] = space_dims_$num[i_CGen_dim];\n"
        s *= "}\n"
        #s *= "std::cout<<\"read size \"<<CGen_HDF5_start_$num[0]<<\" \"<<CGen_HDF5_count_$num[0]<<\" \"<<CGen_HDF5_start_$num[1]<<\" \"<<CGen_HDF5_count_$num[1]<<std::endl;\n"
        s *= "ret_$num = H5Sselect_hyperslab(space_id_$num, H5S_SELECT_SET, CGen_HDF5_start_$num, NULL, CGen_HDF5_count_$num, NULL);\n"
        s *= "assert(ret_$num != -1);\n"
        s *= "hid_t mem_dataspace_$num = H5Screate_simple (data_ndim_$num, CGen_HDF5_count_$num, NULL);\n"
        s *= "assert (mem_dataspace_$num != -1);\n"
        s *= "hid_t xfer_plist_$num = H5Pcreate (H5P_DATASET_XFER);\n"
        s *= "assert(xfer_plist_$num != -1);\n"
        s *= "double h5_read_start_$num = MPI_Wtime();\n"
        s *= "ret_$num = H5Dread(dataset_id_$num, $h5_typ, mem_dataspace_$num, space_id_$num, xfer_plist_$num, $carr.getData());\n"
        s *= "assert(ret_$num != -1);\n"
        #s*="if(__hpat_node_id==__hpat_num_pes/2) printf(\"h5 read %lf\\n\", MPI_Wtime()-h5_read_start_$num);\n"
        s *= ";\n"
    end
    return s
end

function pattern_match_call_data_src_read_seq(f::Any, v::Any, rf::Any, linfo)
    return ""
end

function pattern_match_call_data_src_read(f::GlobalRef, id::Int, arr::RHSVar, start::LHSVar, count::LHSVar,linfo)
    s = ""
    num::AbstractString = string(id)

    if f.name==:__hpat_data_source_HDF5_read
        data_typ = eltype(ParallelAccelerator.CGen.getSymType(arr, linfo))
        h5_typ = ""
        carr = ParallelAccelerator.CGen.from_expr(toLHSVar(arr), linfo)
        if data_typ==Float64
            h5_typ = "H5T_NATIVE_DOUBLE"
        elseif data_typ==Float32
            h5_typ = "H5T_NATIVE_FLOAT"
        elseif data_typ==Int32
            h5_typ = "H5T_NATIVE_INT"
        elseif data_typ==Int64
            h5_typ = "H5T_NATIVE_LLONG"
        else
            println("g5 data type ", data_typ)
            throw("CGen unsupported HDF5 data type")
        end

        # assuming 1st dimension is partitined
        s =  "hsize_t CGen_HDF5_start_$num[data_ndim_$num];\n"
        s *= "hsize_t CGen_HDF5_count_$num[data_ndim_$num];\n"
        s *= "CGen_HDF5_start_$num[0] = $start;\n"
        s *= "CGen_HDF5_count_$num[0] = $count;\n"
        s *= "for(int i_CGen_dim=1; i_CGen_dim<data_ndim_$num; i_CGen_dim++) {\n"
        s *= "CGen_HDF5_start_$num[i_CGen_dim] = 0;\n"
        s *= "CGen_HDF5_count_$num[i_CGen_dim] = space_dims_$num[i_CGen_dim];\n"
        s *= "}\n"
        #s *= "std::cout<<\"read size \"<<CGen_HDF5_start_$num[0]<<\" \"<<CGen_HDF5_count_$num[0]<<\" \"<<CGen_HDF5_start_$num[1]<<\" \"<<CGen_HDF5_count_$num[1]<<std::endl;\n"
        s *= "ret_$num = H5Sselect_hyperslab(space_id_$num, H5S_SELECT_SET, CGen_HDF5_start_$num, NULL, CGen_HDF5_count_$num, NULL);\n"
        s *= "assert(ret_$num != -1);\n"
        s *= "hid_t mem_dataspace_$num = H5Screate_simple (data_ndim_$num, CGen_HDF5_count_$num, NULL);\n"
        s *= "assert (mem_dataspace_$num != -1);\n"
        s *= "hid_t xfer_plist_$num = H5Pcreate (H5P_DATASET_XFER);\n"
        s *= "assert(xfer_plist_$num != -1);\n"
        s *= "double h5_read_start_$num = MPI_Wtime();\n"
        s *= "ret_$num = H5Dread(dataset_id_$num, $h5_typ, mem_dataspace_$num, space_id_$num, xfer_plist_$num, $carr.getData());\n"
        s *= "assert(ret_$num != -1);\n"
        #s*="if(__hpat_node_id==__hpat_num_pes/2) printf(\"h5 read %lf\\n\", MPI_Wtime()-h5_read_start_$num);\n"
        s *= ";\n"
    elseif f.name==:__hpat_data_source_TXT_read
        # assuming 1st dimension is partitined
        data_typ = eltype(ParallelAccelerator.CGen.getSymType(arr, linfo))
        t_typ = ParallelAccelerator.CGen.toCtype(data_typ)

        s = """
            int64_t CGen_txt_start_$num = $start;
            int64_t CGen_txt_count_$num = $count;
            int64_t CGen_txt_end_$num = $start+$count;


            // std::cout<<"rank: "<<__hpat_node_id<<" start: "<<CGen_txt_start_$num<<" end: "<<CGen_txt_end_$num<<" columnSize: "<<CGen_txt_col_size_$num<<std::endl;
            // if data needs to be sent left
            // still call MPI_Send if first character is new line
            int64_t CGen_txt_left_send_size_$num = 0;
            int64_t CGen_txt_tmp_curr_start_$num = CGen_txt_curr_start_$num;
            if(CGen_txt_start_$num>CGen_txt_curr_start_$num)
            {
                while(CGen_txt_tmp_curr_start_$num!=CGen_txt_start_$num)
                {
                    while(CGen_txt_buffer_$num[CGen_txt_left_send_size_$num]!=\'\\n\')
                        CGen_txt_left_send_size_$num++;
                    CGen_txt_left_send_size_$num++; // account for \n
                    CGen_txt_tmp_curr_start_$num++;
                }
            }
            MPI_Request CGen_txt_MPI_request1_$num, CGen_txt_MPI_request2_$num;
            MPI_Status CGen_txt_MPI_status_$num;
            // send left
            if(__hpat_node_id!=0)
            {
                MPI_Isend(&CGen_txt_left_send_size_$num, 1, MPI_LONG_LONG_INT, __hpat_node_id-1, 0, MPI_COMM_WORLD, &CGen_txt_MPI_request1_$num);
                MPI_Isend(CGen_txt_buffer_$num, CGen_txt_left_send_size_$num, MPI_CHAR, __hpat_node_id-1, 1, MPI_COMM_WORLD, &CGen_txt_MPI_request2_$num);
                // std::cout<<"rank: "<<__hpat_node_id<<" sent left "<<CGen_txt_left_send_size_$num<<std::endl;
            }

            char* CGen_txt_right_buff_$num = NULL;
            int64_t CGen_txt_right_recv_size_$num = 0;
            // receive from right
            if(__hpat_node_id!=__hpat_num_pes-1)
            {
                MPI_Recv(&CGen_txt_right_recv_size_$num, 1, MPI_LONG_LONG_INT, __hpat_node_id+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CGen_txt_right_buff_$num = new char[CGen_txt_right_recv_size_$num];
                MPI_Recv(CGen_txt_right_buff_$num, CGen_txt_right_recv_size_$num, MPI_CHAR, __hpat_node_id+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // std::cout<<"rank: "<<__hpat_node_id<<" received right "<<CGen_txt_right_recv_size_$num<<std::endl;
            }

            if(__hpat_node_id!=0)
            {
                MPI_Wait(&CGen_txt_MPI_request1_$num, &CGen_txt_MPI_status_$num);
                MPI_Wait(&CGen_txt_MPI_request2_$num, &CGen_txt_MPI_status_$num);
            }

            // if data needs to be sent right
            // still call MPI_Send if first character is new line
            int64_t CGen_txt_right_send_size_$num = 0;
            int64_t CGen_txt_tmp_curr_end_$num = CGen_txt_curr_end_$num;
            if(__hpat_node_id!=__hpat_num_pes-1 && CGen_txt_curr_end_$num>=CGen_txt_end_$num)
            {
                while(CGen_txt_tmp_curr_end_$num!=CGen_txt_end_$num-1)
                {
                    // -1 to account for \0
                    while(CGen_txt_buffer_$num[CGen_txt_buff_size_$num-CGen_txt_right_send_size_$num-1]!=\'\\n\')
                        CGen_txt_right_send_size_$num++;
                    CGen_txt_tmp_curr_end_$num--;
                    // corner case, last line doesn't have \'\\n\'
                    if (CGen_txt_tmp_curr_end_$num!=CGen_txt_end_$num-1)
                        CGen_txt_right_send_size_$num++; // account for \n
                }
            }
            // send right
            if(__hpat_node_id!=__hpat_num_pes-1)
            {
                MPI_Isend(&CGen_txt_right_send_size_$num, 1, MPI_LONG_LONG_INT, __hpat_node_id+1, 0, MPI_COMM_WORLD, &CGen_txt_MPI_request1_$num);
                MPI_Isend(CGen_txt_buffer_$num+CGen_txt_buff_size_$num-CGen_txt_right_send_size_$num, CGen_txt_right_send_size_$num, MPI_CHAR, __hpat_node_id+1, 1, MPI_COMM_WORLD, &CGen_txt_MPI_request2_$num);
            }
            char* CGen_txt_left_buff_$num = NULL;
            int64_t CGen_txt_left_recv_size_$num = 0;
            // receive from left
            if(__hpat_node_id!=0)
            {
                MPI_Recv(&CGen_txt_left_recv_size_$num, 1, MPI_LONG_LONG_INT, __hpat_node_id-1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                CGen_txt_left_buff_$num = new char[CGen_txt_left_recv_size_$num];
                MPI_Recv(CGen_txt_left_buff_$num, CGen_txt_left_recv_size_$num, MPI_CHAR, __hpat_node_id-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // std::cout<<"rank: "<<__hpat_node_id<<" received left "<<CGen_txt_left_recv_size_$num<<std::endl;
            }
            if(__hpat_node_id!=__hpat_num_pes-1)
            {
                MPI_Wait(&CGen_txt_MPI_request1_$num, &CGen_txt_MPI_status_$num);
                MPI_Wait(&CGen_txt_MPI_request2_$num, &CGen_txt_MPI_status_$num);
                // std::cout<<"rank: "<<__hpat_node_id<<" sent right "<<CGen_txt_right_send_size_$num<<std::endl;
            }

            // int64_t total_data_size = (CGen_txt_end_$num-CGen_txt_start_$num)*CGen_txt_col_size_$num;
            // double *my_data = new double[total_data_size];
            int64_t CGen_txt_data_ind_$num = 0;

            char CGen_txt_sep_char_$num[] = \"\\n\";
            int64_t CGen_txt_curr_row_$num = 0;
            $t_typ * CGen_txt_data_arr = ($t_typ *)$arr.getData();
            while(CGen_txt_curr_row_$num!=CGen_txt_count_$num)
            {
                char* CGen_txt_line;
                if (CGen_txt_curr_row_$num==0)
                {
                    CGen_txt_line = strtok(CGen_txt_buffer_$num, CGen_txt_sep_char_$num);
                    if(CGen_txt_left_recv_size_$num!=0)
                    {
                        char *CGen_txt_tmp_line;
                        CGen_txt_tmp_line = new char[CGen_txt_left_recv_size_$num+strlen(CGen_txt_line)];
                        memcpy(CGen_txt_tmp_line, CGen_txt_left_buff_$num, CGen_txt_left_recv_size_$num);
                        memcpy(CGen_txt_tmp_line+CGen_txt_left_recv_size_$num, CGen_txt_line, strlen(CGen_txt_line));
                        CGen_txt_line = CGen_txt_tmp_line;
                    }
                }
                else if(CGen_txt_curr_row_$num==CGen_txt_count_$num-1)
                {
                    CGen_txt_line = strtok(NULL, CGen_txt_sep_char_$num);
                    if(CGen_txt_right_recv_size_$num!=0)
                    {
                        char *CGen_txt_tmp_line;
                        CGen_txt_tmp_line = new char[CGen_txt_right_recv_size_$num+strlen(CGen_txt_line)];
                        memcpy(CGen_txt_tmp_line, CGen_txt_line, strlen(CGen_txt_line));
                        memcpy(CGen_txt_tmp_line+strlen(CGen_txt_line), CGen_txt_right_buff_$num, CGen_txt_right_recv_size_$num);
                        CGen_txt_line = CGen_txt_tmp_line;
                    }
                }
                else
                {
                    CGen_txt_line = strtok(NULL, CGen_txt_sep_char_$num);
                }
                // parse line separately, not to change strtok's state
                for(int64_t i=0; i<CGen_txt_col_size_$num; i++)
                {
                    if(i==0)
                        CGen_txt_data_arr[CGen_txt_data_ind_$num++] = strtod(CGen_txt_line,&CGen_txt_line);
                    else
                        CGen_txt_data_arr[CGen_txt_data_ind_$num++] = strtod(CGen_txt_line+1,&CGen_txt_line);
         //           std::cout<<$arr[CGen_txt_data_ind_$num-1]<<std::endl;
                }
                CGen_txt_curr_row_$num++;
            }

         //   MPI_File_close(&dsrc_txt_file_$num);
            """
    end
    return s
end

function pattern_match_call_data_src_read(f::Any, v::Any, rf::Any, o::Any, arr::Any,linfo)
    return ""
end

function pattern_match_call_dist_h5_size(f::GlobalRef, size_arr::LHSVar, ind::Union{Int64,RHSVar},linfo)
    s = ""
    if f.name==:__hpat_get_H5_dim_size || f.name==:__hpat_get_TXT_dim_size
        @dprintln(3,"match dist_dim_size ",f," ", size_arr, " ",ind)
        s = ParallelAccelerator.CGen.from_expr(size_arr, linfo)*"["*ParallelAccelerator.CGen.from_expr(ind, linfo)*"-1]"
    end
    return s
end

function pattern_match_call_dist_h5_size(f::Any, size_arr::Any, ind::Any,linfo)
    return ""
end


function pattern_match_call_data_src_read_2d(f::GlobalRef, id::Int, arr::RHSVar,
              start_x::LHSVar, start_y::LHSVar, stride_x::LHSVar, stride_y::LHSVar,
              count_x::LHSVar, count_y::LHSVar, block_x::LHSVar, block_y::LHSVar,
              local_size_x::LHSVar, local_size_y::LHSVar, linfo)
    s = ""
    num::AbstractString = string(id)

    if f.name==:__hpat_data_source_HDF5_read
        data_typ = eltype(ParallelAccelerator.CGen.getSymType(arr, linfo))
        h5_typ = ""
        carr = ParallelAccelerator.CGen.from_expr(toLHSVar(arr), linfo)
        if data_typ==Float64
            h5_typ = "H5T_NATIVE_DOUBLE"
        elseif data_typ==Float32
            h5_typ = "H5T_NATIVE_FLOAT"
        elseif data_typ==Int32
            h5_typ = "H5T_NATIVE_INT"
        elseif data_typ==Int64
            h5_typ = "H5T_NATIVE_LLONG"
        else
            println("g5 data type ", data_typ)
            throw("CGen unsupported HDF5 data type")
        end

        # assuming 1st and 2nd dimensions are partitined
        # hyperslab call input variables
        s =  "hsize_t CGen_HDF5_start_$num[data_ndim_$num];\n"
        s =  "hsize_t CGen_HDF5_stride_$num[data_ndim_$num];\n"
        s *= "hsize_t CGen_HDF5_count_$num[data_ndim_$num];\n"
        s =  "hsize_t CGen_HDF5_block_$num[data_ndim_$num];\n"
        # last 2 dimensions are set using distributed-pass data
        s *= "CGen_HDF5_start_$num[0] = $start_y;\n"
        s *= "CGen_HDF5_start_$num[1] = $start_x;\n"
        s *= "CGen_HDF5_stride_$num[0] = $stride_y;\n"
        s *= "CGen_HDF5_stride_$num[1] = $stride_x;\n"
        s *= "CGen_HDF5_count_$num[0] = $count_y;\n"
        s *= "CGen_HDF5_count_$num[1] = $count_x;\n"
        s *= "CGen_HDF5_block_$num[0] = $block_y;\n"
        s *= "CGen_HDF5_block_$num[1] = $block_x;\n"

        # rest of dimensions, if any, are not divided
        s *= "for(int i_CGen_dim=2; i_CGen_dim<data_ndim_$num; i_CGen_dim++) {\n"
        s *= "  CGen_HDF5_start_$num[i_CGen_dim] = 0;\n"
        s *= "  CGen_HDF5_stride_$num[i_CGen_dim] = 1;\n"
        s *= "  CGen_HDF5_count_$num[i_CGen_dim] = space_dims_$num[i_CGen_dim];\n"
        s *= "  CGen_HDF5_block_$num[i_CGen_dim] = 1;\n"
        s *= "}\n"
        #s *= "std::cout<<\"read size \"<<CGen_HDF5_start_$num[0]<<\" \"<<CGen_HDF5_count_$num[0]<<\" \"<<CGen_HDF5_start_$num[1]<<\" \"<<CGen_HDF5_count_$num[1]<<std::endl;\n"
        s *= "ret_$num = H5Sselect_hyperslab(space_id_$num, H5S_SELECT_SET, CGen_HDF5_start_$num,
                         CGen_HDF5_stride_$num, CGen_HDF5_count_$num, CGen_HDF5_block_$num);\n"
        s *= "assert(ret_$num != -1);\n"

        # size of memory to read to
        s =  "hsize_t CGen_HDF5_memsize_$num[data_ndim_$num];\n"
        s *= "CGen_HDF5_memsize_$num[0] = $local_size_y;\n"
        s *= "CGen_HDF5_memsize_$num[1] = $local_size_x;\n"

        s *= "hid_t mem_dataspace_$num = H5Screate_simple (data_ndim_$num, CGen_HDF5_memsize_$num, NULL);\n"
        s *= "assert (mem_dataspace_$num != -1);\n"
        s *= "hid_t xfer_plist_$num = H5Pcreate (H5P_DATASET_XFER);\n"
        s *= "assert(xfer_plist_$num != -1);\n"
        s *= "double h5_read_start_$num = MPI_Wtime();\n"
        s *= "ret_$num = H5Dread(dataset_id_$num, $h5_typ, mem_dataspace_$num, space_id_$num, xfer_plist_$num, $carr.getData());\n"
        s *= "assert(ret_$num != -1);\n"
        #s*="if(__hpat_node_id==__hpat_num_pes/2) printf(\"h5 read %lf\\n\", MPI_Wtime()-h5_read_start_$num);\n"
        s *= ";\n"
      end
      return s
end

function pattern_match_call_data_src_read_2d(f::ANY, id::ANY, arr::ANY,
              start_x::ANY, start_y::ANY, stride_x::ANY, stride_y::ANY,
              count_x::ANY, count_y::ANY, block_x::ANY, block_y::ANY,
              local_size_x::ANY, local_size_y::ANY, linfo)
  return ""
end

function pattern_match_call_data_sink_write(f::GlobalRef, id::Int, hdf5_var, arr::RHSVar, start::LHSVar, count::LHSVar,tot_size,linfo)
    s = ""
    num::AbstractString = string(id)

    if f.name==:__hpat_data_sink_HDF5_write
        arr_typ = ParallelAccelerator.CGen.getSymType(arr, linfo)
        num_dims = ndims(arr_typ)
        data_typ = eltype(arr_typ)
        h5_typ = ""
        carr = ParallelAccelerator.CGen.from_expr(toLHSVar(arr), linfo)
        if data_typ==Float64
            h5_typ = "H5T_NATIVE_DOUBLE"
        elseif data_typ==Float32
            h5_typ = "H5T_NATIVE_FLOAT"
        elseif data_typ==Int32
            h5_typ = "H5T_NATIVE_INT"
        elseif data_typ==Int64
            h5_typ = "H5T_NATIVE_LLONG"
        else
            println("g5 data type ", data_typ)
            throw("CGen unsupported HDF5 data type")
        end

        # create dataset
        s *= " hid_t dataset_id_$num;\n"
        s *= " hid_t  filespace_$num, memspace_$num;\n"
        s *= " hsize_t  dataset_dims_$num[$num_dims];\n"
        #s *= " for(int i=0; i<$num_dims; i++) dataset_dims_$num[i]=$(ParallelAccelerator.CGen.from_expr(tot_size[i],linfo));\n"
        for i in 1:length(tot_size)
            s*= "dataset_dims_$num[$i-1]=$(ParallelAccelerator.CGen.from_expr(tot_size[i],linfo));\n"
        end
        s *= "  filespace_$num = H5Screate_simple($num_dims, dataset_dims_$num, NULL);\n"
        s *= "  dataset_id_$num = H5Dcreate(file_id_$num, \"$hdf5_var\", $h5_typ, filespace_$num,\n"
        s *=  "     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);\n"
        s *= " H5Sclose(filespace_$num);\n"
        # assuming 1st dimension is partitined
        s *=  "hsize_t CGen_HDF5_start_$num[$num_dims];\n"
        s *= "hsize_t CGen_HDF5_count_$num[$num_dims];\n"
        s *= "CGen_HDF5_start_$num[0] = $start;\n"
        s *= "CGen_HDF5_count_$num[0] = $count;\n"
        #s *= "for(int i_CGen_dim=1; i_CGen_dim<$num_dims; i_CGen_dim++) {\n"
        for i in 1:length(tot_size)-1
            s *= "  CGen_HDF5_start_$num[$i] = 0;\n"
            s *= "  CGen_HDF5_count_$num[$i] = $(ParallelAccelerator.CGen.from_expr(tot_size[i],linfo));\n"
        end
        #s *= "}\n"
        #s *= "std::cout<<\"read size \"<<CGen_HDF5_start_$num[0]<<\" \"<<CGen_HDF5_count_$num[0]<<\" \"<<CGen_HDF5_start_$num[1]<<\" \"<<CGen_HDF5_count_$num[1]<<std::endl;\n"
        s *= "filespace_$num = H5Dget_space(dataset_id_$num);\n"
        s *= "ret_$num = H5Sselect_hyperslab(filespace_$num, H5S_SELECT_SET, CGen_HDF5_start_$num, NULL, CGen_HDF5_count_$num, NULL);\n"
        s *= "assert(ret_$num != -1);\n"
        s *= "hid_t mem_dataspace_$num = H5Screate_simple ($num_dims, CGen_HDF5_count_$num, NULL);\n"
        s *= "assert (mem_dataspace_$num != -1);\n"
        s *= "hid_t xfer_plist_$num = H5Pcreate (H5P_DATASET_XFER);\n"
        s *= "assert(xfer_plist_$num != -1);\n"
        s *= "double h5_read_start_$num = MPI_Wtime();\n"
        s *= "H5Pset_dxpl_mpio(xfer_plist_$num, H5FD_MPIO_COLLECTIVE);\n"
        s *= "ret_$num = H5Dwrite(dataset_id_$num, $h5_typ, mem_dataspace_$num, filespace_$num, xfer_plist_$num, $carr.getData());\n"
        s *= "assert(ret_$num != -1);\n"
        #s*="if(__hpat_node_id==__hpat_num_pes/2) printf(\"h5 read %lf\\n\", MPI_Wtime()-h5_read_start_$num);\n"
        s *= ";\n"
    end
    return s
end

function pattern_match_call_data_sink_write(f::ANY, id::ANY, hdf5_var, arr::ANY, tot_size::ANY, start::ANY, count::ANY,linfo)
    return ""
end

function pattern_match_call(ast::Array{Any, 1}, linfo)

  @dprintln(3,"hpat pattern matching ",ast)
  s = ""
  if length(ast)==1
    @dprintln(3,"ast1_typ = ", typeof(ast[1]))
    s *= pattern_match_call_dist_init(ast[1], linfo)
    s *= pattern_match_call_dist_init2d(ast[1], linfo)
    s *= pattern_match_call_get_sec_since_epoch(ast[1], linfo)
  elseif length(ast)==2
    @dprintln(3,"ast1_typ = ", typeof(ast[1]), " ast2_typ = ", typeof(ast[2]))
    s *= pattern_match_call_data_src_close(ast[1], ast[2], linfo)
    s *= pattern_match_call_get_checkpoint_time(ast[1], ast[2], linfo)
    s *= pattern_match_call_start_checkpoint(ast[1], ast[2], linfo)
    s *= pattern_match_call_end_checkpoint(ast[1], ast[2], linfo)
    s *= pattern_match_call_finish_checkpoint(ast[1], ast[2], linfo)
    s *= pattern_match_call_restore_checkpoint_end(ast[1], ast[2], linfo)
  elseif length(ast)==3
    @dprintln(3,"ast1_typ = ", typeof(ast[1]), " ast2_typ = ", typeof(ast[2]), " ast3_typ = ", typeof(ast[3]))
    s *= pattern_match_call_dist_h5_size(ast[1],ast[2],ast[3], linfo)
    s *= pattern_match_call_dist_bcast(ast[1],ast[2],ast[3], linfo)
    s *= pattern_match_call_value_checkpoint(ast[1], ast[2], ast[3], linfo)
    s *= pattern_match_call_restore_checkpoint_start(ast[1], ast[2], linfo)
    s *= pattern_match_call_restore_checkpoint_value(ast[1], ast[2], ast[3], linfo)
    s *= pattern_match_call_data_src_read_seq(ast[1],ast[2],ast[3], linfo)
  elseif length(ast)==4
    s *= pattern_match_call_dist_reduce(ast[1],ast[2],ast[3], ast[4], linfo)
    # text file read
    s *= pattern_match_call_data_src_open(ast[1],ast[2],ast[3], ast[4], linfo)
  elseif length(ast)==5
    # HDF5 open
    s *= pattern_match_call_data_src_open(ast[1],ast[2],ast[3], ast[4], ast[5], linfo)
    s *= pattern_match_call_data_src_read(ast[1],ast[2],ast[3], ast[4], ast[5], linfo)
    s *= pattern_match_call_dist_allreduce(ast[1],ast[2],ast[3], ast[4], ast[5], linfo)
    s *= pattern_match_call_dist_portion(ast[1],ast[2],ast[3], ast[4], ast[5], linfo)
    s *= pattern_match_call_dist_node_end(ast[1],ast[2],ast[3], ast[4], ast[5], linfo)
  elseif length(ast)==7
    s *= pattern_match_call_data_sink_write(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7], linfo)
  elseif length(ast)==8
    s *= pattern_match_call_kmeans(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8], linfo)
  elseif length(ast)==12
    s *= pattern_match_call_linear_regression(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12], linfo)
  elseif length(ast)==13
    s *= pattern_match_call_naive_bayes(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12],ast[13], linfo)
    s *= pattern_match_call_data_src_read_2d(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12],ast[13], linfo)
  end
  if length(ast)>=4
    s *= pattern_match_call_filter_seq(linfo, ast[1], ast[2], ast[3], ast[4:end])
    s *= pattern_match_call_agg_seq(linfo, ast[1], ast[2], ast[3], ast[4:end])
  end
  if length(ast)>=5
    s *= pattern_match_call_join(linfo, ast[1], ast[2], ast[3], ast[4],ast[5:end])
  end
  return s
end


function from_assignment_match_dist(lhs::RHSVar, rhs::Expr, linfo)
    @dprintln(3, "assignment pattern match dist2: ",lhs," = ",rhs)
    s = ""
    local num::AbstractString
    if rhs.head==:call && (isa(rhs.args[1],GlobalRef) || isa(rhs.args[1],TopNode)) && rhs.args[1].name==:__hpat_data_source_HDF5_size
        num = ParallelAccelerator.CGen.from_expr(rhs.args[2], linfo)
        s = "hid_t space_id_$num = H5Dget_space(dataset_id_$num);\n"
        s *= "assert(space_id_$num != -1);\n"
        s *= "hsize_t data_ndim_$num = H5Sget_simple_extent_ndims(space_id_$num);\n"
        s *= "hsize_t space_dims_$num[data_ndim_$num];\n"
        s *= "H5Sget_simple_extent_dims(space_id_$num, space_dims_$num, NULL);\n"
        s *= ParallelAccelerator.CGen.from_expr(lhs, linfo)*" = space_dims_$num;"
    elseif rhs.head==:call && length(rhs.args)==1 && rhs.args[1]==GlobalRef(HPAT.API,:hpat_dist_num_pes)
        @dprintln(3, "num_pes call")
        c_lhs = ParallelAccelerator.CGen.from_expr(lhs, linfo)
        return "MPI_Comm_size(MPI_COMM_WORLD,&$c_lhs);"
    elseif rhs.head==:call && length(rhs.args)==1 && rhs.args[1]==GlobalRef(HPAT.API,:hpat_dist_num_pes_x)
          @dprintln(3, "num_pes_x call")
          c_lhs = ParallelAccelerator.CGen.from_expr(lhs, linfo)
          return "$c_lhs = __hpat_2d_dims[0];"
    elseif rhs.head==:call && length(rhs.args)==1 && rhs.args[1]==GlobalRef(HPAT.API,:hpat_dist_num_pes_y)
          @dprintln(3, "num_pes_x call")
          c_lhs = ParallelAccelerator.CGen.from_expr(lhs, linfo)
          return """$c_lhs = __hpat_2d_dims[1];\n // create row-major 2D grid
                    blacs_gridinit_( &ictxt, "R", &__hpat_num_pes_x, &__hpat_num_pes_y );
                """
    elseif rhs.head==:call && length(rhs.args)==1 && rhs.args[1]==GlobalRef(HPAT.API,:hpat_dist_node_id)
        @dprintln(3, "node_id call")
        c_lhs = ParallelAccelerator.CGen.from_expr(lhs, linfo)
        return "MPI_Comm_rank(MPI_COMM_WORLD,&$c_lhs);"
    elseif rhs.head==:call && length(rhs.args)==1 && rhs.args[1]==GlobalRef(HPAT.API,:hpat_dist_node_id_x)
          @dprintln(3, "node_id_x call")
          c_lhs = ParallelAccelerator.CGen.from_expr(lhs, linfo)
          return "blacs_gridinfo_( &ictxt, &__hpat_num_pes_x, &__hpat_num_pes_y, &__hpat_node_id_x, &__hpat_node_id_y );"
    elseif rhs.head==:call && length(rhs.args)==1 && rhs.args[1]==GlobalRef(HPAT.API,:hpat_dist_node_id_y)
      return ";"
    elseif rhs.head==:call && length(rhs.args)==1 && isExpr(rhs.args[1])
        @dprintln(3, "one arg call to an Expr")
        expr = rhs.args[1]
        if expr.head == :call && expr.args[1] == TopNode(:getfield)
            this_mod = eval(expr.args[2])
            if this_mod == HPAT.Checkpointing
                c_lhs = ParallelAccelerator.CGen.from_expr(lhs, linfo)
                return assignment_call_internal(c_lhs, expr.args[3].value, linfo)
            end
        end
    elseif rhs.head==:call && (isa(rhs.args[1],GlobalRef) || isa(rhs.args[1],TopNode)) && rhs.args[1].name==:__hpat_data_source_TXT_size
        num = ParallelAccelerator.CGen.from_expr(rhs.args[2], linfo)
        c_lhs = ParallelAccelerator.CGen.from_expr(lhs, linfo)
        s = """
            MPI_Offset CGen_txt_tot_file_size_$num;
            MPI_Offset CGen_txt_buff_size_$num;
            MPI_Offset CGen_txt_offset_start_$num;
            MPI_Offset CGen_txt_offset_end_$num;

            /* divide file read */
            MPI_File_get_size(dsrc_txt_file_$num, &CGen_txt_tot_file_size_$num);
            CGen_txt_buff_size_$num = CGen_txt_tot_file_size_$num/__hpat_num_pes;
            CGen_txt_offset_start_$num = __hpat_node_id * CGen_txt_buff_size_$num;
            CGen_txt_offset_end_$num   = CGen_txt_offset_start_$num + CGen_txt_buff_size_$num - 1;
            if (__hpat_node_id == __hpat_num_pes-1)
                CGen_txt_offset_end_$num = CGen_txt_tot_file_size_$num;
            CGen_txt_buff_size_$num =  CGen_txt_offset_end_$num - CGen_txt_offset_start_$num + 1;

            char* CGen_txt_buffer_$num = new char[CGen_txt_buff_size_$num+1];

            MPI_File_read_at_all(dsrc_txt_file_$num, CGen_txt_offset_start_$num, CGen_txt_buffer_$num, CGen_txt_buff_size_$num, MPI_CHAR, MPI_STATUS_IGNORE);
            CGen_txt_buffer_$num[CGen_txt_buff_size_$num] = \'\\0\';

            // make sure new line is there for last line
            if(__hpat_node_id == __hpat_num_pes-1 && CGen_txt_buffer_$num[CGen_txt_buff_size_$num-2]!=\'\\n\')
                CGen_txt_buffer_$num[CGen_txt_buff_size_$num-1]=\'\\n\';

            // count number of new lines
            int64_t CGen_txt_num_lines_$num = 0;
            int64_t CGen_txt_char_index_$num = 0;
            while (CGen_txt_buffer_$num[CGen_txt_char_index_$num]!=\'\\0\') {
                if(CGen_txt_buffer_$num[CGen_txt_char_index_$num]==\'\\n\')
                    CGen_txt_num_lines_$num++;
                CGen_txt_char_index_$num++;
            }

            // std::cout<<"rank: "<<__hpat_node_id<<" lines: "<<CGen_txt_num_lines_$num<<" startChar: "<<CGen_txt_buffer_$num[0]<<std::endl;
            // get total number of rows
            int64_t CGen_txt_tot_row_size_$num=0;
            MPI_Allreduce(&CGen_txt_num_lines_$num, &CGen_txt_tot_row_size_$num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            // std::cout<<"total rows: "<<CGen_txt_tot_row_size_$num<<std::endl;

            // count number of values in a column
            // 1D data has CGen_txt_col_size_$num==1
            int64_t CGen_txt_col_size_$num = 1;
            CGen_txt_char_index_$num = 0;
            while (CGen_txt_buffer_$num[CGen_txt_char_index_$num]!=\'\\0\' && CGen_txt_buffer_$num[CGen_txt_char_index_$num]!=\'\\n\')
                CGen_txt_char_index_$num++;
            CGen_txt_char_index_$num++;
            while (CGen_txt_buffer_$num[CGen_txt_char_index_$num]!=\'\\0\' && CGen_txt_buffer_$num[CGen_txt_char_index_$num]!=\'\\n\') {
                if(CGen_txt_buffer_$num[CGen_txt_char_index_$num]==',')
                    CGen_txt_col_size_$num++;
                CGen_txt_char_index_$num++;
            }

            // prefix sum to find current global starting line on this node
            int64_t CGen_txt_curr_start_$num = 0;
            MPI_Scan(&CGen_txt_num_lines_$num, &CGen_txt_curr_start_$num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            int64_t CGen_txt_curr_end_$num = CGen_txt_curr_start_$num;
            CGen_txt_curr_start_$num -= CGen_txt_num_lines_$num; // Scan is inclusive
            if(CGen_txt_col_size_$num==1) {
                $c_lhs = new uint64_t[1];
                $c_lhs[0] = CGen_txt_tot_row_size_$num;
            } else {
                $c_lhs = new uint64_t[2];
                $c_lhs[0] = CGen_txt_tot_row_size_$num;
                $c_lhs[1] = CGen_txt_col_size_$num;
            }
            """
    end
    return s
end

# Utilities
function isTopNode(a::TopNode)
    return true
end

function isTopNode(a::ANY)
    return false
end

function isExpr(a::Expr)
    return true
end

function isExpr(a::ANY)
    return false
end

function from_assignment_match_dist(lhs::Any, rhs::Any, linfo)
    return ""
end


function return_j2c_array_name(table_name,table_column)
    return "p"* string(table_name) * "p" * string(table_column)
end

function return_pound_array_name(table_name,table_column)
    return "#"* string(table_name) * "#" * string(table_column)
end

function return_reduction_string_with_closure(agg_key_col_input,expr_arr,agg_map,func)
    s = ""
    if string(func) == "Main.length"
        s *= "if ($agg_map.find($agg_key_col_input.ARRAYELEM(i)) == $agg_map.end())\n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] = 1;\n"
        s *= "else \n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] += 1;\n\n"
    elseif string(func) == "Main.sum"
        s *= "if ($agg_map.find($agg_key_col_input.ARRAYELEM(i)) == $agg_map.end())\n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] = $expr_arr.ARRAYELEM(i) ;\n"
        s *= "else \n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] +=  $expr_arr.ARRAYELEM(i)  ;\n\n"
    elseif string(func) == "Main.max"
        s *= "if ($agg_map.find($agg_key_col_input.ARRAYELEM(i)) == $agg_map.end())){\n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] = $expr_arr.ARRAYELEM(i) ;}\n"
        s *= "else{ \n"
        s *= "if (agg_map_count[$agg_key_col_input.ARRAYELEM(i)] < $expr_arr.ARRAYELEM(i) ) \n"
        s *= "$agg_map[$agg_key_col_input.ARRAYELEM(i)] = $expr_arr.ARRAYELEM(i) ;}\n\n"
    end
    return s
end

end # module
