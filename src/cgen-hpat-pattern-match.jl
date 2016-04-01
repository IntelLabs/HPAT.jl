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

function pattern_match_call_dist_init(f::TopNode)
    if f.name==:hpat_dist_init
        return ";"#"MPI_Init(0,0);"
    else
        return ""
    end
end

function pattern_match_call_dist_init(f::Any)
    return ""
end

function pattern_match_call_get_sec_since_epoch(f::GlobalRef)
    if f.mod == HPAT.Checkpointing && f.name==:hpat_get_sec_since_epoch
        return "MPI_Wtime();"
    else
        return ""
    end
end

function pattern_match_call_get_sec_since_epoch(f::Expr)
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_get_sec_since_epoch f is call to getfield")
        s *= pattern_match_call_get_sec_since_epoch(GlobalRef(eval(f.args[2]), f.args[3].value))
    end
    return s
end

function pattern_match_call_get_sec_since_epoch(f::Any)
    return ""
end

function pattern_match_reduce_sum(reductionFunc::DelayedFunc)
    if reductionFunc.args[1][1].args[2].args[1]==TopNode(:add_float) || reductionFunc.args[1][1].args[2].args[1]==TopNode(:add_int)
        return true
    end
    return false
end

function pattern_match_reduce_sum(reductionFunc::TopNode)
    if reductionFunc.name==:add_float || reductionFunc.name==:add_int
        return true
    end
    return false
end

function pattern_match_call_dist_reduce(f::TopNode, var::SymbolNode, reductionFunc::DelayedFunc, output::Symbol)
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
        if pattern_match_reduce_sum(reductionFunc)
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

function pattern_match_call_dist_reduce(f::Any, v::Any, rf::Any, o::Any)
    return ""
end

function pattern_match_call_dist_portion(f::Symbol, total::Union{SymAllGen,Int}, div::Union{SymAllGen,Int}, num_pes::Symbol, node_id::Symbol)
    s = ""
    if f==:__hpat_get_node_portion
        c_total = ParallelAccelerator.CGen.from_expr(total)
        c_div = ParallelAccelerator.CGen.from_expr(div)
        s = "(($node_id==$num_pes-1) ? $c_total-$node_id*$c_div : $c_div)"
    end
    return s
end

function pattern_match_call_dist_portion(f::ANY, total::ANY, div::ANY, num_pes::ANY, node_id::ANY)
    return ""
end

function pattern_match_call_dist_node_end(f::Symbol, total::SymAllGen, div::SymAllGen, num_pes::Symbol, node_id::Symbol)
    s = ""
    if f==:__hpat_get_node_end
        c_total = ParallelAccelerator.CGen.from_expr(total)
        c_div = ParallelAccelerator.CGen.from_expr(div)
        s = "(($node_id==$num_pes-1) ? $c_total : ($node_id+1)*$c_div)"
    end
    return s
end

function pattern_match_call_dist_node_end(f::ANY, total::ANY, div::ANY, num_pes::ANY, node_id::ANY)
    return ""
end

function pattern_match_call_dist_allreduce(f::TopNode, var::SymAllGen, reductionFunc, output::SymAllGen, size::Union{SymAllGen,Int})
    if f.name==:hpat_dist_allreduce
        mpi_type = ""
        var = toSymGen(var)
        c_var = ParallelAccelerator.CGen.from_expr(var)
        c_output = ParallelAccelerator.CGen.from_expr(output)
        var_typ = ParallelAccelerator.CGen.getSymType(var)
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
        if pattern_match_reduce_sum(reductionFunc)
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

function pattern_match_call_dist_allreduce(f::Any, v::Any, rf::Any, o::Any, s::Any)
    return ""
end

function pattern_match_call_dist_bcast(f::Symbol, var::SymAllGen, size::ANY)
    @dprintln(3, "pattern_match_call_dist_bcast f = ", f)
    c_size = ParallelAccelerator.CGen.from_expr(size)
    if f==:__hpat_dist_broadcast
        mpi_type = ""
        var = toSymGen(var)
        c_var = ParallelAccelerator.CGen.from_expr(var)
        var_typ = ParallelAccelerator.CGen.getSymType(var)
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

function pattern_match_call_dist_bcast(f::GlobalRef, var::SymAllGen, size::ANY)
    @dprintln(3, "pattern_match_call_dist_bcast GlobalRef f = ", f)
    if f.mod == HPAT
        return pattern_match_call_dist_bcast(f.name, var, size)
    end
    return ""
end

function pattern_match_call_dist_bcast(f::Any, v::Any, rf::Any)
    return ""
end

"""
Generate code for HDF5 file open
"""
function pattern_match_call_data_src_open(f::Symbol, id::Int, data_var::Union{SymAllGen,AbstractString}, file_name::Union{SymAllGen,AbstractString}, arr::Symbol)
    s = ""
    if f==:__hpat_data_source_HDF5_open
        num::AbstractString = string(id)
    
        s = "hid_t plist_id_$num = H5Pcreate(H5P_FILE_ACCESS);\n"
        s *= "assert(plist_id_$num != -1);\n"
        s *= "herr_t ret_$num;\n"
        s *= "hid_t file_id_$num;\n"
        s *= "ret_$num = H5Pset_fapl_mpio(plist_id_$num, MPI_COMM_WORLD, MPI_INFO_NULL);\n"
        s *= "assert(ret_$num != -1);\n"
        s *= "file_id_$num = H5Fopen("*ParallelAccelerator.CGen.from_expr(file_name)*", H5F_ACC_RDONLY, plist_id_$num);\n"
        s *= "assert(file_id_$num != -1);\n"
        s *= "ret_$num = H5Pclose(plist_id_$num);\n"
        s *= "assert(ret_$num != -1);\n"
        s *= "hid_t dataset_id_$num;\n"
        s *= "dataset_id_$num = H5Dopen2(file_id_$num, "*ParallelAccelerator.CGen.from_expr(data_var)*", H5P_DEFAULT);\n"
        s *= "assert(dataset_id_$num != -1);\n"
    end
    return s
end

function pattern_match_call_data_src_open(f::Any, v::Any, rf::Any, o::Any, arr::Any)
    return ""
end

"""
Generate code for HDF5 file close 
"""
function pattern_match_call_data_src_close(f::Symbol, id::Int)
    s = ""
    if f==:__hpat_data_source_HDF5_close
        num::AbstractString = string(id)
    
        s *= "H5Dclose(dataset_id_$num);\n"
        s *= "H5Fclose(file_id_$num);\n"
    elseif f==:__hpat_data_source_TXT_close
        num = string(id)
        s *= "MPI_File_close(&dsrc_txt_file_$num);\n"
    end
    return s
end

function pattern_match_call_data_src_close(f::Any, v::Any)
    return ""
end

"""
Generate code for get checkpoint time.
"""
function pattern_match_call_get_checkpoint_time(f::GlobalRef, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_get_checkpoint_time f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_get_checkpoint_time
        s *= "__hpat_get_checkpoint_time(" * ParallelAccelerator.CGen.from_expr(id) * ")"
    end
    @dprintln(3, "pattern_match_call_get_checkpoint_time done s = ", s)
    return s
end

function pattern_match_call_get_checkpoint_time(f::Expr, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_get_checkpoint_time f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_get_checkpoint_time f is call to getfield")
        s *= pattern_match_call_get_checkpoint_time(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_get_checkpoint_time(f::Any, id::Any)
    @dprintln(3, "pattern_match_call_get_checkpoint_time f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end


"""
Generate code for start checkpoint.
"""
function pattern_match_call_start_checkpoint(f::GlobalRef, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_start_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_start_checkpoint
        @dprintln(3, "pattern_match_call_start_checkpoint doing replacement")
        s *= "__hpat_start_checkpoint(" * ParallelAccelerator.CGen.from_expr(id) * ")"
    end
    @dprintln(3, "pattern_match_call_start_checkpoint done s = ", s)
    return s
end

function pattern_match_call_start_checkpoint(f::Expr, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_start_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_start_checkpoint f is call to getfield")
        s *= pattern_match_call_start_checkpoint(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_start_checkpoint(f::Any, id::Any)
    @dprintln(3, "pattern_match_call_start_checkpoint f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end

"""
Generate code for finish checkpoint.
"""
function pattern_match_call_finish_checkpoint(f::GlobalRef, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_finish_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_finish_checkpoint_region
        s *= "__hpat_finish_checkpoint_region(" * ParallelAccelerator.CGen.from_expr(id) * ")"
    end
    @dprintln(3, "pattern_match_call_finish_checkpoint done s = ", s)
    return s
end

function pattern_match_call_finish_checkpoint(f::Expr, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_finish_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_finish_checkpoint f is call to getfield")
        s *= pattern_match_call_finish_checkpoint(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_finish_checkpoint(f::Any, id::Any)
    @dprintln(3, "pattern_match_call_finish_checkpoint f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end


"""
Generate code for end checkpoint.
"""
function pattern_match_call_end_checkpoint(f::GlobalRef, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_end_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_end_checkpoint
        s *= "__hpat_end_checkpoint(" * ParallelAccelerator.CGen.from_expr(id) * ")"
    end
    @dprintln(3, "pattern_match_call_end_checkpoint done s = ", s)
    return s
end

function pattern_match_call_end_checkpoint(f::Expr, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_end_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_end_checkpoint f is call to getfield")
        s *= pattern_match_call_end_checkpoint(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_end_checkpoint(f::Any, id::Any)
    @dprintln(3, "pattern_match_call_end_checkpoint f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end

"""
Generate code for checkpointing a single program element.
"""
function pattern_match_call_value_checkpoint(f::GlobalRef, id::Union{Int,SymAllGen}, value::SymAllGen)
    @dprintln(3, "pattern_match_call_value_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_value_checkpoint
        s *= "__hpat_value_checkpoint(" * ParallelAccelerator.CGen.from_expr(id) * "," * ParallelAccelerator.CGen.from_expr(value) * ")"
    end
    @dprintln(3, "pattern_match_call_value_checkpoint done s = ", s)
    return s
end

function pattern_match_call_value_checkpoint(f::Expr, id::Union{Int,SymAllGen}, value::SymAllGen)
    @dprintln(3, "pattern_match_call_value_checkpoint f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_value_checkpoint f is call to getfield")
        s *= pattern_match_call_value_checkpoint(GlobalRef(eval(f.args[2]), f.args[3].value), id, value)
    end
    return s
end

function pattern_match_call_value_checkpoint(f::Any, id::Any, value::Any)
    @dprintln(3, "pattern_match_call_value_checkpoint f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    return ""
end

function pattern_match_call_restore_checkpoint_start(f::GlobalRef, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_restore_checkpoint_start f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_checkpoint_restore_start
        s *= "__hpat_restore_checkpoint_start(" * ParallelAccelerator.CGen.from_expr(id) * ")"
    end
    @dprintln(3, "pattern_match_call_restore_checkpoint_start done s = ", s)
    return s
end

function pattern_match_call_restore_checkpoint_start(f::Expr, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_restore_checkpoint_start f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_restore_checkpoint_start f is call to getfield")
        s *= pattern_match_call_restore_checkpoint_start(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_restore_checkpoint_start(f::Any, id::Any)
    @dprintln(3, "pattern_match_call_restore_checkpoint_start f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end

"""
Generate code for end checkpoint.
"""
function pattern_match_call_restore_checkpoint_end(f::GlobalRef, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_restore_checkpoint_end f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_checkpoint_restore_end
        s *= "__hpat_restore_checkpoint_end(" * ParallelAccelerator.CGen.from_expr(id) * ")"
    end
    @dprintln(3, "pattern_match_call_restore_checkpoint_end done s = ", s)
    return s
end

function pattern_match_call_restore_checkpoint_end(f::Expr, id::Union{Int,SymAllGen})
    @dprintln(3, "pattern_match_call_restore_checkpoint_end f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_restore_checkpoint_end f is call to getfield")
        s *= pattern_match_call_restore_checkpoint_end(GlobalRef(eval(f.args[2]), f.args[3].value), id)
    end
    return s
end

function pattern_match_call_restore_checkpoint_end(f::Any, id::Any)
    @dprintln(3, "pattern_match_call_restore_checkpoint_end f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id))
    return ""
end

"""
Generate code for checkpointing a single program element.
"""
function pattern_match_call_restore_checkpoint_value(f::GlobalRef, id::Union{Int,SymAllGen}, value::SymAllGen)
    @dprintln(3, "pattern_match_call_restore_checkpoint_value f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    s = ""
    if f.mod == HPAT.Checkpointing && f.name==:hpat_checkpoint_restore_value
        s *= "__hpat_restore_checkpoint_value(" * ParallelAccelerator.CGen.from_expr(id) * "," * ParallelAccelerator.CGen.from_expr(value) * ")"
    end
    @dprintln(3, "pattern_match_call_restore_checkpoint_value done s = ", s)
    return s
end

function pattern_match_call_restore_checkpoint_value(f::Expr, id::Union{Int,SymAllGen}, value::SymAllGen)
    @dprintln(3, "pattern_match_call_restore_checkpoint_value f = ", f, " type = GlobalRef id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    s = ""
    if f.head == :call && f.args[1] == TopNode(:getfield)
        @dprintln(3, "pattern_match_call_restore_checkpoint_value f is call to getfield")
        s *= pattern_match_call_restore_checkpoint_value(GlobalRef(eval(f.args[2]), f.args[3].value), id, value)
    end
    return s
end

function pattern_match_call_restore_checkpoint_value(f::Any, id::Any, value::Any)
    @dprintln(3, "pattern_match_call_restore_checkpoint_value f = ", f, " type = ", typeof(f), " id = ", id, " type = ", typeof(id), " value = ", value, " type = ", typeof(value))
    return ""
end

"""
Generate code for text file open (no variable name input)
"""
function pattern_match_call_data_src_open(f::Symbol, id::Int, file_name::Union{SymAllGen,AbstractString}, arr::Symbol)
    s = ""
    if f==:__hpat_data_source_TXT_open
        num::AbstractString = string(id)
        file_name_str::AbstractString = ParallelAccelerator.CGen.from_expr(file_name)
        s = """
            MPI_File dsrc_txt_file_$num;
            int ierr_$num = MPI_File_open(MPI_COMM_WORLD, $file_name_str, MPI_MODE_RDONLY, MPI_INFO_NULL, &dsrc_txt_file_$num);
            assert(ierr_$num==0);
            """
    end
    return s
end

function pattern_match_call_data_src_open(f::Any, rf::Any, o::Any, arr::Any)
    return ""
end



function pattern_match_call_data_src_read(f::Symbol, id::Int, arr::Symbol, start::Symbol, count::Symbol)
    s = ""
    num::AbstractString = string(id)
    
    if f==:__hpat_data_source_HDF5_read
        data_typ = eltype(ParallelAccelerator.CGen.getSymType(arr))
        h5_typ = ""
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
        s *= "ret_$num = H5Dread(dataset_id_$num, $h5_typ, mem_dataspace_$num, space_id_$num, xfer_plist_$num, $arr.getData());\n"
        s *= "assert(ret_$num != -1);\n"
        #s*="if(__hpat_node_id==__hpat_num_pes/2) printf(\"h5 read %lf\\n\", MPI_Wtime()-h5_read_start_$num);\n"
        s *= ";\n"
    elseif f==:__hpat_data_source_TXT_read
        # assuming 1st dimension is partitined
        data_typ = eltype(ParallelAccelerator.CGen.getSymType(arr))
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

function pattern_match_call_data_src_read(f::Any, v::Any, rf::Any, o::Any, arr::Any)
    return ""
end

function pattern_match_call_dist_h5_size(f::Symbol, size_arr::GenSym, ind::Union{Int64,SymAllGen})
    s = ""
    if f==:__hpat_get_H5_dim_size || f==:__hpat_get_TXT_dim_size
        @dprintln(3,"match dist_dim_size ",f," ", size_arr, " ",ind)
        s = ParallelAccelerator.CGen.from_expr(size_arr)*"["*ParallelAccelerator.CGen.from_expr(ind)*"-1]"
    end
    return s
end

function pattern_match_call_dist_h5_size(f::Any, size_arr::Any, ind::Any)
    return ""
end



function pattern_match_call(ast::Array{Any, 1})

    @dprintln(3,"hpat pattern matching ",ast)
    s = ""
    if length(ast)==1
        @dprintln(3,"ast1_typ = ", typeof(ast[1]))
        s *= pattern_match_call_dist_init(ast[1])
        s *= pattern_match_call_get_sec_since_epoch(ast[1]) 
    elseif length(ast)==2
        @dprintln(3,"ast1_typ = ", typeof(ast[1]), " ast2_typ = ", typeof(ast[2]))
        s *= pattern_match_call_data_src_close(ast[1], ast[2])
        s *= pattern_match_call_get_checkpoint_time(ast[1], ast[2])
        s *= pattern_match_call_start_checkpoint(ast[1], ast[2])
        s *= pattern_match_call_end_checkpoint(ast[1], ast[2])
        s *= pattern_match_call_finish_checkpoint(ast[1], ast[2])
        s *= pattern_match_call_restore_checkpoint_end(ast[1], ast[2])
    elseif(length(ast)==3) 
        @dprintln(3,"ast1_typ = ", typeof(ast[1]), " ast2_typ = ", typeof(ast[2]), " ast3_typ = ", typeof(ast[3]))
        s *= pattern_match_call_dist_h5_size(ast[1],ast[2],ast[3])
        s *= pattern_match_call_dist_bcast(ast[1],ast[2],ast[3])
        s *= pattern_match_call_value_checkpoint(ast[1], ast[2], ast[3])
        s *= pattern_match_call_restore_checkpoint_start(ast[1], ast[2])
        s *= pattern_match_call_restore_checkpoint_value(ast[1], ast[2], ast[3])
    elseif(length(ast)==4)
        s *= pattern_match_call_dist_reduce(ast[1],ast[2],ast[3], ast[4])
        # text file read
        s *= pattern_match_call_data_src_open(ast[1],ast[2],ast[3], ast[4])
    elseif(length(ast)==5)
        # HDF5 open
        s *= pattern_match_call_data_src_open(ast[1],ast[2],ast[3], ast[4], ast[5])
        s *= pattern_match_call_data_src_read(ast[1],ast[2],ast[3], ast[4], ast[5])
        s *= pattern_match_call_dist_allreduce(ast[1],ast[2],ast[3], ast[4], ast[5])
        s *= pattern_match_call_dist_portion(ast[1],ast[2],ast[3], ast[4], ast[5])
        s *= pattern_match_call_dist_node_end(ast[1],ast[2],ast[3], ast[4], ast[5])
    elseif(length(ast)==8)
        s *= pattern_match_call_kmeans(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8])
    elseif(length(ast)==12)
        s *= pattern_match_call_linear_regression(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12])
    elseif(length(ast)==13)
        s *= pattern_match_call_naive_bayes(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12],ast[13])
    end
    return s
end

function assignment_call_internal(c_lhs, dist_call)
    @dprintln(3, "assignment_call_internal c_lhs = ", c_lhs, " dist_call = ", dist_call)
    if dist_call==:hpat_dist_num_pes
        return "MPI_Comm_size(MPI_COMM_WORLD,&$c_lhs);"
    elseif dist_call==:hpat_dist_node_id
        return "MPI_Comm_rank(MPI_COMM_WORLD,&$c_lhs);"
    end
    return ""
end

function from_assignment_match_dist(lhs::SymAllGen, rhs::Expr)
    @dprintln(3, "assignment pattern match dist2: ",lhs," = ",rhs)
    s = ""
    local num::AbstractString
    if rhs.head==:call && rhs.args[1]==:__hpat_data_source_HDF5_size
        num = ParallelAccelerator.CGen.from_expr(rhs.args[2])
        s = "hid_t space_id_$num = H5Dget_space(dataset_id_$num);\n"    
        s *= "assert(space_id_$num != -1);\n"    
        s *= "hsize_t data_ndim_$num = H5Sget_simple_extent_ndims(space_id_$num);\n"
        s *= "hsize_t space_dims_$num[data_ndim_$num];\n"    
        s *= "H5Sget_simple_extent_dims(space_id_$num, space_dims_$num, NULL);\n"
        s *= ParallelAccelerator.CGen.from_expr(lhs)*" = space_dims_$num;"
    elseif rhs.head==:call && length(rhs.args)==1 && isTopNode(rhs.args[1])
        @dprintln(3, "one arg call to a TopNode")
        dist_call = rhs.args[1].name
        c_lhs = ParallelAccelerator.CGen.from_expr(lhs)
        return assignment_call_internal(c_lhs, dist_call)
    elseif rhs.head==:call && length(rhs.args)==1 && isExpr(rhs.args[1])
        @dprintln(3, "one arg call to an Expr")
        expr = rhs.args[1]
        if expr.head == :call && expr.args[1] == TopNode(:getfield)
            this_mod = eval(expr.args[2])
            if this_mod == HPAT.Checkpointing
                c_lhs = ParallelAccelerator.CGen.from_expr(lhs)
                return assignment_call_internal(c_lhs, expr.args[3].value)
            end
        end
    elseif rhs.head==:call && rhs.args[1]==:__hpat_data_source_TXT_size
        num = ParallelAccelerator.CGen.from_expr(rhs.args[2])
        c_lhs = ParallelAccelerator.CGen.from_expr(lhs)
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

function from_assignment_match_dist(lhs::Any, rhs::Any)
    return ""
end

end # module
