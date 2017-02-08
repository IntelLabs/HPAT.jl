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
import ParallelAccelerator.CGen.from_arraysize
import CompilerTools.DebugMsg
using CompilerTools.LambdaHandling
using CompilerTools.Helper
DebugMsg.init()

import HPAT

#using Debug

include("cgen-hpat-pattern-match-table.jl")
include("cgen-hpat-pattern-match-daal.jl")

function pattern_match_call_dist_init(f::GlobalRef,linfo)
    if f.name==:hpat_dist_init
        return ";"#"MPI_Init(0,0);"
    else
        return ""
    end
end

function pattern_match_call_dist_init_gaas(f::GlobalRef,linfo)
    # TODO Make separate functions for each call below as done by HPAT
    # This can break in future
    if f.name==:hpat_dist_init_gaas
        s = ""
        s *= "int32_t __hpat_node_id_local;\n"
        s *= "int32_t __hpat_num_pes_local;\n"
        s *= "MPI_Comm __hpat_local_comm;\n"
        s *= "MPI_Comm __hpat_bridge_comm;\n"
        s *= "MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,__hpat_node_id, MPI_INFO_NULL, &__hpat_local_comm);\n"
        s *= "MPI_Comm_rank(__hpat_local_comm,&__hpat_node_id_local);\n"
        s *= "MPI_Comm_split(MPI_COMM_WORLD, __hpat_node_id_local, __hpat_node_id, &__hpat_bridge_comm);\n"
        s *= "MPI_Comm_size(__hpat_local_comm,&__hpat_num_pes_local);\n"
        return s
    else
        return ""
    end
end

function pattern_match_call_dist_init(f::Any,linfo)
    return ""
end

function pattern_match_call_dist_init_gaas(f::Any,linfo)
    return ""
end

function pattern_match_call_dist_init2d(f::GlobalRef,linfo)
    if f.name==:hpat_dist_2d_init
        return """    MKL_INT tmp_nodeid = (MKL_INT)__hpat_node_id;
                      MKL_INT tmp_numpes = (MKL_INT)__hpat_num_pes;
                      blacs_setup_( &tmp_nodeid, &tmp_numpes);
                      // get default context
                      double  zero = 0.0E+0, one = 1.0E+0, two = 2.0E+0, negone = -1.0E+0;
                      MKL_INT i_zero = 0, i_one = 1, i_four = 4, i_negone = -1;
                      MKL_INT ictxt=-1;
                      blacs_get_( &i_negone, &i_zero, &ictxt );
                      int __hpat_2d_dims[2];
                      __hpat_2d_dims[0] = __hpat_2d_dims[1] = 0;
                      MPI_Dims_create(__hpat_num_pes, 2, __hpat_2d_dims);
                      __hpat_num_pes_x = __hpat_2d_dims[0];
                      __hpat_num_pes_y = __hpat_2d_dims[1];

                      // create row-major 2D grid
                      MKL_INT tmp_pes_x=(MKL_INT)__hpat_num_pes_x, tmp_pes_y=(MKL_INT)__hpat_num_pes_y;
                      MKL_INT tmp_id_x, tmp_id_y; // 32-bit or 64-bit
                      blacs_gridinit_( &ictxt, "R", &tmp_pes_x, &tmp_pes_y );
                      blacs_gridinfo_( &ictxt, &tmp_pes_x, &tmp_pes_y, &tmp_id_x, &tmp_id_y);

                      __hpat_num_pes_x = (int)tmp_pes_x;
                      __hpat_num_pes_y = (int)tmp_pes_y;
                      __hpat_node_id_x = (int)tmp_id_x;
                      __hpat_node_id_y = (int)tmp_id_y;
                      """

    else
        return ""
    end
end

function pattern_match_call_dist_init2d(f::ANY,linfo)
    return ""
end

function pattern_match_call_dist_init_stencil_reqs(f::GlobalRef,linfo)
    s = ""
    if f==GlobalRef(HPAT.API,:__hpat_init_stencil_reqs)
        # TODO: assuming 1D stencil
        s *= "MPI_Request mpi_req_send_left, mpi_req_recv_left, mpi_req_send_right, mpi_req_recv_right;"
    end
    return s
end

pattern_match_call_dist_init_stencil_reqs(f::Any,l) = ""

function pattern_match_call_dist_send_left(f::GlobalRef, in_arr::LHSVar, linfo)
    s = ""
    if f==GlobalRef(HPAT.API,:__hpat_send_left)
        # TODO: assuming 1D stencil
        c_in_arr = ParallelAccelerator.CGen.from_expr(in_arr, linfo)
        typ = eltype(ParallelAccelerator.CGen.getSymType(in_arr, linfo))
        mpi_typ = get_mpi_type_from_var_type(typ)
        s *= "MPI_Isend($c_in_arr.data, 1, $mpi_typ, __hpat_node_id-1, 11, MPI_COMM_WORLD, &mpi_req_send_left);"
    end
    return s
end

pattern_match_call_dist_send_left(f::ANY, in_arr::ANY, linfo) = ""

function pattern_match_call_dist_recv_left(f::GlobalRef, tmp_var::LHSVar, linfo)
    s = ""
    if f==GlobalRef(HPAT.API,:__hpat_recv_left)
        # TODO: assuming 1D stencil
        c_tmp_var = ParallelAccelerator.CGen.from_expr(tmp_var, linfo)
        typ = ParallelAccelerator.CGen.getSymType(tmp_var, linfo)
        mpi_typ = get_mpi_type_from_var_type(typ)
        s *= "MPI_Irecv(&$c_tmp_var, 1, $mpi_typ, __hpat_node_id-1, 22, MPI_COMM_WORLD, &mpi_req_recv_left);"
    end
    return s
end

pattern_match_call_dist_recv_left(f::ANY, tmp_var::ANY, linfo) = ""

function pattern_match_call_dist_send_right(f::GlobalRef, in_arr::LHSVar, linfo)
    s = ""
    if f==GlobalRef(HPAT.API,:__hpat_send_right)
        # TODO: assuming 1D stencil
        c_in_arr = ParallelAccelerator.CGen.from_expr(in_arr, linfo)
        typ = eltype(ParallelAccelerator.CGen.getSymType(in_arr, linfo))
        mpi_typ = get_mpi_type_from_var_type(typ)
        s *= "MPI_Isend(&$c_in_arr.data[$c_in_arr.ARRAYLEN()-1], 1, $mpi_typ, __hpat_node_id+1, 22, MPI_COMM_WORLD, &mpi_req_send_right);"
    end
    return s
end

pattern_match_call_dist_send_right(f::ANY, in_arr::ANY, linfo) = ""

function pattern_match_call_dist_recv_right(f::GlobalRef, tmp_var::LHSVar, linfo)
    s = ""
    if f==GlobalRef(HPAT.API,:__hpat_recv_right)
        # TODO: assuming 1D stencil
        c_tmp_var = ParallelAccelerator.CGen.from_expr(tmp_var, linfo)
        typ = ParallelAccelerator.CGen.getSymType(tmp_var, linfo)
        mpi_typ = get_mpi_type_from_var_type(typ)
        s *= "MPI_Irecv(&$c_tmp_var, 1, $mpi_typ, __hpat_node_id+1, 11, MPI_COMM_WORLD, &mpi_req_recv_right);"
    end
    return s
end

pattern_match_call_dist_recv_right(f::ANY, in_arr::ANY, linfo) = ""

function pattern_match_call_dist_wait_left(f::GlobalRef, linfo)
    s = ""
    if f==GlobalRef(HPAT.API,:__hpat_wait_left)
        # TODO: assuming 1D stencil
        s *= "MPI_Wait(&mpi_req_recv_left, MPI_STATUS_IGNORE);\n"
        s *= "MPI_Wait(&mpi_req_send_left, MPI_STATUS_IGNORE);\n"
    end
    return s
end

pattern_match_call_dist_wait_left(f::ANY, linfo) = ""

function pattern_match_call_dist_wait_right(f::GlobalRef, linfo)
    s = ""
    if f==GlobalRef(HPAT.API,:__hpat_wait_right)
        # TODO: assuming 1D stencil
        s *= "MPI_Wait(&mpi_req_recv_right, MPI_STATUS_IGNORE);\n"
        s *= "MPI_Wait(&mpi_req_send_right, MPI_STATUS_IGNORE);\n"
    end
    return s
end

pattern_match_call_dist_wait_right(f::ANY, linfo) = ""

function pattern_match_call_dist_add_extra_block(f::GlobalRef, local_blocks::LHSVar,
      total_blocks::LHSVar, node_id::LHSVar, num_pes::LHSVar, linfo)
  s = ""
  if f==GlobalRef(HPAT.API,:__hpat_add_extra_block)
    # similar to numroc.f
    c_local_blocks = ParallelAccelerator.CGen.from_expr(local_blocks, linfo)
    c_total_blocks = ParallelAccelerator.CGen.from_expr(total_blocks, linfo)
    c_node_id = ParallelAccelerator.CGen.from_expr(node_id, linfo)
    c_num_pes = ParallelAccelerator.CGen.from_expr(num_pes, linfo)
    s *= "$c_local_blocks += ($c_node_id<($c_total_blocks%$c_num_pes)?1:0);\n"
  end
  return s
end

function pattern_match_call_dist_add_extra_block(f::ANY, local_blocks::ANY,
      total_blocks::ANY, node_id::ANY, num_pes::ANY, linfo)
  return ""
end

function pattern_match_call_dist_get_leftovers(f::GlobalRef,
      total_blocks::LHSVar, node_id::LHSVar, num_pes::LHSVar, total_data_size::LHSVar, block_size::LHSVar, linfo)
  s = ""
  if f==GlobalRef(HPAT.API,:__hpat_get_leftovers)
    c_total_data_size = ParallelAccelerator.CGen.from_expr(total_data_size, linfo)
    # similar to numroc.f
    c_total_blocks = ParallelAccelerator.CGen.from_expr(total_blocks, linfo)
    c_block_size = ParallelAccelerator.CGen.from_expr(block_size, linfo)
    c_node_id = ParallelAccelerator.CGen.from_expr(node_id, linfo)
    c_num_pes = ParallelAccelerator.CGen.from_expr(num_pes, linfo)
    s *= "(($c_node_id==($c_total_blocks%$c_num_pes))? ($c_total_data_size%$c_block_size):0)"
  end
  return s
end

function pattern_match_call_dist_get_leftovers(f::ANY,
      total_blocks::ANY, node_id::ANY, num_pes::ANY,total_data_size::ANY,block_size::ANY, linfo)
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

function pattern_match_reduce_maximum(reductionFunc::DelayedFunc,linfo)
    @dprintln(3, "pattern_match_reduce_maximum ", reductionFunc)
    args = reductionFunc.args[1]
    if length(args)==3 && args[1].args[2].args[1].name==:slt_int &&
        args[2].args[2].args[1].name==:select_value
        return true
    elseif args[1].args[2].args[1]==GlobalRef(Base,:max)
        return true
    end
    return false
end

function pattern_match_reduce_maximum(reductionFunc::GlobalRef,linfo)
    @dprintln(3, "pattern_match_reduce_maximum ", reductionFunc)
    if reductionFunc.name==:max
        return true
    end
    return false
end

function pattern_match_reduce_sum(reductionFunc::DelayedFunc,linfo)
    reduce_box = reductionFunc.args[1][1].args[2]
    if reduce_box.args[1]==GlobalRef(Core.Intrinsics,:box)
    #@assert reduce_box.args[1]==GlobalRef(Core.Intrinsics,:box) "invalid reduction function"
        if reduce_box.args[3].args[1].name==:add_float || reduce_box.args[3].args[1].name==:add_int
            return true
        end
    elseif reduce_box.args[1]==GlobalRef(Base,:+)
        return true
    end
    return false
end

function pattern_match_reduce_sum(reductionFunc::GlobalRef,linfo)
    if reductionFunc.name==:add_float || reductionFunc.name==:add_int || reductionFunc.name==:(+)
        return true
    end
    return false
end

function pattern_match_call_dist_reduce(f::GlobalRef, var::TypedVar, reductionFunc::DelayedFunc, output::LHSVar,linfo)
    if f.name==:hpat_dist_reduce
        mpi_type = get_mpi_type_from_var_type(var.typ)
        mpi_func = ""
        if pattern_match_reduce_sum(reductionFunc, linfo)
            mpi_func = "MPI_SUM"
        else
            throw("CGen unsupported MPI reduction function")
        end
        var = toLHSVar(var)
        c_var = ParallelAccelerator.CGen.from_expr(var, linfo)
        c_output = ParallelAccelerator.CGen.from_expr(output, linfo)
        s="MPI_Reduce(&$(c_var), &$c_output, 1, $mpi_type, $mpi_func, 0, MPI_COMM_WORLD);"
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
        c_node_id = ParallelAccelerator.CGen.from_expr(node_id, linfo)
        c_num_pes = ParallelAccelerator.CGen.from_expr(num_pes, linfo)
        s = "(($c_node_id==$c_num_pes-1) ? $c_total-$c_node_id*$c_div : $c_div)"
    end
    return s
end

function pattern_match_call_dist_portion(f::ANY, total::ANY, div::ANY, num_pes::ANY, node_id::ANY,linfo)
    return ""
end

function pattern_match_call_dist_node_end(f::GlobalRef, total::Union{RHSVar,Int}, div::RHSVar, num_pes::LHSVar, node_id::LHSVar,linfo)
    s = ""
    if f.name==:__hpat_get_node_end
        c_total = ParallelAccelerator.CGen.from_expr(total, linfo)
        c_div = ParallelAccelerator.CGen.from_expr(div, linfo)
        c_node_id = ParallelAccelerator.CGen.from_expr(node_id, linfo)
        c_num_pes = ParallelAccelerator.CGen.from_expr(num_pes, linfo)
        s = "(($c_node_id==$c_num_pes-1) ? $c_total : ($c_node_id+1)*$c_div)"
    end
    return s
end

function pattern_match_call_dist_node_end(f::ANY, total::ANY, div::ANY, num_pes::ANY, node_id::ANY,linfo)
    return ""
end

function pattern_match_call_dist_allreduce(f::GlobalRef, var::RHSVar, reductionFunc, output::RHSVar, size::Union{Expr,RHSVar,Int},linfo)
    if f.name==:hpat_dist_allreduce
        mpi_type = ""
        var = toLHSVar(var)
        c_var = ParallelAccelerator.CGen.from_expr(var, linfo)
        c_size = ParallelAccelerator.CGen.from_expr(size, linfo)
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
        mpi_type = get_mpi_type_from_var_type(var_typ)

        mpi_func = ""
        if pattern_match_reduce_sum(reductionFunc, linfo)
            mpi_func = "MPI_SUM"
        elseif pattern_match_reduce_maximum(reductionFunc, linfo)
            mpi_func = "MPI_MAX"
        else
            throw("CGen unsupported MPI reduction function $reductionFunc")
        end

        s="MPI_Allreduce($c_var, $c_output, $c_size, $mpi_type, $mpi_func, MPI_COMM_WORLD);"
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
    if f.name==:__hpat_dist_broadcast
        mpi_type = ""
        var = toLHSVar(var)
        c_var = ParallelAccelerator.CGen.from_expr(var, linfo)
        var_typ = ParallelAccelerator.CGen.getSymType(var, linfo)
        c_size = ParallelAccelerator.CGen.from_expr(size, linfo)
        is_array =  var_typ<:Array
        if is_array
            var_typ = eltype(var_typ)
            c_var *= ".data"
        else
            c_var = "&"*c_var
        end
        mpi_type = get_mpi_type_from_var_type(var_typ)
        s="MPI_Bcast($c_var, $c_size, $mpi_type, 0, MPI_COMM_WORLD);"
        return s
    else
        return ""
    end
end


function pattern_match_call_dist_bcast(f::Any, v::Any, rf::Any,linfo)
    return ""
end

function pattern_match_call_dist_cumsum(f::GlobalRef, out_arr::LHSVar, in_arr::LHSVar, linfo)
    @dprintln(3, "pattern_match_call_dist_cumsum f = ", f)
    s = ""
    if f==GlobalRef(HPAT.API,:dist_cumsum!)
        c_out_arr = ParallelAccelerator.CGen.from_expr(out_arr, linfo)
        c_in_arr = ParallelAccelerator.CGen.from_expr(in_arr, linfo)
        s *= "$c_out_arr;\n"

        typ = eltype(ParallelAccelerator.CGen.getSymType(in_arr, linfo))
        ctyp = ParallelAccelerator.CGen.toCtype(typ)
        mpi_typ = get_mpi_type_from_var_type(typ)

        sum_var = "tmp_sum_$c_in_arr"
        prefix_var = "prefix_$c_in_arr"
        size_var = "size_$c_in_arr"
        s *= "int64_t $size_var = " * ParallelAccelerator.CGen.from_arraysize(in_arr,1,linfo) *";\n"
        s *= "$ctyp $sum_var=0, $prefix_var=0;\n"
        s *= "for(int i=0; i<$size_var; i++)\n"
        s *= "  $sum_var += $c_in_arr.data[i];\n"
        s *= "MPI_Exscan(&$sum_var, &$prefix_var, 1, $mpi_typ, MPI_SUM, MPI_COMM_WORLD);\n"
        s *= "for(int i=0; i<$size_var; i++) {\n"
        s *= "  $prefix_var += $c_in_arr.data[i];\n"
        s *= "  $c_out_arr.data[i] = $prefix_var;\n"
        s *= "}\n"
    end
    return s
end


function pattern_match_call_dist_cumsum(f::Any, v::Any, rf::Any,linfo)
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
      s *= "file_id_$num = H5Fopen((const char*)"*ParallelAccelerator.CGen.from_expr(file_name, linfo)*".data.data, H5F_ACC_RDWR, plist_id_$num);\n"
      s *= "assert(file_id_$num != -1);\n"
      s *= "ret_$num = H5Pclose(plist_id_$num);\n"
      s *= "assert(ret_$num != -1);\n"
      #s *= "hid_t dataset_id_$num;\n"
      #s *= "dataset_id_$num = H5Dcreate(file_id_$num, "*ParallelAccelerator.CGen.from_expr(data_var, linfo)*", H5P_DEFAULT);\n"
      #s *= "assert(dataset_id_$num != -1);\n"
    elseif f.name==:__hpat_data_sink_HDF5_create
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
    elseif f.name==:__hpat_data_sink_TXT_close
        num = string(id)
        s *= "MPI_File_close(&dsink_txt_file_$num);\n"
    end
    return s
end

function pattern_match_call_data_src_close(f::Any, v::Any,linfo)
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
        num = string(id)
        file_name_str = ParallelAccelerator.CGen.from_expr(file_name, linfo)
        s = """
            MPI_File dsrc_txt_file_$num;
            int ierr_$num = MPI_File_open(MPI_COMM_WORLD, (const char*)$file_name_str.data.data, MPI_MODE_RDONLY, MPI_INFO_NULL, &dsrc_txt_file_$num);
            assert(ierr_$num==0);
            """
    elseif f.name==:__hpat_data_sink_TXT_open
        num = string(id)
        file_name_str = ParallelAccelerator.CGen.from_expr(file_name, linfo)
        s = """
            MPI_File dsink_txt_file_$num;
            int ierr_$num = MPI_File_open(MPI_COMM_WORLD, (const char*)$file_name_str.data.data, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &dsink_txt_file_$num);
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
            println("h5 data type ", data_typ)
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

        c_start = ParallelAccelerator.CGen.from_expr(start, linfo)
        c_count = ParallelAccelerator.CGen.from_expr(count, linfo)

        # assuming 1st dimension is partitined
        s =  "hsize_t CGen_HDF5_start_$num[data_ndim_$num];\n"
        s *= "hsize_t CGen_HDF5_count_$num[data_ndim_$num];\n"
        s *= "CGen_HDF5_start_$num[0] = $c_start;\n"
        s *= "CGen_HDF5_count_$num[0] = $c_count;\n"
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
        s *= "H5Pset_dxpl_mpio(xfer_plist_$num, H5FD_MPIO_COLLECTIVE);\n"
        s *= "ret_$num = H5Dread(dataset_id_$num, $h5_typ, mem_dataspace_$num, space_id_$num, xfer_plist_$num, $carr.getData());\n"
        s *= "assert(ret_$num != -1);\n"
        #s*="if(__hpat_node_id==__hpat_num_pes/2) printf(\"h5 read %lf\\n\", MPI_Wtime()-h5_read_start_$num);\n"
        s *= ";\n"
    elseif f.name==:__hpat_data_source_TXT_read
        # assuming 1st dimension is partitined
        data_typ = eltype(ParallelAccelerator.CGen.getSymType(arr, linfo))
        t_typ = ParallelAccelerator.CGen.toCtype(data_typ)
        carr = ParallelAccelerator.CGen.from_expr(toLHSVar(arr), linfo)
        c_start = ParallelAccelerator.CGen.from_expr(start, linfo)
        c_count = ParallelAccelerator.CGen.from_expr(count, linfo)
        c_arr = ParallelAccelerator.CGen.from_expr(toLHSVar(arr), linfo)

        s = """
            int64_t CGen_txt_start_$num = $c_start;
            int64_t CGen_txt_count_$num = $c_count;
            int64_t CGen_txt_end_$num = $c_start+$c_count;


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
            $t_typ * CGen_txt_data_arr = ($t_typ *)$c_arr.getData();
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
         //           std::cout<<$c_arr[CGen_txt_data_ind_$num-1]<<std::endl;
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
              local_size_x::LHSVar, local_size_y::LHSVar,
              leftover_x::LHSVar, leftover_y::LHSVar, linfo)
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

        c_start_x = ParallelAccelerator.CGen.from_expr(start_x, linfo)
        c_start_y = ParallelAccelerator.CGen.from_expr(start_y, linfo)
        c_stride_x = ParallelAccelerator.CGen.from_expr(stride_x, linfo)
        c_stride_y = ParallelAccelerator.CGen.from_expr(stride_y, linfo)
        c_count_x = ParallelAccelerator.CGen.from_expr(count_x, linfo)
        c_count_y = ParallelAccelerator.CGen.from_expr(count_y, linfo)
        c_block_x = ParallelAccelerator.CGen.from_expr(block_x, linfo)
        c_block_y = ParallelAccelerator.CGen.from_expr(block_y, linfo)
        c_local_size_x = ParallelAccelerator.CGen.from_expr(local_size_x, linfo)
        c_local_size_y = ParallelAccelerator.CGen.from_expr(local_size_y, linfo)
        c_leftover_x = ParallelAccelerator.CGen.from_expr(leftover_x, linfo)
        c_leftover_y = ParallelAccelerator.CGen.from_expr(leftover_y, linfo)

        # assuming 1st and 2nd dimensions are partitined
        # hyperslab call input variables
        s *=  "hsize_t CGen_HDF5_start_$num[data_ndim_$num];\n"
        s *=  "hsize_t CGen_HDF5_stride_$num[data_ndim_$num];\n"
        s *= "hsize_t CGen_HDF5_count_$num[data_ndim_$num];\n"
        s *=  "hsize_t CGen_HDF5_block_$num[data_ndim_$num];\n"
        # last 2 dimensions are set using distributed-pass data
        s *= "CGen_HDF5_start_$num[0] = $c_start_y;\n"
        s *= "CGen_HDF5_start_$num[1] = $c_start_x;\n"
        s *= "CGen_HDF5_stride_$num[0] = $c_stride_y;\n"
        s *= "CGen_HDF5_stride_$num[1] = $c_stride_x;\n"
        s *= "CGen_HDF5_count_$num[0] = $c_count_y;\n"
        s *= "CGen_HDF5_count_$num[1] = $c_count_x;\n"
        s *= "CGen_HDF5_block_$num[0] = $c_block_y;\n"
        s *= "CGen_HDF5_block_$num[1] = $c_block_x;\n"

        # rest of dimensions, if any, are not divided
        s *= "for(int i_CGen_dim=2; i_CGen_dim<data_ndim_$num; i_CGen_dim++) {\n"
        s *= "  CGen_HDF5_start_$num[i_CGen_dim] = 0;\n"
        s *= "  CGen_HDF5_stride_$num[i_CGen_dim] = 1;\n"
        s *= "  CGen_HDF5_count_$num[i_CGen_dim] = space_dims_$num[i_CGen_dim];\n"
        s *= "  CGen_HDF5_block_$num[i_CGen_dim] = 1;\n"
        s *= "}\n"
        #s *= "std::cout<<\"read size \"<<CGen_HDF5_start_$num[0]<<\" \"<<CGen_HDF5_count_$num[0]<<\" \"<<CGen_HDF5_start_$num[1]<<\" \"<<CGen_HDF5_count_$num[1]<<std::endl;\n"
        s *= """ret_$num = H5Sselect_hyperslab(space_id_$num, H5S_SELECT_SET, CGen_HDF5_start_$num,
                         CGen_HDF5_stride_$num, CGen_HDF5_count_$num, CGen_HDF5_block_$num);\n"""
        s *= "assert(ret_$num != -1);\n"

        # select leftover y data
        # y dimension is lefover columns, x dimension is as before
        s *= "CGen_HDF5_start_$num[0] = $c_start_y+$c_stride_y*$c_count_y;\n"
        s *= "CGen_HDF5_start_$num[1] = $c_start_x;\n"
        s *= "CGen_HDF5_stride_$num[0] = 1;\n"
        s *= "CGen_HDF5_stride_$num[1] = $c_stride_x;\n"
        s *= "CGen_HDF5_count_$num[0] = $c_leftover_y;\n"
        s *= "CGen_HDF5_count_$num[1] = $c_count_x;\n"
        s *= "CGen_HDF5_block_$num[0] = 1;\n"
        s *= "CGen_HDF5_block_$num[1] = $c_block_x;\n"

        s *= """ret_$num = H5Sselect_hyperslab(space_id_$num, H5S_SELECT_OR, CGen_HDF5_start_$num,
                         CGen_HDF5_stride_$num, CGen_HDF5_count_$num, CGen_HDF5_block_$num);\n"""
        s *= "assert(ret_$num != -1);\n"

        # select leftover x data
        # x dimension is lefover columns, y dimension is as before
        s *= "CGen_HDF5_start_$num[0] = $c_start_y;\n"
        s *= "CGen_HDF5_start_$num[1] = $c_start_x+$c_stride_x*$c_count_x;\n"
        s *= "CGen_HDF5_stride_$num[0] = $c_stride_y;\n"
        s *= "CGen_HDF5_stride_$num[1] = 1;\n"
        s *= "CGen_HDF5_count_$num[0] = $c_count_y;\n"
        s *= "CGen_HDF5_count_$num[1] = $c_leftover_x;\n"
        s *= "CGen_HDF5_block_$num[0] = $c_block_y;\n"
        s *= "CGen_HDF5_block_$num[1] = 1;\n"
        s *= """ret_$num = H5Sselect_hyperslab(space_id_$num, H5S_SELECT_OR, CGen_HDF5_start_$num,
                         CGen_HDF5_stride_$num, CGen_HDF5_count_$num, CGen_HDF5_block_$num);\n"""
        s *= "assert(ret_$num != -1);\n"

        # intersection of x-y leftovers
        s *= "CGen_HDF5_start_$num[0] = $c_start_y+$c_stride_y*$c_count_y;\n"
        s *= "CGen_HDF5_start_$num[1] = $c_start_x+$c_stride_x*$c_count_x;\n"
        s *= "CGen_HDF5_stride_$num[0] = 1;\n"
        s *= "CGen_HDF5_stride_$num[1] = 1;\n"
        s *= "CGen_HDF5_count_$num[0] = $c_leftover_y;\n"
        s *= "CGen_HDF5_count_$num[1] = $c_leftover_x;\n"
        s *= "CGen_HDF5_block_$num[0] = 1;\n"
        s *= "CGen_HDF5_block_$num[1] = 1;\n"

        s *= """ret_$num = H5Sselect_hyperslab(space_id_$num, H5S_SELECT_OR, CGen_HDF5_start_$num,
                         CGen_HDF5_stride_$num, CGen_HDF5_count_$num, CGen_HDF5_block_$num);\n"""
        s *= "assert(ret_$num != -1);\n"

        # size of memory to read to
        s *=  "hsize_t CGen_HDF5_memsize_$num[data_ndim_$num];\n"
        s *= "CGen_HDF5_memsize_$num[0] = $c_local_size_y;\n"
        s *= "CGen_HDF5_memsize_$num[1] = $c_local_size_x;\n"

        s *= "hid_t mem_dataspace_$num = H5Screate_simple (data_ndim_$num, CGen_HDF5_memsize_$num, NULL);\n"
        s *= "assert (mem_dataspace_$num != -1);\n"
        s *= "hid_t xfer_plist_$num = H5Pcreate (H5P_DATASET_XFER);\n"
        s *= "assert(xfer_plist_$num != -1);\n"
        s *= "double h5_read_start_$num = MPI_Wtime();\n"
        s *= "H5Pset_dxpl_mpio(xfer_plist_$num, H5FD_MPIO_COLLECTIVE);\n"
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
              local_size_x::ANY, local_size_y::ANY,leftover_x::ANY, leftover_y::ANY, linfo)
  return ""
end

function pattern_match_call_data_sink_write(f::GlobalRef, id::Int, hdf5_var, arr::RHSVar, start::LHSVar, count::LHSVar,tot_size,linfo)
    s = ""
    num::AbstractString = string(id)

    if f.name==:__hpat_data_sink_HDF5_write
        arr_typ = ParallelAccelerator.CGen.getSymType(arr, linfo)
        num_dims = ndims(arr_typ)
        @assert num_dims==length(tot_size) "sink total size dimension error"
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

        c_start = ParallelAccelerator.CGen.from_expr(start, linfo)
        c_count = ParallelAccelerator.CGen.from_expr(count, linfo)

        # create dataset
        s *= " hid_t dataset_id_$num;\n"
        s *= " hid_t  filespace_$num, memspace_$num;\n"
        s *= " hsize_t  dataset_dims_$num[$num_dims];\n"
        #s *= " for(int i=0; i<$num_dims; i++) dataset_dims_$num[i]=$(ParallelAccelerator.CGen.from_expr(tot_size[i],linfo));\n"
        for i in 1:num_dims
            s*= "dataset_dims_$num[$(num_dims-i)]=$(ParallelAccelerator.CGen.from_expr(tot_size[i],linfo));\n"
        end
        s *= "  filespace_$num = H5Screate_simple($num_dims, dataset_dims_$num, NULL);\n"
        s *= "  dataset_id_$num = H5Dcreate(file_id_$num, \"$hdf5_var\", $h5_typ, filespace_$num,\n"
        s *=  "     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);\n"
        s *= " H5Sclose(filespace_$num);\n"
        # assuming 1st dimension is partitined
        s *=  "hsize_t CGen_HDF5_start_$num[$num_dims];\n"
        s *= "hsize_t CGen_HDF5_count_$num[$num_dims];\n"
        s *= "CGen_HDF5_start_$num[0] = $c_start;\n"
        s *= "CGen_HDF5_count_$num[0] = $c_count;\n"
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

function pattern_match_call_data_sink_txt_write(f::GlobalRef, id::Int, arr::RHSVar, start::LHSVar, count::LHSVar,tot_size,linfo)
    s = ""
    num::AbstractString = string(id)

    if f.name==:__hpat_data_sink_TXT_write
        arr_typ = ParallelAccelerator.CGen.getSymType(arr, linfo)
        num_dims = ndims(arr_typ)
        @assert num_dims==length(tot_size) "sink total size dimension error"
        data_typ = eltype(arr_typ)
        c_arr = ParallelAccelerator.CGen.from_expr(toLHSVar(arr), linfo)
        c_start = ParallelAccelerator.CGen.from_expr(start, linfo)
        c_count = ParallelAccelerator.CGen.from_expr(count, linfo)
        slice_count = string(1)
        if num_dims>1
            slice_count = mapfoldl(i->string(tot_size[i]), (a,b)->a*"*"*b, 1:num_dims-1)
        end
        s *= """
        // convert to string
        std::stringstream CGen_txt_ss_$num;
        for(uint64_t i=0; i<$c_count; i++) {
            for(uint64_t j=0; j <$slice_count; j++) {
                CGen_txt_ss_$num<<$c_arr.data[i*$slice_count+j];
                if(j==$slice_count-1)
                    CGen_txt_ss_$num<<"\\n";
                else
                    CGen_txt_ss_$num<<",";
            }
        }
        CGen_txt_ss_$num.flush();
        std::string CGen_txt_str_$num =CGen_txt_ss_$num.str();

        MPI_Offset CGen_txt_buff_size_$num = CGen_txt_str_$num.length();
        MPI_Offset CGen_txt_offset_$num = 0;
        MPI_Exscan(&CGen_txt_buff_size_$num, &CGen_txt_offset_$num, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_File_write_at_all(dsink_txt_file_$num, CGen_txt_offset_$num, CGen_txt_str_$num.c_str(),
                          CGen_txt_buff_size_$num, MPI_CHAR, MPI_STATUS_IGNORE);
        """
    end
    return s
end

function pattern_match_call_data_sink_txt_write(f::ANY, id::ANY, arr::ANY, tot_size::ANY, start::ANY, count::ANY,linfo)
    return ""
end


function pattern_match_call_data_sink_write_2d(f::GlobalRef, id::Int, hdf5_var, arr::RHSVar,
            start_x::LHSVar, start_y::LHSVar, stride_x::LHSVar, stride_y::LHSVar,
            count_x::LHSVar, count_y::LHSVar, block_x::LHSVar, block_y::LHSVar,
            local_size_x::LHSVar, local_size_y::LHSVar, leftover_x::LHSVar, leftover_y::LHSVar,tot_size,linfo)
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
            println("h5 data type ", data_typ)
            throw("CGen unsupported HDF5 data type")
        end

        c_start_x = ParallelAccelerator.CGen.from_expr(start_x, linfo)
        c_start_y = ParallelAccelerator.CGen.from_expr(start_y, linfo)
        c_stride_x = ParallelAccelerator.CGen.from_expr(stride_x, linfo)
        c_stride_y = ParallelAccelerator.CGen.from_expr(stride_y, linfo)
        c_count_x = ParallelAccelerator.CGen.from_expr(count_x, linfo)
        c_count_y = ParallelAccelerator.CGen.from_expr(count_y, linfo)
        c_block_x = ParallelAccelerator.CGen.from_expr(block_x, linfo)
        c_block_y = ParallelAccelerator.CGen.from_expr(block_y, linfo)
        c_local_size_x = ParallelAccelerator.CGen.from_expr(local_size_x, linfo)
        c_local_size_y = ParallelAccelerator.CGen.from_expr(local_size_y, linfo)
        c_leftover_x = ParallelAccelerator.CGen.from_expr(leftover_x, linfo)
        c_leftover_y = ParallelAccelerator.CGen.from_expr(leftover_y, linfo)

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

        # assuming 1st and 2nd dimensions are partitined
        # hyperslab call input variables
        s *=  "hsize_t CGen_HDF5_start_$num[$num_dims];\n"
        s *=  "hsize_t CGen_HDF5_stride_$num[$num_dims];\n"
        s *= "hsize_t CGen_HDF5_count_$num[$num_dims];\n"
        s *=  "hsize_t CGen_HDF5_block_$num[$num_dims];\n"
        # last 2 dimensions are set using distributed-pass data
        s *= "CGen_HDF5_start_$num[0] = $c_start_y;\n"
        s *= "CGen_HDF5_start_$num[1] = $c_start_x;\n"
        s *= "CGen_HDF5_stride_$num[0] = $c_stride_y;\n"
        s *= "CGen_HDF5_stride_$num[1] = $c_stride_x;\n"
        s *= "CGen_HDF5_count_$num[0] = $c_count_y;\n"
        s *= "CGen_HDF5_count_$num[1] = $c_count_x;\n"
        s *= "CGen_HDF5_block_$num[0] = $c_block_y;\n"
        s *= "CGen_HDF5_block_$num[1] = $c_block_x;\n"

        #s *= "for(int i_CGen_dim=1; i_CGen_dim<$num_dims; i_CGen_dim++) {\n"
        for i in 2:length(tot_size)-1
            s *= "  CGen_HDF5_start_$num[$i] = 0;\n"
            s *= "  CGen_HDF5_stride_$num[$i] = 1;\n"
            s *= "  CGen_HDF5_count_$num[$i] = $(ParallelAccelerator.CGen.from_expr(tot_size[i],linfo));\n"
            s *= "  CGen_HDF5_block_$num[$i] = 1;\n"
        end
        #s *= "}\n"
        #s *= "std::cout<<\"read size \"<<CGen_HDF5_start_$num[0]<<\" \"<<CGen_HDF5_count_$num[0]<<\" \"<<CGen_HDF5_start_$num[1]<<\" \"<<CGen_HDF5_count_$num[1]<<std::endl;\n"
        s *= "filespace_$num = H5Dget_space(dataset_id_$num);\n"
        s *= """ret_$num = H5Sselect_hyperslab(filespace_$num, H5S_SELECT_SET,
                 CGen_HDF5_start_$num, CGen_HDF5_stride_$num, CGen_HDF5_count_$num, CGen_HDF5_block_$num);\n"""
        s *= "assert(ret_$num != -1);\n"

        # y leftovers, x dimension as before
        s *= "CGen_HDF5_start_$num[0] = $c_start_y+$c_stride_y*$c_count_y;\n"
        s *= "CGen_HDF5_start_$num[1] = $c_start_x;\n"
        s *= "CGen_HDF5_stride_$num[0] = 1;\n"
        s *= "CGen_HDF5_stride_$num[1] = $c_stride_x;\n"
        s *= "CGen_HDF5_count_$num[0] = $c_leftover_y;\n"
        s *= "CGen_HDF5_count_$num[1] = $c_count_x;\n"
        s *= "CGen_HDF5_block_$num[0] = 1;\n"
        s *= "CGen_HDF5_block_$num[1] = $c_block_x;\n"

        s *= """ret_$num = H5Sselect_hyperslab(filespace_$num, H5S_SELECT_OR,
                 CGen_HDF5_start_$num, CGen_HDF5_stride_$num, CGen_HDF5_count_$num, CGen_HDF5_block_$num);\n"""
        s *= "assert(ret_$num != -1);\n"

        # x leftovers, y dimension as before
        s *= "CGen_HDF5_start_$num[0] = $c_start_y;\n"
        s *= "CGen_HDF5_start_$num[1] = $c_start_x+$c_stride_x*$c_count_x;\n"
        s *= "CGen_HDF5_stride_$num[0] = $c_stride_y;\n"
        s *= "CGen_HDF5_stride_$num[1] = 1;\n"
        s *= "CGen_HDF5_count_$num[0] = $c_count_y;\n"
        s *= "CGen_HDF5_count_$num[1] = $c_leftover_x;\n"
        s *= "CGen_HDF5_block_$num[0] = $c_block_y;\n"
        s *= "CGen_HDF5_block_$num[1] = 1;\n"

        s *= """ret_$num = H5Sselect_hyperslab(filespace_$num, H5S_SELECT_OR,
                 CGen_HDF5_start_$num, CGen_HDF5_stride_$num, CGen_HDF5_count_$num, CGen_HDF5_block_$num);\n"""
        s *= "assert(ret_$num != -1);\n"

        # both x-y leftovers
        s *= "CGen_HDF5_start_$num[0] = $c_start_y+$c_stride_y*$c_count_y;\n"
        s *= "CGen_HDF5_start_$num[1] = $c_start_x+$c_stride_x*$c_count_x;\n"
        s *= "CGen_HDF5_stride_$num[0] = 1;\n"
        s *= "CGen_HDF5_stride_$num[1] = 1;\n"
        s *= "CGen_HDF5_count_$num[0] = $c_leftover_y;\n"
        s *= "CGen_HDF5_count_$num[1] = $c_leftover_x;\n"
        s *= "CGen_HDF5_block_$num[0] = 1;\n"
        s *= "CGen_HDF5_block_$num[1] = 1;\n"

        s *= """ret_$num = H5Sselect_hyperslab(filespace_$num, H5S_SELECT_OR,
                 CGen_HDF5_start_$num, CGen_HDF5_stride_$num, CGen_HDF5_count_$num, CGen_HDF5_block_$num);\n"""
        s *= "assert(ret_$num != -1);\n"

        # size of memory to read to
        s *=  "hsize_t CGen_HDF5_memsize_$num[$num_dims];\n"
        s *= "CGen_HDF5_memsize_$num[0] = $c_local_size_y;\n"
        s *= "CGen_HDF5_memsize_$num[1] = $c_local_size_x;\n"

        s *= "hid_t mem_dataspace_$num = H5Screate_simple ($num_dims, CGen_HDF5_memsize_$num, NULL);\n"
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

function pattern_match_call_data_sink_write_2d(f::ANY, id::ANY, hdf5_var::ANY, arr::ANY,
            start_x::ANY, start_y::ANY, stride_x::ANY, stride_y::ANY,
            count_x::ANY, count_y::ANY, block_x::ANY, block_y::ANY,
            local_size_x::ANY, local_size_y::ANY,leftover_x::ANY, leftover_y::ANY,tot_size,linfo)
    return ""
end

function pattern_match_call_tic_toc(func::GlobalRef, linfo)
    if func==GlobalRef(Base,:tic)
        return "0; double __hpat_t1 = MPI_Wtime()"
    elseif func==GlobalRef(Base,:toc)
        return "0; if(__hpat_node_id==0) printf(\"exec time %lf\\n\", MPI_Wtime()-__hpat_t1);"
    end
    return ""
end

pattern_match_call_tic_toc(func::ANY, linfo) = ""

function pattern_match_call(ast::Array{Any, 1}, linfo)

  @dprintln(3,"hpat pattern matching ",ast)
  s = ""
  if length(ast)==1
    @dprintln(3,"ast1_typ = ", typeof(ast[1]))
    s *= pattern_match_call_dist_init(ast[1], linfo)
    s *= pattern_match_call_dist_init_gaas(ast[1], linfo)
    s *= pattern_match_call_dist_init2d(ast[1], linfo)
    s *= pattern_match_call_dist_init_stencil_reqs(ast[1], linfo)
    s *= pattern_match_call_dist_wait_left(ast[1], linfo)
    s *= pattern_match_call_dist_wait_right(ast[1], linfo)
    s *= pattern_match_call_get_sec_since_epoch(ast[1], linfo)
    s *= pattern_match_call_tic_toc(ast[1], linfo)
  elseif length(ast)==2
    @dprintln(3,"ast1_typ = ", typeof(ast[1]), " ast2_typ = ", typeof(ast[2]))
    s *= pattern_match_call_data_src_close(ast[1], ast[2], linfo)
    s *= pattern_match_call_get_checkpoint_time(ast[1], ast[2], linfo)
    s *= pattern_match_call_start_checkpoint(ast[1], ast[2], linfo)
    s *= pattern_match_call_end_checkpoint(ast[1], ast[2], linfo)
    s *= pattern_match_call_finish_checkpoint(ast[1], ast[2], linfo)
    s *= pattern_match_call_restore_checkpoint_end(ast[1], ast[2], linfo)
    s *= pattern_match_call_dist_send_left(ast[1], ast[2], linfo)
    s *= pattern_match_call_dist_recv_left(ast[1], ast[2], linfo)
    s *= pattern_match_call_dist_send_right(ast[1], ast[2], linfo)
    s *= pattern_match_call_dist_recv_right(ast[1], ast[2], linfo)
  elseif length(ast)==3
    @dprintln(3,"ast1_typ = ", typeof(ast[1]), " ast2_typ = ", typeof(ast[2]), " ast3_typ = ", typeof(ast[3]))
    s *= pattern_match_call_dist_h5_size(ast[1],ast[2],ast[3], linfo)
    s *= pattern_match_call_dist_bcast(ast[1],ast[2],ast[3], linfo)
    s *= pattern_match_call_dist_cumsum(ast[1],ast[2],ast[3], linfo)
    s *= pattern_match_call_value_checkpoint(ast[1], ast[2], ast[3], linfo)
    s *= pattern_match_call_restore_checkpoint_start(ast[1], ast[2], linfo)
    s *= pattern_match_call_restore_checkpoint_value(ast[1], ast[2], ast[3], linfo)
    s *= pattern_match_call_data_src_read_seq(ast[1],ast[2],ast[3], linfo)
    s *= pattern_match_call_rebalance(ast[1],ast[2],ast[3], linfo)
    s *= pattern_match_call_h5size(ast[1],ast[2],ast[3], linfo)
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
    s *= pattern_match_call_dist_add_extra_block(ast[1],ast[2],ast[3], ast[4], ast[5], linfo)
  elseif length(ast)==6
    s *= pattern_match_call_dist_get_leftovers(ast[1],ast[2],ast[3], ast[4], ast[5],ast[6], linfo)
    s *= pattern_match_call_data_sink_txt_write(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6], linfo)
  elseif length(ast)==24
    s *= pattern_match_call_gemm_2d(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],
          ast[7],ast[8],ast[9],ast[10],ast[11],ast[12],ast[13],ast[14],ast[15],
          ast[16],ast[17],ast[18],ast[19],ast[20],ast[21],ast[22],ast[23],ast[24],linfo)
  elseif length(ast)==7
    s *= pattern_match_call_data_sink_write(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7], linfo)
  elseif length(ast)==9
    s *= pattern_match_call_kmeans(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9], linfo)
  elseif length(ast)==12
    s *= pattern_match_call_linear_regression(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12], linfo)
  elseif length(ast)==13
    s *= pattern_match_call_naive_bayes(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12],ast[13], linfo)
  elseif length(ast)==15
    s *= pattern_match_call_data_src_read_2d(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12],ast[13],ast[14],ast[15], linfo)
  elseif length(ast)==17
    s *= pattern_match_call_data_sink_write_2d(ast[1],ast[2],ast[3],ast[4],ast[5],ast[6],ast[7],ast[8],ast[9],ast[10],ast[11],ast[12],ast[13],ast[14],ast[15],ast[16],ast[17],linfo)
  end
  if length(ast)>=5
    s *= pattern_match_call_filter(linfo, ast[1], ast[2], ast[3], ast[4], ast[5:end])
    s *= pattern_match_call_agg(linfo, ast[1], ast[2], ast[3], ast[4], ast[5:end])
  end
  if length(ast)>=6
    s *= pattern_match_call_join(linfo, ast[1], ast[2], ast[3], ast[4], ast[5],ast[6:end])
  end
  return s
end

function pattern_match_call_gemm_2d(fun::GlobalRef, C::RHSVar, tA::Char, tB::Char, A::RHSVar, B::RHSVar,
                C_total_size_x::LHSVar, C_total_size_y::LHSVar,
                C_block_x::LHSVar, C_block_y::LHSVar,
                C_local_size_x::LHSVar, C_local_size_y::LHSVar,
                A_total_size_x::LHSVar, A_total_size_y::LHSVar,
                A_block_x::LHSVar, A_block_y::LHSVar,
                A_local_size_x::LHSVar, A_local_size_y::LHSVar,
                B_total_size_x::LHSVar, B_total_size_y::LHSVar,
                B_block_x::LHSVar, B_block_y::LHSVar,
                B_local_size_x::LHSVar, B_local_size_y::LHSVar,
                 linfo)
    if fun.mod!=HPAT.API || fun.name!=:__hpat_gemm_2d
        return ""
    end
    cblas_fun = ""
    typ = ParallelAccelerator.CGen.getSymType(A, linfo)
    if ParallelAccelerator.CGen.getSymType(B, linfo) != typ || ParallelAccelerator.CGen.getSymType(C, linfo) != typ
        return ""
    end
    if typ==Array{Float32,2}
        cblas_fun = "psgemm_"
    elseif typ==Array{Float64,2}
        cblas_fun = "pdgemm_"
    else
        return ""
    end

    lda = ParallelAccelerator.CGen.from_arraysize(A,1,linfo)
    ldb = ParallelAccelerator.CGen.from_arraysize(B,1,linfo)
    ldc = ParallelAccelerator.CGen.from_arraysize(C,1,linfo)

    c_C = ParallelAccelerator.CGen.from_expr(C,linfo)
    c_A = ParallelAccelerator.CGen.from_expr(A,linfo)
    c_B = ParallelAccelerator.CGen.from_expr(B,linfo)

    c_C_total_size_x = ParallelAccelerator.CGen.from_expr(C_total_size_x,linfo)
    c_C_total_size_y = ParallelAccelerator.CGen.from_expr(C_total_size_y,linfo)
    c_C_block_x = ParallelAccelerator.CGen.from_expr(C_block_x,linfo)
    c_C_block_y = ParallelAccelerator.CGen.from_expr(C_block_y,linfo)
    c_C_local_size_x = ParallelAccelerator.CGen.from_expr(C_local_size_x,linfo)
    c_C_local_size_y = ParallelAccelerator.CGen.from_expr(C_local_size_y,linfo)

    c_A_total_size_x = ParallelAccelerator.CGen.from_expr(A_total_size_x,linfo)
    c_A_total_size_y = ParallelAccelerator.CGen.from_expr(A_total_size_y,linfo)
    c_A_block_x = ParallelAccelerator.CGen.from_expr(A_block_x,linfo)
    c_A_block_y = ParallelAccelerator.CGen.from_expr(A_block_y,linfo)
    c_A_local_size_x = ParallelAccelerator.CGen.from_expr(A_local_size_x,linfo)
    c_A_local_size_y = ParallelAccelerator.CGen.from_expr(A_local_size_y,linfo)

    c_B_total_size_x = ParallelAccelerator.CGen.from_expr(B_total_size_x,linfo)
    c_B_total_size_y = ParallelAccelerator.CGen.from_expr(B_total_size_y,linfo)
    c_B_block_x = ParallelAccelerator.CGen.from_expr(B_block_x,linfo)
    c_B_block_y = ParallelAccelerator.CGen.from_expr(B_block_y,linfo)
    c_B_local_size_x = ParallelAccelerator.CGen.from_expr(B_local_size_x,linfo)
    c_B_local_size_y = ParallelAccelerator.CGen.from_expr(B_local_size_y,linfo)

    s = ""
    s *= "$c_C;\n"
    s *= "MKL_INT desc_$c_C[9], desc_$c_A[9], desc_$c_B[9], info=0;\n"
    s *= "MKL_INT a32_$c_A_total_size_y = $c_A_total_size_y;\n"
    s *= "MKL_INT a32_$c_A_total_size_x = $c_A_total_size_x;\n"
    s *= "MKL_INT a32y_$c_A_block_y = $c_A_block_y;\n"
    s *= "MKL_INT a32_$c_A_block_x = $c_A_block_x;\n"
    s *= "MKL_INT a32_$c_A_local_size_x = $c_A_local_size_x;\n"
    s *= "MKL_INT a32_$c_A_local_size_y = $c_A_local_size_y;\n"

    s *= "MKL_INT a32_$c_B_total_size_y = $c_B_total_size_y;\n"
    s *= "MKL_INT a32_$c_B_total_size_x = $c_B_total_size_x;\n"
    s *= "MKL_INT a32y_$c_B_block_y = $c_B_block_y;\n"
    s *= "MKL_INT a32_$c_B_block_x = $c_B_block_x;\n"
    s *= "MKL_INT a32_$c_B_local_size_x = $c_B_local_size_x;\n"
    s *= "MKL_INT a32_$c_B_local_size_y = $c_B_local_size_y;\n"

    s *= "MKL_INT a32_$c_C_total_size_y = $c_C_total_size_y;\n"
    s *= "MKL_INT a32_$c_C_total_size_x = $c_C_total_size_x;\n"
    s *= "MKL_INT a32y_$c_C_block_y = $c_C_block_y;\n"
    s *= "MKL_INT a32_$c_C_block_x = $c_C_block_x;\n"
    s *= "MKL_INT a32_$c_C_local_size_x = $c_C_local_size_x;\n"
    s *= "MKL_INT a32_$c_C_local_size_y = $c_C_local_size_y;\n"

    s *= "descinit_( desc_$c_A, &a32_$c_A_total_size_y, &a32_$c_A_total_size_x, &a32y_$c_A_block_y, &a32_$c_A_block_x, &i_zero, &i_zero, &ictxt, &a32_$c_A_local_size_x, &info );\n"
    s *= "descinit_( desc_$c_B, &a32_$c_B_total_size_y, &a32_$c_B_total_size_x, &a32y_$c_B_block_y, &a32_$c_B_block_x, &i_zero, &i_zero, &ictxt, &a32_$c_B_local_size_x, &info );\n"
    s *= "descinit_( desc_$c_C, &a32_$c_C_total_size_y, &a32_$c_C_total_size_x, &a32y_$c_C_block_y, &a32_$c_C_block_x, &i_zero, &i_zero, &ictxt, &a32_$c_C_local_size_x, &info );\n"

    # GEMM wants dimensions after possible transpose
    m = (tA == 'N') ? c_A_total_size_y : c_A_total_size_x
    k = (tA == 'N') ? c_A_total_size_x : c_A_total_size_y
    n = (tB == 'N') ? c_B_total_size_x : c_B_total_size_y


    _tA = tA == 'N' ? "\"N\"" : "\"T\""
    _tB = tB == 'N' ? "\"N\"" : "\"T\""

    s *= "MKL_INT a32m_$m = $m;\n"
    s *= "MKL_INT a32n_$n = $n;\n"
    s *= "MKL_INT a32k_$k = $k;\n"

    s *= """$(cblas_fun)($(_tA), $(_tB), &a32m_$m, &a32n_$n, &a32k_$k,&one,
        $c_A.data, &i_one, &i_one, desc_$c_A, $c_B.data, &i_one, &i_one, desc_$c_B,
         &zero, $c_C.data, &i_one, &i_one, desc_$c_C)"""


    return s
end

function pattern_match_call_gemm_2d(fun::ANY, C::ANY, tA::ANY, tB::ANY, A::ANY, B::ANY,
            C_total_size_x::ANY, C_total_size_y::ANY,
            C_block_x::ANY, C_block_y::ANY,
            C_local_size_x::ANY, C_local_size_y::ANY,
            A_total_size_x::ANY, A_total_size_y::ANY,
            A_block_x::ANY, A_block_y::ANY,
            A_local_size_x::ANY, A_local_size_y::ANY,
            B_total_size_x::ANY, B_total_size_y::ANY,
            B_block_x::ANY, B_block_y::ANY,
            B_local_size_x::ANY, B_local_size_y::ANY,linfo)
    return ""
end

function pattern_match_call_h5size(fun::GlobalRef, id, lhs, linfo)
    @dprintln(3, "assignment pattern match h5size: ", fun, " ", id)
    s = ""
    if fun.name==:__hpat_data_source_HDF5_size
        num = ParallelAccelerator.CGen.from_expr(id, linfo)
        s = "hid_t space_id_$num = H5Dget_space(dataset_id_$num);\n"
        s *= "assert(space_id_$num != -1);\n"
        s *= "hsize_t data_ndim_$num = H5Sget_simple_extent_ndims(space_id_$num);\n"
        s *= "hsize_t space_dims_$num[data_ndim_$num];\n"
        s *= "H5Sget_simple_extent_dims(space_id_$num, space_dims_$num, NULL);\n"
    end
    return s
end

pattern_match_call_h5size(fun::ANY, id, lhs, linfo) = ""

function from_assignment_match_transpose_hcat(lhs, rhs::Expr, linfo)
    s = ""
    if isCall(rhs) && rhs.args[1]==GlobalRef(HPAT.API,:__hpat_transpose_hcat)
        args = getCallArguments(rhs)
        in_typ = getType(args[1], linfo)
        # TODO: transpose hcat of single values like transpose(hcat(1,2,3)) => 2D array [1;2;3]
        #if !(in_typ<:Array)
        #    return single_value_hcat(lhs, args, linfo)
        #end
        for a in args
            atyp = getType(a, linfo)
            @assert atyp<:Array && ndims(atyp)==1 "CGen only supports hcat of 1D arrays"
        end
        typ = eltype(getType(lhs, linfo))
        size = length(args)
        ctyp = ParallelAccelerator.CGen.toCtype(typ)
        len = ParallelAccelerator.CGen.from_arraysize(args[1],1,linfo)
        clhs = ParallelAccelerator.CGen.from_expr(lhs,linfo)
        s *= "$clhs = j2c_array<$ctyp>::new_j2c_array_2d(NULL, $size, $len);\n"
        s *= "for(int i=0; i<$len; i++) {\n"
        for j in 1:size
            arr = ParallelAccelerator.CGen.from_expr(args[j],linfo)
            s *= "$clhs.data[i*$size+$(j-1)] = $arr.data[i];\n"
        end
        s *= "}\n"
    end
    return s
end

function from_assignment_match_dist(lhs::RHSVar, rhs::Expr, linfo)
    @dprintln(3, "assignment pattern match dist2: ",lhs," = ",rhs)
    s = ""
    s *= from_assignment_match_transpose_hcat(lhs, rhs, linfo)
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
    elseif rhs.head==:call && length(rhs.args)==1 && rhs.args[1]==GlobalRef(HPAT.API,:hpat_dist_node_id)
        @dprintln(3, "node_id call")
        c_lhs = ParallelAccelerator.CGen.from_expr(lhs, linfo)
        return "MPI_Comm_rank(MPI_COMM_WORLD,&$c_lhs);"
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

function get_mpi_type_from_array(input_array,linfo)
    input_type = eltype(ParallelAccelerator.CGen.getSymType(input_array,linfo))
    return get_mpi_type_from_var_type(input_type)
end

function get_j2c_type_from_array(input_array,linfo)
    input_type = eltype(ParallelAccelerator.CGen.getSymType(input_array,linfo))
    return ParallelAccelerator.CGen.toCtype(input_type)
end

function get_mpi_type_from_var_type(var_typ)
    mpi_type = ""
    if var_typ==Float64
        mpi_type = "MPI_DOUBLE"
    elseif var_typ==Float32
        mpi_type = "MPI_FLOAT"
    elseif var_typ==Int32
        mpi_type = "MPI_INT"
    elseif var_typ==Int64
        mpi_type = "MPI_INT64_T"
    elseif var_typ==Bool
        mpi_type = "MPI_C_BOOL"
    else
        throw("CGen unsupported MPI reduction type $var_typ")
    end
    return mpi_type
end

end # module
