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
        s *=  "hsize_t CGen_HDF5_start_$num[data_ndim_$num];\n"
        s *=  "hsize_t CGen_HDF5_stride_$num[data_ndim_$num];\n"
        s *= "hsize_t CGen_HDF5_count_$num[data_ndim_$num];\n"
        s *=  "hsize_t CGen_HDF5_block_$num[data_ndim_$num];\n"
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
        s *= """ret_$num = H5Sselect_hyperslab(space_id_$num, H5S_SELECT_SET, CGen_HDF5_start_$num,
                         CGen_HDF5_stride_$num, CGen_HDF5_count_$num, CGen_HDF5_block_$num);\n"""
        s *= "assert(ret_$num != -1);\n"

        # size of memory to read to
        s *=  "hsize_t CGen_HDF5_memsize_$num[data_ndim_$num];\n"
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
