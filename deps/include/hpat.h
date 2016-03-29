#include <stdint.h>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <iostream>

class file_j2c_array_io : public j2c_array_io {
protected:
    std::ofstream *checkpoint_file;
public:
    file_j2c_array_io(std::ofstream *cf) : checkpoint_file(cf) {}

    virtual void write_in(void *arr, uint64_t arr_length, unsigned int elem_size, bool immutable) {
        checkpoint_file->write((char*)&arr_length, sizeof(arr_length));
        checkpoint_file->write((char*)&elem_size,  sizeof(elem_size));
        checkpoint_file->write((char*)arr, arr_length * elem_size);
    }

    virtual void write(void *arr, uint64_t arr_length, unsigned int elem_size, bool immutable) {
        write_in(arr, arr_length, elem_size, immutable);
        //checkpoint_file->write((char*)&arr_length, sizeof(arr_length));
        //checkpoint_file->write((char*)&elem_size,  sizeof(elem_size));
        //checkpoint_file->write((char*)arr, arr_length * elem_size);
    }

    virtual void read(void **arr, uint64_t *length) {
        unsigned int elem_size;
        checkpoint_file->read((char*)length,     sizeof(*length));
        checkpoint_file->read((char*)&elem_size, sizeof(elem_size));
        checkpoint_file->read((char*)arr, *length * elem_size);
    }
};

int32_t g_checkpoint_handle = 0;
std::ofstream checkpoint_file;
int64_t g_unique;

int32_t __hpat_start_checkpoint(int64_t unique_checkpoint_location) {
    MPI_Barrier(MPI_COMM_WORLD);
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        checkpoint_file.open("hpat_checkpoint_in_progress", std::ios::binary);
        g_unique = unique_checkpoint_location;
        return ++g_checkpoint_handle; 
    } else {
        return 0;
    }
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, int32_t value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        checkpoint_file.write((char*)&value, sizeof(value));
    }
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, int64_t value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        checkpoint_file.write((char*)&value, sizeof(value));
    }
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, float value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        checkpoint_file.write((char*)&value, sizeof(value));
    }
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, double value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        checkpoint_file.write((char*)&value, sizeof(value));
    }
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, j2c_array<float> &value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        file_j2c_array_io fjai(&checkpoint_file);
        value.serialize(&fjai);
    }
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, j2c_array<double> &value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        file_j2c_array_io fjai(&checkpoint_file);
        value.serialize(&fjai);
    }
}

int32_t __hpat_end_checkpoint(int32_t checkpoint_handle) {
    if (__hpat_node_id == 0) {
        std::stringstream ss;
        ss << "checkpoint_file_" << g_unique; 
        remove(ss.str().c_str());
        rename("hpat_checkpoint_in_progress", ss.str().c_str()); 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}
