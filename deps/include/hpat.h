#include <stdint.h>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <iostream>
#include <ctime>

#define CHECKPOINT_DEBUG

class file_j2c_array_io : public j2c_array_io {
protected:
    std::fstream *checkpoint_file;
public:
    file_j2c_array_io(std::fstream *cf) : checkpoint_file(cf) {}

    virtual void write_in(void *arr, uint64_t arr_length, unsigned int elem_size, bool immutable) {
#ifdef CHECKPOINT_DEBUG
        std::cout << "checkpoint write arr_length = " << arr_length << " elem_size = " << elem_size << std::endl;
#endif
        checkpoint_file->write((char*)&arr_length, sizeof(arr_length));
        checkpoint_file->write((char*)&elem_size,  sizeof(elem_size));
        checkpoint_file->write((char*)arr, arr_length * elem_size);
    }

    virtual void write(void *arr, uint64_t arr_length, unsigned int elem_size, bool immutable) {
        write_in(arr, arr_length, elem_size, immutable);
    }

    virtual void read(void **arr, uint64_t *length) {
        unsigned int elem_size;
        uint64_t arr_length;
        checkpoint_file->read((char*)&arr_length, sizeof(arr_length));
        checkpoint_file->read((char*)&elem_size, sizeof(elem_size));
        char *newarr = (char*)malloc(arr_length * elem_size);
        checkpoint_file->read(newarr, arr_length * elem_size);
#ifdef CHECKPOINT_DEBUG
        std::cout << "checkpoint read arr_length = " << arr_length << " elem_size = " << elem_size << std::endl;
#endif
        *length = arr_length;
        *arr = newarr;
    }
};

int32_t g_checkpoint_handle = 0;
std::fstream checkpoint_file;
int64_t g_unique;
int32_t g_checkpoint_start_time;
double  g_checkpoint_time = 1.0 / 60.0; // Estimated 1 min checkpoint time.  1/60 of hour.

double __hpat_get_checkpoint_time(int64_t unique_checkpoint_location) {
    return g_checkpoint_time;
}

int32_t __hpat_start_checkpoint(int64_t unique_checkpoint_location) {
    g_checkpoint_start_time = std::time(nullptr);

    MPI_Barrier(MPI_COMM_WORLD);
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_start_checkpoint location = " << unique_checkpoint_location << " checkpoint# = " << g_checkpoint_handle + 1 << " start_time = " << g_checkpoint_start_time << std::endl;
#endif
        checkpoint_file.open("hpat_checkpoint_in_progress", std::ios::out | std::ios::binary);
        if (checkpoint_file.fail()) {
          std::cout << "Failed to open checkpoint file." << std::endl;
        }
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
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_value_checkpoint int32 value = " << value << std::endl;
#endif
        assert(checkpoint_file.is_open());
        checkpoint_file.write((char*)&value, sizeof(value));
    }
    return 0;
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, int64_t value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_value_checkpoint int64 value = " << value << std::endl;
#endif
        assert(checkpoint_file.is_open());
        checkpoint_file.write((char*)&value, sizeof(value));
    }
    return 0;
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, float value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_value_checkpoint float value = " << value << std::endl;
#endif
        assert(checkpoint_file.is_open());
        checkpoint_file.write((char*)&value, sizeof(value));
    }
    return 0;
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, double value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_value_checkpoint double value = " << value << std::endl;
#endif
        assert(checkpoint_file.is_open());
        checkpoint_file.write((char*)&value, sizeof(value));
    }
    return 0;
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, j2c_array<float> &value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_value_checkpoint array<float>" << std::endl;
#endif
        assert(checkpoint_file.is_open());
        file_j2c_array_io fjai(&checkpoint_file);
        value.serialize(&fjai);
    }
    return 0;
}

int32_t __hpat_value_checkpoint(int32_t checkpoint_handle, j2c_array<double> &value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_value_checkpoint array<double>" << std::endl;
#endif
        assert(checkpoint_file.is_open());
        file_j2c_array_io fjai(&checkpoint_file);
        value.serialize(&fjai);
    }
    return 0;
}

int32_t __hpat_end_checkpoint(int32_t checkpoint_handle) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        checkpoint_file.close();

        std::stringstream ss;
        ss << "checkpoint_file_" << g_unique; 
        remove(ss.str().c_str());
        rename("hpat_checkpoint_in_progress", ss.str().c_str()); 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int32_t cur_time = std::time(nullptr);
    int32_t checkpoint_time = cur_time - g_checkpoint_start_time;
    if (checkpoint_time < 1) {
        checkpoint_time = 1;
    }
    g_checkpoint_time = checkpoint_time / 3600.0;
#ifdef CHECKPOINT_DEBUG
    std::cout << "__hpat_end_checkpoint cur_time(s) = " << cur_time << " checkpoint_length(s) = " << checkpoint_time << " new_checkpoint_est(h) = " << g_checkpoint_time << std::endl;
#endif
    return 0;
}

int32_t __hpat_finish_checkpoint_region(int64_t unique_checkpoint_location) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        std::stringstream ss;
        ss << "checkpoint_file_" << g_unique; 
        remove(ss.str().c_str());
    }
    return 0;
}



// Restore functions are implemented below.


int32_t __hpat_restore_checkpoint_start(int64_t unique_checkpoint_location) {
    MPI_Barrier(MPI_COMM_WORLD);
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_restore_checkpoint_start location = " << unique_checkpoint_location << std::endl;
#endif
        g_unique = unique_checkpoint_location;
        std::stringstream ss;
        ss << "checkpoint_file_" << g_unique; 
 
        checkpoint_file.open(ss.str().c_str(), std::ios::in | std::ios::binary);
        if (checkpoint_file.fail()) {
          std::cout << "Failed to open checkpoint file." << std::endl;
          exit(-1);
        }
        return ++g_checkpoint_handle; 
    } else {
        return 0;
    }
}

int32_t __hpat_restore_checkpoint_value(int32_t checkpoint_handle, int32_t &value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        checkpoint_file.read((char*)&value, sizeof(value));
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_restore_checkpoint_value int32 value = " << value << std::endl;
#endif
    }
    return 0;
}

int32_t __hpat_restore_checkpoint_value(int32_t checkpoint_handle, int64_t &value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        checkpoint_file.read((char*)&value, sizeof(value));
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_restore_checkpoint_value int64 value = " << value << std::endl;
#endif
    }
    return 0;
}

int32_t __hpat_restore_checkpoint_value(int32_t checkpoint_handle, float &value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        checkpoint_file.read((char*)&value, sizeof(value));
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_restore_checkpoint_value float value = " << value << std::endl;
#endif
    }
    return 0;
}

int32_t __hpat_restore_checkpoint_value(int32_t checkpoint_handle, double &value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        checkpoint_file.read((char*)&value, sizeof(value));
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_restore_checkpoint_value double value = " << value << std::endl;
#endif
    }
    return 0;
}

int32_t __hpat_restore_checkpoint_value(int32_t checkpoint_handle, j2c_array<float> &value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_restore_checkpoint_value array<float>" << std::endl;
#endif
        assert(checkpoint_file.is_open());
        file_j2c_array_io fjai(&checkpoint_file);
        value.deserialize(&fjai);
    }
    return 0;
}

int32_t __hpat_restore_checkpoint_value(int32_t checkpoint_handle, j2c_array<double> &value) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
    if (__hpat_node_id == 0) {
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_restore_checkpoint_value array<double>" << std::endl;
#endif
        assert(checkpoint_file.is_open());
        file_j2c_array_io fjai(&checkpoint_file);
        value.deserialize(&fjai);
    }
    return 0;
}

int32_t __hpat_restore_checkpoint_end(int32_t checkpoint_handle) {
    int32_t __hpat_node_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&__hpat_node_id);
#ifdef CHECKPOINT_DEBUG
        std::cout << "__hpat_restore_checkpoint_value array<double>" << " id = " << __hpat_node_id << std::endl;
#endif
    if (__hpat_node_id == 0) {
        assert(checkpoint_file.is_open());
        checkpoint_file.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}
