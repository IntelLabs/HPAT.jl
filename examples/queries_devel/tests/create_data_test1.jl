using HDF5

u1 = [1,2,3,1,2]
v2 = [1.1,2.1,3.1,3.2,1.9]

u2 = [1,3]
v3 = [7.1,8.3]

file_name = "test1_1.hdf5"
file_name2 = "test1_2.hdf5"

h5write(file_name ,"/userid",u1)
h5write(file_name, "/val2",v2)

h5write(file_name2, "/userid",u2)
h5write(file_name2, "/val3",v3)


