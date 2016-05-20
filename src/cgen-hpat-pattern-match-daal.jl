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



function pattern_match_call_kmeans(f::GlobalRef, cluster_out::RHSVar, arr::RHSVar, 
                                   num_clusters::RHSVar, start::Symbol, count::Symbol, 
                                   col_size::Union{RHSVar,Int,Expr}, tot_row_size::Union{RHSVar,Int,Expr}, linfo)
    s = ""
    if f.name==:Kmeans_dist
        c_arr = ParallelAccelerator.CGen.from_expr(arr, linfo)
        c_num_clusters = ParallelAccelerator.CGen.from_expr(num_clusters, linfo)
        c_col_size = ParallelAccelerator.CGen.from_expr(col_size, linfo)
        c_tot_row_size = ParallelAccelerator.CGen.from_expr(tot_row_size, linfo)
        c_cluster_out = ParallelAccelerator.CGen.from_expr(cluster_out, linfo)        
        
        s *= """
        services::Environment::getInstance()->setNumberOfThreads(omp_get_max_threads());
        byte   *nodeCentroids;
        size_t CentroidsArchLength;
        services::SharedPtr<NumericTable> centroids;
        InputDataArchive centroidsDataArch;
        int nIterations = 10;
        int mpi_root = 0;
        int rankId = __hpat_node_id;

        HomogenNumericTable<double>* dataTable = new HomogenNumericTable<double>((double*)$c_arr.getData(), $c_col_size, $count);
        services::SharedPtr<NumericTable> dataTablePointer(dataTable);
        kmeans::init::Distributed<step1Local,double,kmeans::init::randomDense>
                       localInit($c_num_clusters, $c_tot_row_size, $start);
        localInit.input.set(kmeans::init::data, dataTablePointer);
        
        /* Compute k-means */
        localInit.compute();

        /* Serialize partial results required by step 2 */
        services::SharedPtr<byte> serializedData;
        InputDataArchive dataArch;
        localInit.getPartialResult()->serialize( dataArch );
        size_t perNodeArchLength = dataArch.getSizeOfArchive();

        /* Serialized data is of equal size on each node if each node called compute() equal number of times */
        if (rankId == mpi_root)
        {   
            serializedData = services::SharedPtr<byte>( new byte[ perNodeArchLength * __hpat_num_pes ] );
        }   

        byte *nodeResults = new byte[ perNodeArchLength ];
        dataArch.copyArchiveToArray( nodeResults, perNodeArchLength );

        /* Transfer partial results to step 2 on the root node */
        MPI_Gather( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, mpi_root,
                MPI_COMM_WORLD);

        delete[] nodeResults;

        if(rankId == mpi_root)
        {   
            /* Create an algorithm to compute k-means on the master node */
            kmeans::init::Distributed<step2Master, double, kmeans::init::randomDense> masterInit($c_num_clusters);

            for( size_t i = 0; i < __hpat_num_pes ; i++ )
            {   
                /* Deserialize partial results from step 1 */
                OutputDataArchive dataArch( serializedData.get() + perNodeArchLength * i, perNodeArchLength );

                services::SharedPtr<kmeans::init::PartialResult> dataForStep2FromStep1 = services::SharedPtr<kmeans::init::PartialResult>(
                                                                               new kmeans::init::PartialResult() );
                dataForStep2FromStep1->deserialize(dataArch);

                /* Set local partial results as input for the master-node algorithm */
                masterInit.input.add(kmeans::init::partialResults, dataForStep2FromStep1 );
        }

        /* Merge and finalizeCompute k-means on the master node */
        masterInit.compute();
        masterInit.finalizeCompute();

        centroids = masterInit.getResult()->get(kmeans::init::centroids);
        }
        
        for(int iter=0; iter<nIterations; iter++) 
        {
        
            if(rankId == mpi_root)
            {
                /*Retrieve the algorithm results and serialize them */
                centroids->serialize( centroidsDataArch );
                CentroidsArchLength = centroidsDataArch.getSizeOfArchive();
            }

             /* Get partial results from the root node */
             MPI_Bcast( &CentroidsArchLength, sizeof(size_t), MPI_CHAR, mpi_root, MPI_COMM_WORLD );

              nodeCentroids = new byte[ CentroidsArchLength ];

            if(rankId == mpi_root)
            {
                centroidsDataArch.copyArchiveToArray( nodeCentroids, CentroidsArchLength );
            }

            MPI_Bcast( nodeCentroids, CentroidsArchLength, MPI_CHAR, mpi_root, MPI_COMM_WORLD );

            /* Deserialize centroids data */
            OutputDataArchive centroidsDataArch( nodeCentroids, CentroidsArchLength );

            centroids = services::SharedPtr<NumericTable>( new HomogenNumericTable<double>() );

            centroids->deserialize(centroidsDataArch);

            /* Create an algorithm to compute k-means on local nodes */
            kmeans::Distributed<step1Local> localAlgorithm($c_num_clusters);

            /* Set the input data set to the algorithm */
            localAlgorithm.input.set(kmeans::data,           dataTablePointer);
            localAlgorithm.input.set(kmeans::inputCentroids, centroids);
    
            /* Compute k-means */
            localAlgorithm.compute();

            /* Serialize partial results required by step 2 */
            services::SharedPtr<byte> serializedData;
            InputDataArchive dataArch;
            localAlgorithm.getPartialResult()->serialize( dataArch );
            size_t perNodeArchLength = dataArch.getSizeOfArchive();

            /* Serialized data is of equal size on each node if each node called compute() equal number of times */
            if (rankId == mpi_root)
            {
                serializedData = services::SharedPtr<byte>( new byte[ perNodeArchLength * __hpat_num_pes ] );
            }
            byte *nodeResults = new byte[ perNodeArchLength ];
            dataArch.copyArchiveToArray( nodeResults, perNodeArchLength );

            /* Transfer partial results to step 2 on the root node */
            MPI_Gather( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, mpi_root,
                        MPI_COMM_WORLD);

            delete[] nodeResults;

            if(rankId == mpi_root)
            {
               /* Create an algorithm to compute k-means on the master node */
               kmeans::Distributed<step2Master> masterAlgorithm($c_num_clusters);

               for( size_t i = 0; i < __hpat_num_pes ; i++ )
                {
                    /* Deserialize partial results from step 1 */
                    OutputDataArchive dataArch( serializedData.get() + perNodeArchLength * i, perNodeArchLength );

                    services::SharedPtr<kmeans::PartialResult> dataForStep2FromStep1 = services::SharedPtr<kmeans::PartialResult>(
                                                                               new kmeans::PartialResult() );
                    dataForStep2FromStep1->deserialize(dataArch);

                    /* Set local partial results as input for the master-node algorithm */
                    masterAlgorithm.input.add(kmeans::partialResults, dataForStep2FromStep1 );
                }

                /* Merge and finalizeCompute k-means on the master node */
                masterAlgorithm.compute();
                masterAlgorithm.finalizeCompute();

                /* Retrieve the algorithm results */
                centroids = masterAlgorithm.getResult()->get(kmeans::centroids);
            }
            delete[] nodeCentroids;
        }
        
        BlockDescriptor<double> block;
        
        //std::cout<<centroids->getNumberOfRows()<<std::endl;
        //std::cout<<centroids->getNumberOfColumns()<<std::endl;
        
        centroids->getBlockOfRows(0, $c_num_clusters, readOnly, block);
        double* out_arr = block.getBlockPtr();
        //std::cout<<"output ";
        //for(int i=0; i<$c_col_size*$c_num_clusters; i++)
        //{
        //    std::cout<<" "<<out_arr[i];
        //}
        //std::cout<<std::endl;
        int64_t res_dims[] = {$c_col_size,$c_num_clusters};
        double* out_data = new double[$c_col_size*$c_num_clusters];
        memcpy(out_data, block.getBlockPtr(), $c_col_size*$c_num_clusters*sizeof(double));
        j2c_array<double> kmeans_out(out_data,2,res_dims);
        $c_cluster_out = kmeans_out;
    """
        
    end
    return s
end


function pattern_match_call_kmeans(f::ANY, cluster_out::ANY, arr::ANY, num_clusters::ANY, start::ANY, count::ANY, cols::ANY, rows::ANY, linfo)
    return ""
end

function pattern_match_call_linear_regression(f::GlobalRef, coeff_out::RHSVar, points::RHSVar, 
                                   responses::RHSVar, start_points::Symbol, count_points::Symbol, 
                                   col_size_points::Union{RHSVar,Int,Expr}, tot_row_size_points::Union{RHSVar,Int,Expr},
                                   start_responses::Symbol, count_responses::Symbol, 
                                   col_size_responses::Union{RHSVar,Int,Expr}, tot_row_size_responses::Union{RHSVar,Int,Expr}, linfo)
    s = ""
    if f.name==:LinearRegression_dist
        c_points = ParallelAccelerator.CGen.from_expr(points, linfo)
        c_responses = ParallelAccelerator.CGen.from_expr(responses, linfo)
        c_col_size_points = ParallelAccelerator.CGen.from_expr(col_size_points, linfo)
        c_tot_row_size_points = ParallelAccelerator.CGen.from_expr(tot_row_size_points, linfo)
        c_col_size_responses = ParallelAccelerator.CGen.from_expr(col_size_responses, linfo)
        c_tot_row_size_responses = ParallelAccelerator.CGen.from_expr(tot_row_size_responses, linfo)
        c_coeff_out = ParallelAccelerator.CGen.from_expr(coeff_out, linfo)
        s = """
            assert($c_tot_row_size_points==$c_tot_row_size_responses);
            int mpi_root = 0;
            int rankId = __hpat_node_id;
            services::Environment::getInstance()->setNumberOfThreads(omp_get_max_threads());
            
            HomogenNumericTable<double>* dataTable = new HomogenNumericTable<double>((double*)$c_points.getData(), $c_col_size_points, $count_points);
            HomogenNumericTable<double>* responseTable = new HomogenNumericTable<double>((double*)$c_responses.getData(), $c_col_size_responses, $count_responses);
            services::SharedPtr<NumericTable> trainData(dataTable);
            services::SharedPtr<NumericTable> trainDependentVariables(responseTable);
            services::SharedPtr<linear_regression::training::Result> trainingResult; 
        
            /* Create an algorithm object to train the multiple linear regression model based on the local-node data */
            linear_regression::training::Distributed<step1Local, double, linear_regression::training::qrDense> localAlgorithm;
        
            /* Pass a training data set and dependent values to the algorithm */
            localAlgorithm.input.set(linear_regression::training::data, trainData);
            localAlgorithm.input.set(linear_regression::training::dependentVariables, trainDependentVariables);
        
            /* Train the multiple linear regression model on local nodes */
            localAlgorithm.compute();
        
            /* Serialize partial results required by step 2 */
            services::SharedPtr<byte> serializedData;
            InputDataArchive dataArch;
            localAlgorithm.getPartialResult()->serialize( dataArch );
            size_t perNodeArchLength = dataArch.getSizeOfArchive();
        
            /* Serialized data is of equal size on each node if each node called compute() equal number of times */
            if (rankId == mpi_root)
            {
                serializedData = services::SharedPtr<byte>( new byte[ perNodeArchLength * __hpat_num_pes] );
            }
        
            byte *nodeResults = new byte[ perNodeArchLength ];
            dataArch.copyArchiveToArray( nodeResults, perNodeArchLength );
        
            /* Transfer partial results to step 2 on the root node */
            MPI_Gather( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, mpi_root,
                        MPI_COMM_WORLD);
        
            delete[] nodeResults;
            services::SharedPtr<NumericTable> trainingCoeffsTable;
            if(rankId == mpi_root)
            {
                /* Create an algorithm object to build the final multiple linear regression model on the master node */
                linear_regression::training::Distributed<step2Master, double, linear_regression::training::qrDense> masterAlgorithm;
        
                for( size_t i = 0; i < __hpat_num_pes; i++ )
                {
                    /* Deserialize partial results from step 1 */
                    OutputDataArchive dataArch( serializedData.get() + perNodeArchLength * i, perNodeArchLength );
        
                    services::SharedPtr<linear_regression::training::PartialResult> dataForStep2FromStep1 = services::SharedPtr<linear_regression::training::PartialResult>
                                                                               ( new linear_regression::training::PartialResult() );
                    dataForStep2FromStep1->deserialize(dataArch);
        
                    /* Set the local multiple linear regression model as input for the master-node algorithm */
                    masterAlgorithm.input.add(linear_regression::training::partialModels, dataForStep2FromStep1);
                }
        
                /* Merge and finalizeCompute the multiple linear regression model on the master node */
                masterAlgorithm.compute();
                masterAlgorithm.finalizeCompute();
        
                /* Retrieve the algorithm results */
                trainingResult = masterAlgorithm.getResult();
                // printNumericTable(trainingResult->get(linear_regression::training::model)->getBeta(), "Linear Regression coefficients:");
                trainingCoeffsTable = trainingResult->get(linear_regression::training::model)->getBeta();
            
                BlockDescriptor<double> block;
        
            //std::cout<<trainingCoeffsTable->getNumberOfRows()<<std::endl;
            //std::cout<<trainingCoeffsTable->getNumberOfColumns()<<std::endl;
        
                trainingCoeffsTable->getBlockOfRows(0, $c_col_size_responses, readOnly, block);
                double* out_arr = block.getBlockPtr();
            
            // assuming intercept is required
            int64_t coeff_size = $c_col_size_points+1;
            //std::cout<<"output ";
            //for(int i=0; i<coeff_size*$c_col_size_responses; i++)
            //{
            //    std::cout<<" "<<out_arr[i];
            //}
            //std::cout<<std::endl;
            
                int64_t res_dims[] = {coeff_size,$c_col_size_responses};
                double* out_data = new double[coeff_size*$c_col_size_responses];
                memcpy(out_data, block.getBlockPtr(), coeff_size*$c_col_size_responses*sizeof(double));
                j2c_array<double> linear_regression_out(out_data,2,res_dims);
                $c_coeff_out = linear_regression_out;
            }
            """
    end
    return s
end

function pattern_match_call_linear_regression(f::ANY, coeff_out::ANY, arr::ANY, num_clusters::ANY, 
          start::ANY, count::ANY, cols::ANY, rows::ANY, start2::ANY, count2::ANY, cols2::ANY, rows2::ANY, linfo)
    return ""
end

function pattern_match_call_naive_bayes(f::GlobalRef, coeff_out::RHSVar, points::RHSVar, 
                                   labels::RHSVar, num_classes::Union{RHSVar,Int,Expr}, start_points::Symbol, count_points::Symbol, 
                                   col_size_points::Union{RHSVar,Int,Expr}, tot_row_size_points::Union{RHSVar,Int,Expr},
                                   start_labels::Symbol, count_labels::Symbol, 
                                   col_size_labels::Union{RHSVar,Int,Expr}, tot_row_size_labels::Union{RHSVar,Int,Expr}, linfo)
    s = ""
    if f.name==:NaiveBayes_dist
        c_points = ParallelAccelerator.CGen.from_expr(points, linfo)
        c_labels = ParallelAccelerator.CGen.from_expr(labels, linfo)
        c_col_size_points = ParallelAccelerator.CGen.from_expr(col_size_points, linfo)
        c_tot_row_size_points = ParallelAccelerator.CGen.from_expr(tot_row_size_points, linfo)
        c_col_size_labels = ParallelAccelerator.CGen.from_expr(col_size_labels, linfo)
        c_tot_row_size_labels = ParallelAccelerator.CGen.from_expr(tot_row_size_labels, linfo)
        c_coeff_out = ParallelAccelerator.CGen.from_expr(coeff_out, linfo)
        c_num_classes = ParallelAccelerator.CGen.from_expr(num_classes, linfo)
        
        s = """
            assert($c_tot_row_size_points==$c_tot_row_size_labels);
            int mpi_root = 0;
            int rankId = __hpat_node_id;
            services::Environment::getInstance()->setNumberOfThreads(omp_get_max_threads());
            
            HomogenNumericTable<double>* dataTable = new HomogenNumericTable<double>((double*)$c_points.getData(), $c_col_size_points, $count_points);
            HomogenNumericTable<double>* responseTable = new HomogenNumericTable<double>((double*)$c_labels.getData(), $c_col_size_labels, $count_labels);
            services::SharedPtr<NumericTable> trainData(dataTable);
            services::SharedPtr<NumericTable> trainGroundTruth(responseTable);
            services::SharedPtr<multinomial_naive_bayes::training::Result> trainingResult; 
        
        
            /* Create an algorithm object to train the Na__ve Bayes model based on the local-node data */
            multinomial_naive_bayes::training::Distributed<step1Local> localAlgorithm($c_num_classes);
        
            /* Pass a training data set and dependent values to the algorithm */
            localAlgorithm.input.set(classifier::training::data,   trainData);
            localAlgorithm.input.set(classifier::training::labels, trainGroundTruth);
        
            /* Train the Na__ve Bayes model on local nodes */
            localAlgorithm.compute();
        
            /* Serialize partial results required by step 2 */
            services::SharedPtr<byte> serializedData;
            InputDataArchive dataArch;
            localAlgorithm.getPartialResult()->serialize(dataArch);
            size_t perNodeArchLength = dataArch.getSizeOfArchive();
        
            /* Serialized data is of equal size on each node if each node called compute() equal number of times */
            if (rankId == mpi_root)
            {
                serializedData = services::SharedPtr<byte>(new byte[perNodeArchLength * __hpat_num_pes]);
            }
        
            byte *nodeResults = new byte[perNodeArchLength];
            dataArch.copyArchiveToArray( nodeResults, perNodeArchLength );
        
            /* Transfer partial results to step 2 on the root node */
            MPI_Gather( nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, mpi_root,
                        MPI_COMM_WORLD);
        
            delete[] nodeResults;
        
            if(rankId == mpi_root)
            {
                /* Create an algorithm object to build the final Na__ve Bayes model on the master node */
                multinomial_naive_bayes::training::Distributed<step2Master> masterAlgorithm($c_num_classes);
        
                for(size_t i = 0; i < __hpat_num_pes ; i++)
                {
                    /* Deserialize partial results from step 1 */
                    OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i, perNodeArchLength);
        
                    services::SharedPtr<classifier::training::PartialResult> dataForStep2FromStep1 = services::SharedPtr<classifier::training::PartialResult>
                                                                               (new classifier::training::PartialResult());
                    dataForStep2FromStep1->deserialize(dataArch);
        
                    /* Set the local Na__ve Bayes model as input for the master-node algorithm */
                    masterAlgorithm.input.add(classifier::training::partialModels, dataForStep2FromStep1);
                }
        
                /* Merge and finalizeCompute the Na__ve Bayes model on the master node */
                masterAlgorithm.compute();
                masterAlgorithm.finalizeCompute();
        
                /* Retrieve the algorithm results */
                trainingResult = masterAlgorithm.getResult();
                
                services::SharedPtr<NumericTable> trainingLogpTable = trainingResult->get(classifier::training::model)->getLogP();
                services::SharedPtr<NumericTable> trainingLogThetaTable = trainingResult->get(classifier::training::model)->getLogTheta();
            
                BlockDescriptor<double> block1, block2;
            
                //std::cout<<trainingLogThetaTable->getNumberOfRows()<<std::endl;
                //std::cout<<trainingLogThetaTable->getNumberOfColumns()<<std::endl;
                //std::cout<<trainingLogpTable->getNumberOfRows()<<std::endl;
                //std::cout<<trainingLogpTable->getNumberOfColumns()<<std::endl;
        
                trainingLogpTable->getBlockOfRows(0, $c_num_classes, readOnly, block1);
                trainingLogThetaTable->getBlockOfRows(0, $c_num_classes, readOnly, block2);
                double* out_arr1 = block1.getBlockPtr();
                double* out_arr2 = block2.getBlockPtr();
            
            //std::cout<<"output ";
            //for(int i=0; i<coeff_size*$c_col_size_labels; i++)
            //{
            //    std::cout<<" "<<out_arr[i];
            //}
            //std::cout<<std::endl;
            
                int64_t res_dims[] = {$c_num_classes,$c_num_classes+1};
                double* out_data = new double[$c_num_classes*($c_num_classes+1)];
                memcpy(out_data, block1.getBlockPtr(), $c_num_classes*sizeof(double));
                memcpy(out_data+$c_num_classes, block2.getBlockPtr(), $c_num_classes*$c_num_classes*sizeof(double));
                j2c_array<double> naive_bayes_out(out_data,2,res_dims);
                $c_coeff_out = naive_bayes_out;
            }
            """
    end
    return s
end

function pattern_match_call_naive_bayes(f::ANY, coeff_out::ANY, arr::ANY, arr2::ANY, numClass::Any, 
          start::ANY, count::ANY, cols::ANY, rows::ANY, start2::ANY, count2::ANY, cols2::ANY, rows2::ANY, linfo)
    return ""
end
