// double *dummySlice, *dummyModel, *dummyData;
// dummySlice = new double[host_nVel];
// dummyModel = new double[host_nModelExt];
// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];


	// cudaMemcpy(dummyModel, dev_modelBornExt[iGpu], host_nModelExt*sizeof(double), cudaMemcpyDeviceToHost);
	// std::cout << "Dummy model min #0 = " << *std::min_element(dummyModel,dummyModel+host_nModelExt) << std::endl;
	// std::cout << "Dummy model max #0 = " << *std::max_element(dummyModel,dummyModel+host_nModelExt) << std::endl;

// Debug Laplacian
void laplacianFwd_3d(double *modelDebug, double *dataDebug, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Model
	cuda_call(cudaMalloc((void**) &dev_modelDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_modelDebug[iGpu], modelDebug, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
	cuda_call(cudaMalloc((void**) &dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemset(dev_dataDebug[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));

	// std::cout << "Inside Laplacian cuda forward 1" << std::endl;

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Apply forward Laplacian
	kernel_exec(LaplacianFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_modelDebug[iGpu], dev_dataDebug[iGpu], dev_vel2Dtw2[iGpu]));

	// std::cout << "Inside Laplacian cuda forward 2" << std::endl;

	// Copy data back to host
	cuda_call(cudaMemcpy(dataDebug, dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToHost));

}

void laplacianAdj_3d(double *modelDebug, double *dataDebug, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Model
	cuda_call(cudaMalloc((void**) &dev_modelDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));

	// Data
	cuda_call(cudaMalloc((void**) &dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_dataDebug[iGpu], dataDebug, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

	// Initialize model to zero
	cuda_call(cudaMemset(dev_modelDebug[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Apply adjoint Laplacian
	kernel_exec(LaplacianAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_modelDebug[iGpu], dev_dataDebug[iGpu],dev_vel2Dtw2[iGpu]));
	// kernel_exec(LaplacianFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_dataDebug[iGpu], dev_modelDebug[iGpu], dev_vel2Dtw2[iGpu]));

	// Copy data back to host
	cuda_call(cudaMemcpy(modelDebug, dev_modelDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToHost));

}

void freeSurfaceDebugFwd(double *model, double *data, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);


	cuda_call(cudaMemcpy(dev_p0[iGpu], model, host_nModel*sizeof(double), cudaMemcpyHostToDevice));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	kernel_exec(setFreeSurfaceConditionFwdGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface>>>(dev_p0[iGpu]));
	kernel_exec(derivFwdVelGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_vel2Dtw2[iGpu]));
	// kernel_exec(derivFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));
	// kernel_exec(derivFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));
	cuda_call(cudaMemcpy(data, dev_p1[iGpu],host_nModel*sizeof(double), cudaMemcpyDeviceToHost));

}

void freeSurfaceDebugAdj(double *model, double *data, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);
	cudaStreamCreate(&compStreamTop[iGpu]);
	cudaStreamCreate(&compStreamBody[iGpu]);
	cuda_call(cudaMemcpy(dev_p0[iGpu], data, host_nModel*sizeof(double), cudaMemcpyHostToDevice));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));


	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimGridTop(1, nblocky);
	dim3 dimGridBody(nblockx-1, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// kernel_exec(setFreeSurfaceToZero<<<dimGridFreeSurface, dimBlockFreeSurface>>>(dev_p0[iGpu]));
	// derivTopAdjGpu_3D<<<dimGridTop, dimBlock, 0, compStreamTop[iGpu]>>>(dev_p1[iGpu], dev_p0[iGpu]);
	// derivBodyAdjGpu_3D<<<dimGridBody, dimBlock, 0, compStreamBody[iGpu]>>>(dev_p1[iGpu], dev_p0[iGpu]);
	kernel_exec(derivAdjVelGpu_3D<<<dimGrid, dimBlock>>>(dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));
	// kernel_exec(derivFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));
	// kernel_exec(setFreeSurfaceConditionAdjGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface>>>(dev_p1[iGpu]));

	cuda_call(cudaMemcpy(model, dev_p1[iGpu],host_nModel*sizeof(double), cudaMemcpyDeviceToHost));

}


/******************************************************************************/
/******************************* Debug Laplacian ******************************/
/******************************************************************************/
//
__global__ void LaplacianFwdGpu_3D(double *dev_model, double *dev_data, double *dev_vel2Dtw2) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current wavefield y-slice block

    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Coordinate of current thread on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Coordinate of current thread on the x-axis

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1];

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction into dev_c_y (Remember: each thread has its own version of this array)
    dev_c_y[1] = dev_model[iGlobalTemp]; // iy = 0
    dev_c_y[2] = dev_model[iGlobalTemp+=yStride]; // iy = 1
    dev_c_y[3] = dev_model[iGlobalTemp+=yStride]; // iy = 2
    shared_c[ixLocal][izLocal] = dev_model[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
    dev_c_y[5] = dev_model[iGlobalTemp+=yStride]; // iy = 4
    dev_c_y[6] = dev_model[iGlobalTemp+=yStride]; // iy = 5
    dev_c_y[7] = dev_model[iGlobalTemp+=yStride];// iy = 6
    dev_c_y[8] = dev_model[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 7

    // Loop over y
    for (long long iy=FAT; iy<dev_ny-FAT; iy++){

        // Update values along the y-axis
        dev_c_y[0] = dev_c_y[1];
        dev_c_y[1] = dev_c_y[2];
        dev_c_y[2] = dev_c_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block
        shared_c[ixLocal][izLocal] = dev_c_y[5]; // Store the middle one in the shared memory (it will be re-used to compute the Laplacian in the z- and x-directions)
        dev_c_y[5] = dev_c_y[6];
        dev_c_y[6] = dev_c_y[7];
        dev_c_y[7] = dev_c_y[8];
        dev_c_y[8] = dev_model[iGlobalTemp+=yStride]; // The last point of the stencil now points to the next y-slice

        // Remark on assignments just above:
        // iyTemp = iy + FAT
        // This guy points to the iy with the largest y-index needed to compute the Laplacian at the new y-position

        // Load the halos in the x-direction
        // Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
        if (threadIdx.y < FAT) {
			shared_c[threadIdx.y][izLocal] = dev_model[iGlobal-dev_nz*FAT]; // Left side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_model[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
    	}

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
    		shared_c[ixLocal][threadIdx.x] = dev_model[iGlobal-FAT]; // Up
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_model[iGlobal+BLOCK_SIZE_Z]; // Down
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads(); // Synchronise all threads within each block
    	// For a given block, we have now loaded the entire "block slice" plus the halos on both directions into the shared memory
    	// We can now compute the Laplacian value at each point of the entire block slice

        // Apply forward stepping operator
        dev_data[iGlobal] = dev_vel2Dtw2[iGlobal] * (

            dev_coeff[C0] * shared_c[ixLocal][izLocal]

            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] )
            + dev_coeff[CY1] * ( dev_c_y[3] + dev_c_y[5] )

            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] )
            + dev_coeff[CY2] * ( dev_c_y[2] + dev_c_y[6] )

            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] )
            + dev_coeff[CY3] * ( dev_c_y[1] + dev_c_y[7] )

            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] )
            + dev_coeff[CY4] * ( dev_c_y[0] + dev_c_y[8] )

        );

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
}

/* Forward stepper (no damping) */
__global__ void LaplacianAdjGpu_3D(double *dev_model, double *dev_data, double *dev_vel2Dtw2) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current wavefield y-slice block
	__shared__ double shared_vel[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT]; // Scaled velocity y-slice block

    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Coordinate of current thread on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Coordinate of current thread on the x-axis

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1];
	double dev_vel_y[2*FAT+1];

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction into dev_c_y (Remember: each thread has its own version of this array)
    dev_c_y[1] = dev_data[iGlobalTemp]; // iy = 0
	dev_vel_y[1] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[2] = dev_data[iGlobalTemp+=yStride]; // iy = 1
	dev_vel_y[2] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[3] = dev_data[iGlobalTemp+=yStride]; // iy = 2
	dev_vel_y[3] = dev_vel2Dtw2[iGlobalTemp];

    shared_c[ixLocal][izLocal] = dev_data[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	shared_vel[ixLocal][izLocal] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[5] = dev_data[iGlobalTemp+=yStride]; // iy = 4
	dev_vel_y[5] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[6] = dev_data[iGlobalTemp+=yStride]; // iy = 5
	dev_vel_y[6] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[7] = dev_data[iGlobalTemp+=yStride];// iy = 6
	dev_vel_y[7] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[8] = dev_data[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 7
	dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp];

    // Loop over y
    for (long long iy=FAT; iy<dev_ny-FAT; iy++){

        // Update temporary arrays with current wavefield values along the y-axis
        dev_c_y[0] = dev_c_y[1];
		dev_vel_y[0] = dev_vel_y[1];
        dev_c_y[1] = dev_c_y[2];
		dev_vel_y[1] = dev_vel_y[2];
        dev_c_y[2] = dev_c_y[3];
		dev_vel_y[2] = dev_vel_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		dev_vel_y[3] = shared_vel[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block
        shared_c[ixLocal][izLocal] = dev_c_y[5];
		shared_vel[ixLocal][izLocal] = dev_vel_y[5]; // Load central points to shared memory (for both current slice and scaled velocity)
        dev_c_y[5] = dev_c_y[6];
		dev_vel_y[5] = dev_vel_y[6];
        dev_c_y[6] = dev_c_y[7];
		dev_vel_y[6] = dev_vel_y[7];
        dev_c_y[7] = dev_c_y[8];
		dev_vel_y[7] = dev_vel_y[8];
        dev_c_y[8] = dev_data[iGlobalTemp+=yStride];
		dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp];

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
            // Top halo
    		shared_c[ixLocal][izLocal-FAT] = dev_data[iGlobal-FAT];
            shared_vel[ixLocal][izLocal-FAT] = dev_vel2Dtw2[iGlobal-FAT];
            // Bottom halo
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_data[iGlobal+BLOCK_SIZE_Z];
    		shared_vel[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_vel2Dtw2[iGlobal+BLOCK_SIZE_Z];
    	}
        // Load the halos in the x-direction
        if (threadIdx.y < FAT) {
            // Left side
    		shared_c[ixLocal-FAT][izLocal] = dev_data[iGlobal-dev_nz*FAT];
            shared_vel[ixLocal-FAT][izLocal] = dev_vel2Dtw2[iGlobal-dev_nz*FAT];
            // Right side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_data[iGlobal+dev_nz*BLOCK_SIZE_X];
    		shared_vel[ixLocal+BLOCK_SIZE_X][izLocal] = dev_vel2Dtw2[iGlobal+dev_nz*BLOCK_SIZE_X];
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads();

        // Apply adjoint stepping operator
        dev_model[iGlobal] = (

            dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
            + dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
            + dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
            + dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])
        );

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;
    }
}

/******************************************************************************/
/******************************** Imaging kernels *****************************/
/******************************************************************************/
// Forward non-extended
__global__ void imagingFwdGpu_3D_zLoop(double *dev_model, double *dev_data, double *dev_sourceWavefieldDts) {

	int iyGlobal = FAT + blockIdx.y * BLOCK_SIZE_Y + threadIdx.y; // Coordinate on y-axis

	// Make sure you are inside FAT and ny-FAT (non-included)
	if (iyGlobal < dev_ny-FAT){

		int ixGlobal = FAT + blockIdx.x * BLOCK_SIZE_X + threadIdx.x; // Coordinate on x-axis
		int iGlobal = iyGlobal * dev_yStride + ixGlobal * dev_nz; // Global coordinate on the time slice
		// Loop over z-axis
		for (int iz=FAT; iz<dev_nz-FAT; iz++){
			dev_data[iGlobal] = dev_model[iGlobal] * dev_sourceWavefieldDts[iGlobal];
			iGlobal=iGlobal+1;
		}
	}
}

__global__ void imagingFwdGpu_3D_yLoop(double *dev_model, double *dev_data, double *dev_sourceWavefieldDts) {

	// Global coordinates for the faster two axes (z and x)
	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
    int yStride = dev_nz * dev_nx;
    int iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal; // Global position on the cube

    for (int iy=FAT; iy<dev_ny-FAT; iy++){
		dev_data[iGlobal] = dev_model[iGlobal] * dev_sourceWavefieldDts[iGlobal];
		iGlobal+=yStride;
	}
}



	// sum1 = 0.0;
	// sum2 = 0.0;
	// sum3 = 0.0;
	//
	// for (int iy=0; iy<_fdParam_3D->_ny; iy++){
	// 	for (int ix=0; ix<_fdParam_3D->_nx; ix++){
	// 		for (int iz=0; iz<5; iz++){
	// 			sum1 += (*p0->_mat)[iy][ix][iz];
	// 			sum2 += (*p1->_mat)[iy][ix][iz];
	// 			sum3 += (*p2->_mat)[iy][ix][iz];
	// 		}
	// 	}
	// }
	// std::cout << "sum1 forward: " << sum1 << std::endl;
	// std::cout << "sum2 forward: " << sum2 << std::endl;
	// std::cout << "sum3 forward: " << sum3 << std::endl;

	// int index;
	// for (int iy=0; iy<_fdParam_3D->_ny;iy++){
	// 	for (int ix=0; ix<_fdParam_3D->_nx;ix++){
	// 		index = iy * _fdParam_3D->_nx * _fdParam_3D->_nz + ix * _fdParam_3D->_nz + 4;
	// 		std::cout << "iy = " << iy << std::endl;
	// 		std::cout << "ix = " << ix << std::endl;
	// 		for (int its=0; its<_fdParam_3D->_nts; its++){
	// 			std::cout << "nts = " << its << std::endl;
	// 			std::cout << "Data at free surface = " << (*dataRegDts->_mat)[0][its] << std::endl;
	// 		}
	// 	}
	// }

// _freeSurfaceDebugOpObj = std::make_shared<freeSurfaceDebugOp>(_fdParam_3D->_vel, _fdParam_3D->_par, _nGpu, _iGpu, _iGpuId, _iGpuAlloc);

	// std::shared_ptr<double3DReg> modelTest(new double3DReg(_fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny));
	// std::shared_ptr<double3DReg> dataTest(new double3DReg(_fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny));
	//
	// _freeSurfaceDebugOpObj(_fdParam_3D->_vel, _fdParam_3D->_par, _nGpu, _iGpu, _iGpuId, _iGpuAlloc);

	// _freeSurfaceDebugOpObj->dotTest(true);
	// _freeSurfaceDebugOpObj->dotTest(true);
	// _freeSurfaceDebugOpObj->dotTest(true);
	// exit(0);

/******************************************************************************/
/******************************* Debug Laplacian ******************************/
/******************************************************************************/
__global__ void LaplacianFwdGpu_3D(double *dev_model, double *dev_data, double *dev_vel2Dtw2) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current wavefield y-slice block

    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Coordinate of current thread on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Coordinate of current thread on the x-axis

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1];

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction into dev_c_y (Remember: each thread has its own version of this array)
    dev_c_y[1] = dev_model[iGlobalTemp]; // iy = 0
    dev_c_y[2] = dev_model[iGlobalTemp+=yStride]; // iy = 1
    dev_c_y[3] = dev_model[iGlobalTemp+=yStride]; // iy = 2
    shared_c[ixLocal][izLocal] = dev_model[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
    dev_c_y[5] = dev_model[iGlobalTemp+=yStride]; // iy = 4
    dev_c_y[6] = dev_model[iGlobalTemp+=yStride]; // iy = 5
    dev_c_y[7] = dev_model[iGlobalTemp+=yStride];// iy = 6
    dev_c_y[8] = dev_model[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 7

    // Loop over y
    for (long long iy=FAT; iy<dev_ny-FAT; iy++){

        // Update values along the y-axis
        dev_c_y[0] = dev_c_y[1];
        dev_c_y[1] = dev_c_y[2];
        dev_c_y[2] = dev_c_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block
        shared_c[ixLocal][izLocal] = dev_c_y[5]; // Store the middle one in the shared memory (it will be re-used to compute the Laplacian in the z- and x-directions)
        dev_c_y[5] = dev_c_y[6];
        dev_c_y[6] = dev_c_y[7];
        dev_c_y[7] = dev_c_y[8];
        dev_c_y[8] = dev_model[iGlobalTemp+=yStride]; // The last point of the stencil now points to the next y-slice

        // Remark on assignments just above:
        // iyTemp = iy + FAT
        // This guy points to the iy with the largest y-index needed to compute the Laplacian at the new y-position

        // Load the halos in the x-direction
        // Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
        if (threadIdx.y < FAT) {
			shared_c[threadIdx.y][izLocal] = dev_model[iGlobal-dev_nz*FAT]; // Left side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_model[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
    	}

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
    		shared_c[ixLocal][threadIdx.x] = dev_model[iGlobal-FAT]; // Up
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_model[iGlobal+BLOCK_SIZE_Z]; // Down
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads(); // Synchronise all threads within each block
    	// For a given block, we have now loaded the entire "block slice" plus the halos on both directions into the shared memory
    	// We can now compute the Laplacian value at each point of the entire block slice

        // Apply forward stepping operator
        dev_data[iGlobal] = dev_vel2Dtw2[iGlobal] * (

            dev_coeff[C0] * shared_c[ixLocal][izLocal]

            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] )
            + dev_coeff[CY1] * ( dev_c_y[3] + dev_c_y[5] )

            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] )
            + dev_coeff[CY2] * ( dev_c_y[2] + dev_c_y[6] )

            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] )
            + dev_coeff[CY3] * ( dev_c_y[1] + dev_c_y[7] )

            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] )
            + dev_coeff[CY4] * ( dev_c_y[0] + dev_c_y[8] )

        );

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
}

__global__ void LaplacianAdjGpu_3D(double *dev_model, double *dev_data, double *dev_vel2Dtw2) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current wavefield y-slice block
	__shared__ double shared_vel[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT]; // Scaled velocity y-slice block

    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Coordinate of current thread on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Coordinate of current thread on the x-axis

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1];
	double dev_vel_y[2*FAT+1];

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction into dev_c_y (Remember: each thread has its own version of this array)
    dev_c_y[1] = dev_data[iGlobalTemp]; // iy = 0
	dev_vel_y[1] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[2] = dev_data[iGlobalTemp+=yStride]; // iy = 1
	dev_vel_y[2] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[3] = dev_data[iGlobalTemp+=yStride]; // iy = 2
	dev_vel_y[3] = dev_vel2Dtw2[iGlobalTemp];

    shared_c[ixLocal][izLocal] = dev_data[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	shared_vel[ixLocal][izLocal] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[5] = dev_data[iGlobalTemp+=yStride]; // iy = 4
	dev_vel_y[5] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[6] = dev_data[iGlobalTemp+=yStride]; // iy = 5
	dev_vel_y[6] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[7] = dev_data[iGlobalTemp+=yStride];// iy = 6
	dev_vel_y[7] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[8] = dev_data[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 7
	dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp];

    // Loop over y
    for (long long iy=FAT; iy<dev_ny-FAT; iy++){

        // Update temporary arrays with current wavefield values along the y-axis
        dev_c_y[0] = dev_c_y[1];
		dev_vel_y[0] = dev_vel_y[1];
        dev_c_y[1] = dev_c_y[2];
		dev_vel_y[1] = dev_vel_y[2];
        dev_c_y[2] = dev_c_y[3];
		dev_vel_y[2] = dev_vel_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		dev_vel_y[3] = shared_vel[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block
        shared_c[ixLocal][izLocal] = dev_c_y[5];
		shared_vel[ixLocal][izLocal] = dev_vel_y[5]; // Load central points to shared memory (for both current slice and scaled velocity)
        dev_c_y[5] = dev_c_y[6];
		dev_vel_y[5] = dev_vel_y[6];
        dev_c_y[6] = dev_c_y[7];
		dev_vel_y[6] = dev_vel_y[7];
        dev_c_y[7] = dev_c_y[8];
		dev_vel_y[7] = dev_vel_y[8];
        dev_c_y[8] = dev_data[iGlobalTemp+=yStride];
		dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp];

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
            // Top halo
    		shared_c[ixLocal][izLocal-FAT] = dev_data[iGlobal-FAT];
            shared_vel[ixLocal][izLocal-FAT] = dev_vel2Dtw2[iGlobal-FAT];
            // Bottom halo
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_data[iGlobal+BLOCK_SIZE_Z];
    		shared_vel[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_vel2Dtw2[iGlobal+BLOCK_SIZE_Z];
    	}
        // Load the halos in the x-direction
        if (threadIdx.y < FAT) {
            // Left side
    		shared_c[ixLocal-FAT][izLocal] = dev_data[iGlobal-dev_nz*FAT];
            shared_vel[ixLocal-FAT][izLocal] = dev_vel2Dtw2[iGlobal-dev_nz*FAT];
            // Right side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_data[iGlobal+dev_nz*BLOCK_SIZE_X];
    		shared_vel[ixLocal+BLOCK_SIZE_X][izLocal] = dev_vel2Dtw2[iGlobal+dev_nz*BLOCK_SIZE_X];
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads();

        // Apply adjoint stepping operator
        dev_model[iGlobal] = (

            dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
            + dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
            + dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
            + dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])
        );

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;
    }
}

__global__ void LaplacianAdjTopGpu_3D(double *dev_model, double *dev_data, double *dev_vel2Dtw2) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT]; // Current wavefield y-slice block
	__shared__ double shared_vel[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT]; // Scaled velocity y-slice block

    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1]; // Array for the current wavefield y-slice
    double dev_vel_y[2*FAT+1]; // Array for the scaled velocity y-slice

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction (Remember: each thread has its own version of dev_c_y and dev_vel_y array)
    // Points from the current wavefield time-slice that will be used by the current block
    dev_c_y[1] = dev_data[iGlobalTemp];								dev_vel_y[1] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[2] = dev_data[iGlobalTemp+=yStride];						dev_vel_y[2] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[3] = dev_data[iGlobalTemp+=yStride];						dev_vel_y[3] = dev_vel2Dtw2[iGlobalTemp];
	// These ones go to shared memory because used multiple times in Laplacian computation for the z- and x-directions
    shared_c[ixLocal][izLocal] = dev_data[iGlobalTemp+=yStride]; 		shared_vel[ixLocal][izLocal] = dev_vel2Dtw2[iGlobalTemp];
	dev_c_y[5] = dev_data[iGlobalTemp+=yStride];						dev_vel_y[5] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[6] = dev_data[iGlobalTemp+=yStride];						dev_vel_y[6] = dev_vel2Dtw2[iGlobalTemp];
	dev_c_y[7] = dev_data[iGlobalTemp+=yStride];						dev_vel_y[7] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[8] = dev_data[iGlobalTemp+=yStride];						dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp];

    // Loop over y
    for (long long iyGlobal=FAT; iyGlobal<dev_ny-FAT; iyGlobal++){

        // Update temporary arrays with current wavefield values along the y-axis
        dev_c_y[0] = dev_c_y[1];						dev_vel_y[0] = dev_vel_y[1];
        dev_c_y[1] = dev_c_y[2];						dev_vel_y[1] = dev_vel_y[2];
        dev_c_y[2] = dev_c_y[3];						dev_vel_y[2] = dev_vel_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];		dev_vel_y[3] = shared_vel[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
        shared_c[ixLocal][izLocal] = dev_c_y[5]; 		shared_vel[ixLocal][izLocal] = dev_vel_y[5]; // Load central points to shared memory (for both current slice and scaled velocity)
        dev_c_y[5] = dev_c_y[6];						dev_vel_y[5] = dev_vel_y[6];
        dev_c_y[6] = dev_c_y[7];						dev_vel_y[6] = dev_vel_y[7];
        dev_c_y[7] = dev_c_y[8];						dev_vel_y[7] = dev_vel_y[8];
        dev_c_y[8] = dev_data[iGlobalTemp+=yStride];		dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp];

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
            // Top halo
    		shared_c[ixLocal][izLocal-FAT] = dev_data[iGlobal-FAT];
            shared_vel[ixLocal][izLocal-FAT] = dev_vel2Dtw2[iGlobal-FAT];
            // Bottom halo
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_data[iGlobal+BLOCK_SIZE_Z];
    		shared_vel[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_vel2Dtw2[iGlobal+BLOCK_SIZE_Z];
    	}
        // Load the halos in the x-direction
        if (threadIdx.y < FAT) {
            // Left side
    		shared_c[ixLocal-FAT][izLocal] = dev_data[iGlobal-dev_nz*FAT];
            shared_vel[ixLocal-FAT][izLocal] = dev_vel2Dtw2[iGlobal-dev_nz*FAT];
            // Right side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_data[iGlobal+dev_nz*BLOCK_SIZE_X];
    		shared_vel[ixLocal+BLOCK_SIZE_X][izLocal] = dev_vel2Dtw2[iGlobal+dev_nz*BLOCK_SIZE_X];
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads();

        // Apply adjoint stepping operator
		if (izGlobal <= 4){
			dev_model[iGlobal] = 0.0;
		}
		if (izGlobal == 5){
	        dev_model[iGlobal] = (

	            dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
	            + dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

				- dev_coeff[CZ2] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]
				- dev_coeff[CZ3] * shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1]
				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2]

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
	            + dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])

	        );

	        // Move forward one grid point in the y-direction
	        iGlobal = iGlobal + yStride;
		}

		if (izGlobal == 6){

			dev_model[iGlobal] = (

				dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

				+ dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
				+ dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
				+ dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

				- dev_coeff[CZ3] * shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1]
				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

				+ dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
				+ dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
				+ dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

				+ dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
				+ dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
				+ dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

				+ dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
				+ dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
				+ dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])

			);

		}
		if (izGlobal == 7){

			dev_model[iGlobal] = (

				dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

				+ dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
				+ dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
				+ dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2]

				+ dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
				+ dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
				+ dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

				+ dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
				+ dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
				+ dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

				+ dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
				+ dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
				+ dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])

			);

		}
		if (izGlobal > 7){

			dev_model[iGlobal] = (

				dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

				+ dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
				+ dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
				+ dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

				+ dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
				+ dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
				+ dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

				+ dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
				+ dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
				+ dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

				+ dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
				+ dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
				+ dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])

			);
		}

    }
}

__global__ void setFreeSurfaceToZero(double *dev_c) {

	// Global coordinates for the slowest axis
	long long iyGlobal = FAT + blockIdx.y * BLOCK_SIZE_Y + threadIdx.y; // Global y-coordinate
	long long ixGlobal = FAT + blockIdx.x * BLOCK_SIZE_X + threadIdx.x; // Global x-coordinate
	long long iGlobal = iyGlobal * dev_yStride + ixGlobal * dev_nz;

	if (iyGlobal < dev_ny-FAT){
		dev_c[iGlobal+FAT] = 0.0;
	}
}

__global__ void setFreeSurfaceConditionAdjGpu_3D(double *dev_c) {

	// Global coordinates for the slowest axis
	long long iyGlobal = FAT + blockIdx.y * BLOCK_SIZE_Y + threadIdx.y; // Global y-coordinate
	long long ixGlobal = FAT + blockIdx.x * BLOCK_SIZE_X + threadIdx.x; // Global x-coordinate
	long long iGlobal = iyGlobal * dev_yStride + ixGlobal * dev_nz;

	if (iyGlobal < dev_ny-FAT){

		dev_c[iGlobal+FAT] = 0.0;
		dev_c[iGlobal+2*FAT] = dev_c[iGlobal+2*FAT] - dev_c[iGlobal];
		dev_c[iGlobal+2*FAT-1] = dev_c[iGlobal+2*FAT-1] - dev_c[iGlobal+1];
		dev_c[iGlobal+2*FAT-2] = dev_c[iGlobal+2*FAT-2] - dev_c[iGlobal+2];
		dev_c[iGlobal+2*FAT-3] = dev_c[iGlobal+2*FAT-3] - dev_c[iGlobal+3];
		dev_c[iGlobal] = 0.0;
		dev_c[iGlobal+1] = 0.0;
		dev_c[iGlobal+2] = 0.0;
		dev_c[iGlobal+3] = 0.0;

	}
}

__global__ void subtractNewDebug_3D(double *dev_o, double *dev_n) {

	// Global coordinates for the slowest axis
	long long iyGlobal = FAT + blockIdx.y * BLOCK_SIZE_Y + threadIdx.y; // Global y-coordinate
	long long ixGlobal = FAT + blockIdx.x * BLOCK_SIZE_X + threadIdx.x; // Global x-coordinate
	long long iGlobal = iyGlobal * dev_yStride + ixGlobal * dev_nz;

	if (iyGlobal < dev_ny-FAT){
		for (int iz=0; iz<dev_nz; iz++){
			dev_o[iGlobal+iz] -= dev_n[iGlobal+iz];
		}
	}
}

__global__ void addCurAndStep(double *dev_o, double *dev_c, double *dev_n) {

	// Global coordinates for the slowest axis
	long long iyGlobal = FAT + blockIdx.y * BLOCK_SIZE_Y + threadIdx.y; // Global y-coordinate
	long long ixGlobal = FAT + blockIdx.x * BLOCK_SIZE_X + threadIdx.x; // Global x-coordinate
	long long iGlobal = iyGlobal * dev_yStride + ixGlobal * dev_nz;

	if (iyGlobal < dev_ny-FAT){
		for (int iz=FAT; iz<dev_nz-FAT; iz++){
			dev_n[iGlobal+iz] += 2 * dev_c[iGlobal+iz] - dev_o[iGlobal+iz];
		}
	}
}

__global__ void stepAdjGpuDebug_3D(double *dev_o, double *dev_c, double *dev_n, double *dev_vel2Dtw2) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT]; // Current wavefield y-slice block
	__shared__ double shared_vel[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT]; // Scaled velocity y-slice block

    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1]; // Array for the current wavefield y-slice
    double dev_vel_y[2*FAT+1]; // Array for the scaled velocity y-slice

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction (Remember: each thread has its own version of dev_c_y and dev_vel_y array)
    // Points from the current wavefield time-slice that will be used by the current block
    dev_c_y[1] = dev_c[iGlobalTemp];								dev_vel_y[1] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[2] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[2] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[3] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[3] = dev_vel2Dtw2[iGlobalTemp];
	// These ones go to shared memory because used multiple times in Laplacian computation for the z- and x-directions
    shared_c[ixLocal][izLocal] = dev_c[iGlobalTemp+=yStride]; 		shared_vel[ixLocal][izLocal] = dev_vel2Dtw2[iGlobalTemp];
	dev_c_y[5] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[5] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[6] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[6] = dev_vel2Dtw2[iGlobalTemp];
	dev_c_y[7] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[7] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[8] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp];

    // Loop over y
    for (long long iyGlobal=FAT; iyGlobal<dev_ny-FAT; iyGlobal++){

        // Update temporary arrays with current wavefield values along the y-axis
        dev_c_y[0] = dev_c_y[1];						dev_vel_y[0] = dev_vel_y[1];
        dev_c_y[1] = dev_c_y[2];						dev_vel_y[1] = dev_vel_y[2];
        dev_c_y[2] = dev_c_y[3];						dev_vel_y[2] = dev_vel_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];		dev_vel_y[3] = shared_vel[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
        shared_c[ixLocal][izLocal] = dev_c_y[5]; 		shared_vel[ixLocal][izLocal] = dev_vel_y[5]; // Load central points to shared memory (for both current slice and scaled velocity)
        dev_c_y[5] = dev_c_y[6];						dev_vel_y[5] = dev_vel_y[6];
        dev_c_y[6] = dev_c_y[7];						dev_vel_y[6] = dev_vel_y[7];
        dev_c_y[7] = dev_c_y[8];						dev_vel_y[7] = dev_vel_y[8];
        dev_c_y[8] = dev_c[iGlobalTemp+=yStride];		dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp];

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
            // Top halo
    		shared_c[ixLocal][izLocal-FAT] = dev_c[iGlobal-FAT];
            shared_vel[ixLocal][izLocal-FAT] = dev_vel2Dtw2[iGlobal-FAT];
            // Bottom halo
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c[iGlobal+BLOCK_SIZE_Z];
    		shared_vel[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_vel2Dtw2[iGlobal+BLOCK_SIZE_Z];
    	}
        // Load the halos in the x-direction
        if (threadIdx.y < FAT) {
            // Left side
    		shared_c[ixLocal-FAT][izLocal] = dev_c[iGlobal-dev_nz*FAT];
            shared_vel[ixLocal-FAT][izLocal] = dev_vel2Dtw2[iGlobal-dev_nz*FAT];
            // Right side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c[iGlobal+dev_nz*BLOCK_SIZE_X];
    		shared_vel[ixLocal+BLOCK_SIZE_X][izLocal] = dev_vel2Dtw2[iGlobal+dev_nz*BLOCK_SIZE_X];
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads();

        // Apply adjoint stepping operator
        dev_o[iGlobal] = (

            dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
            + dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
            + dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
            + dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])

        ) + 2.0 * shared_c[ixLocal][izLocal];

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
}

//
__global__ void derivFwdGpu_3D(double *dev_model, double *dev_data) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current wavefield y-slice block

    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Coordinate of current thread on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Coordinate of current thread on the x-axis

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1];

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction into dev_c_y (Remember: each thread has its own version of this array)
    dev_c_y[1] = dev_model[iGlobalTemp]; // iy = 0
    dev_c_y[2] = dev_model[iGlobalTemp+=yStride]; // iy = 1
    dev_c_y[3] = dev_model[iGlobalTemp+=yStride]; // iy = 2
    shared_c[ixLocal][izLocal] = dev_model[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
    dev_c_y[5] = dev_model[iGlobalTemp+=yStride]; // iy = 4
    dev_c_y[6] = dev_model[iGlobalTemp+=yStride]; // iy = 5
    dev_c_y[7] = dev_model[iGlobalTemp+=yStride];// iy = 6
    dev_c_y[8] = dev_model[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 7

    // Loop over y
    for (long long iy=FAT; iy<dev_ny-FAT; iy++){

        // Update values along the y-axis
        dev_c_y[0] = dev_c_y[1];
        dev_c_y[1] = dev_c_y[2];
        dev_c_y[2] = dev_c_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block
        shared_c[ixLocal][izLocal] = dev_c_y[5]; // Store the middle one in the shared memory (it will be re-used to compute the Laplacian in the z- and x-directions)
        dev_c_y[5] = dev_c_y[6];
        dev_c_y[6] = dev_c_y[7];
        dev_c_y[7] = dev_c_y[8];
        dev_c_y[8] = dev_model[iGlobalTemp+=yStride]; // The last point of the stencil now points to the next y-slice

        // Remark on assignments just above:
        // iyTemp = iy + FAT
        // This guy points to the iy with the largest y-index needed to compute the Laplacian at the new y-position

        // Load the halos in the x-direction
        // Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
        if (threadIdx.y < FAT) {
			shared_c[threadIdx.y][izLocal] = dev_model[iGlobal-dev_nz*FAT]; // Left side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_model[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
    	}

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
    		shared_c[ixLocal][threadIdx.x] = dev_model[iGlobal-FAT]; // Up
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_model[iGlobal+BLOCK_SIZE_Z]; // Down
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads(); // Synchronise all threads within each block
    	// For a given block, we have now loaded the entire "block slice" plus the halos on both directions into the shared memory
    	// We can now compute the Laplacian value at each point of the entire block slice

        // Apply forward stepping operator
        dev_data[iGlobal] = (

            dev_coeff[C0] * shared_c[ixLocal][izLocal]

            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] )
            + dev_coeff[CY1] * ( dev_c_y[3] + dev_c_y[5] )

            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] )
            + dev_coeff[CY2] * ( dev_c_y[2] + dev_c_y[6] )

            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] )
            + dev_coeff[CY3] * ( dev_c_y[1] + dev_c_y[7] )

            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] )
            + dev_coeff[CY4] * ( dev_c_y[0] + dev_c_y[8] )

        );

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
}

__global__ void derivTopAdjGpu_3D(double *dev_model, double *dev_data) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT]; // Current wavefield y-slice block


    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1]; // Array for the current wavefield y-slice

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction (Remember: each thread has its own version of dev_c_y and dev_vel_y array)
    // Points from the current wavefield time-slice that will be used by the current block
    dev_c_y[1] = dev_data[iGlobalTemp];
    dev_c_y[2] = dev_data[iGlobalTemp+=yStride];
    dev_c_y[3] = dev_data[iGlobalTemp+=yStride];
	// These ones go to shared memory because used multiple times in Laplacian computation for the z- and x-directions
    shared_c[ixLocal][izLocal] = dev_data[iGlobalTemp+=yStride];
	dev_c_y[5] = dev_data[iGlobalTemp+=yStride];
    dev_c_y[6] = dev_data[iGlobalTemp+=yStride];
	dev_c_y[7] = dev_data[iGlobalTemp+=yStride];
    dev_c_y[8] = dev_data[iGlobalTemp+=yStride];

    // Loop over y
    for (long long iyGlobal=FAT; iyGlobal<dev_ny-FAT; iyGlobal++){

        // Update temporary arrays with current wavefield values along the y-axis
        dev_c_y[0] = dev_c_y[1];
        dev_c_y[1] = dev_c_y[2];
        dev_c_y[2] = dev_c_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
        shared_c[ixLocal][izLocal] = dev_c_y[5];
        dev_c_y[5] = dev_c_y[6];
        dev_c_y[6] = dev_c_y[7];
        dev_c_y[7] = dev_c_y[8];
        dev_c_y[8] = dev_data[iGlobalTemp+=yStride];

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
            // Top halo
    		shared_c[ixLocal][izLocal-FAT] = dev_data[iGlobal-FAT];
            // Bottom halo
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_data[iGlobal+BLOCK_SIZE_Z];

    	}
        // Load the halos in the x-direction
        if (threadIdx.y < FAT) {
            // Left side
    		shared_c[ixLocal-FAT][izLocal] = dev_data[iGlobal-dev_nz*FAT];
            // Right side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_data[iGlobal+dev_nz*BLOCK_SIZE_X];

    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads();

		if (izGlobal == 4){
			// dev_model[iGlobal] = 0.0;
			// iGlobal = iGlobal + yStride;
		}
		if (izGlobal == 5){
	        dev_model[iGlobal] = (

	            dev_coeff[C0] * shared_c[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] )
	            + dev_coeff[CY1] * ( dev_c_y[3] + dev_c_y[5] )

				- dev_coeff[CZ2] * shared_c[ixLocal][izLocal]
				- dev_coeff[CZ3] * shared_c[ixLocal][izLocal+1]
				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal+2]

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] )
	            + dev_coeff[CY2] * ( dev_c_y[2] + dev_c_y[6] )

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] + dev_c_y[7] )

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] + dev_c_y[8] )

	        );

	        // Move forward one grid point in the y-direction
	        // iGlobal = iGlobal + yStride;
		}

		if (izGlobal == 6){

			dev_model[iGlobal] = (

	            dev_coeff[C0] * shared_c[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] )
	            + dev_coeff[CY1] * ( dev_c_y[3] + dev_c_y[5])

				- dev_coeff[CZ3] * shared_c[ixLocal][izLocal-1]
				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal]

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] )
	            + dev_coeff[CY2] * ( dev_c_y[2] + dev_c_y[6])

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] + dev_c_y[7] )

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] + dev_c_y[8] )

	        );

			// Move forward one grid point in the y-direction
	        // iGlobal = iGlobal + yStride;

		}
		if (izGlobal == 7){

			dev_model[iGlobal] = (

	            dev_coeff[C0] * shared_c[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] )
	            + dev_coeff[CY1] * ( dev_c_y[3] + dev_c_y[5])

				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal-2]

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] )
	            + dev_coeff[CY2] * ( dev_c_y[2] + dev_c_y[6])

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] + dev_c_y[7] )

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] + dev_c_y[8] )

	        );
			// Move forward one grid point in the y-direction
	        // iGlobal = iGlobal + yStride;
		}

		if (izGlobal > 7){

			dev_model[iGlobal] = (

	            dev_coeff[C0] * shared_c[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] )
	            + dev_coeff[CY1] * ( dev_c_y[3] + dev_c_y[5])

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] )
	            + dev_coeff[CY2] * ( dev_c_y[2] + dev_c_y[6])

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] + dev_c_y[7] )

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] + dev_c_y[8] )

	        );
			// iGlobal = iGlobal + yStride;
		}
		iGlobal = iGlobal + yStride;
    }
}

__global__ void derivBodyAdjGpu_3D(double *dev_model, double *dev_data) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT]; // Current wavefield y-slice block


    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + (blockIdx.x + 1) * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1]; // Array for the current wavefield y-slice

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction (Remember: each thread has its own version of dev_c_y and dev_vel_y array)
    // Points from the current wavefield time-slice that will be used by the current block
    dev_c_y[1] = dev_data[iGlobalTemp];
    dev_c_y[2] = dev_data[iGlobalTemp+=yStride];
    dev_c_y[3] = dev_data[iGlobalTemp+=yStride];
	// These ones go to shared memory because used multiple times in Laplacian computation for the z- and x-directions
    shared_c[ixLocal][izLocal] = dev_data[iGlobalTemp+=yStride];
	dev_c_y[5] = dev_data[iGlobalTemp+=yStride];
    dev_c_y[6] = dev_data[iGlobalTemp+=yStride];
	dev_c_y[7] = dev_data[iGlobalTemp+=yStride];
    dev_c_y[8] = dev_data[iGlobalTemp+=yStride];

    // Loop over y
    for (long long iyGlobal=FAT; iyGlobal<dev_ny-FAT; iyGlobal++){

        // Update temporary arrays with current wavefield values along the y-axis
        dev_c_y[0] = dev_c_y[1];
        dev_c_y[1] = dev_c_y[2];
        dev_c_y[2] = dev_c_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
        shared_c[ixLocal][izLocal] = dev_c_y[5];
        dev_c_y[5] = dev_c_y[6];
        dev_c_y[6] = dev_c_y[7];
        dev_c_y[7] = dev_c_y[8];
        dev_c_y[8] = dev_data[iGlobalTemp+=yStride];

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
            // Top halo
    		shared_c[ixLocal][izLocal-FAT] = dev_data[iGlobal-FAT];
            // Bottom halo
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_data[iGlobal+BLOCK_SIZE_Z];

    	}
        // Load the halos in the x-direction
        if (threadIdx.y < FAT) {
            // Left side
    		shared_c[ixLocal-FAT][izLocal] = dev_data[iGlobal-dev_nz*FAT];
            // Right side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_data[iGlobal+dev_nz*BLOCK_SIZE_X];

    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads();

		// if (izGlobal > 7){

			dev_model[iGlobal] = (

	            dev_coeff[C0] * shared_c[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] )
	            + dev_coeff[CY1] * ( dev_c_y[3] + dev_c_y[5])

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] )
	            + dev_coeff[CY2] * ( dev_c_y[2] + dev_c_y[6])

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] + dev_c_y[7] )

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] + dev_c_y[8] )

	        );
			// iGlobal = iGlobal + yStride;
		// }
		iGlobal = iGlobal + yStride;
    }
}

__global__ void derivFwdVelGpu_3D(double *dev_model, double *dev_data, double *dev_vel2Dtw2) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current wavefield y-slice block

    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Coordinate of current thread on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Coordinate of current thread on the x-axis

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1];

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction into dev_c_y (Remember: each thread has its own version of this array)
    dev_c_y[1] = dev_model[iGlobalTemp]; // iy = 0
    dev_c_y[2] = dev_model[iGlobalTemp+=yStride]; // iy = 1
    dev_c_y[3] = dev_model[iGlobalTemp+=yStride]; // iy = 2
    shared_c[ixLocal][izLocal] = dev_model[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
    dev_c_y[5] = dev_model[iGlobalTemp+=yStride]; // iy = 4
    dev_c_y[6] = dev_model[iGlobalTemp+=yStride]; // iy = 5
    dev_c_y[7] = dev_model[iGlobalTemp+=yStride];// iy = 6
    dev_c_y[8] = dev_model[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 7

    // Loop over y
    for (long long iy=FAT; iy<dev_ny-FAT; iy++){

        // Update values along the y-axis
        dev_c_y[0] = dev_c_y[1];
        dev_c_y[1] = dev_c_y[2];
        dev_c_y[2] = dev_c_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block
        shared_c[ixLocal][izLocal] = dev_c_y[5]; // Store the middle one in the shared memory (it will be re-used to compute the Laplacian in the z- and x-directions)
        dev_c_y[5] = dev_c_y[6];
        dev_c_y[6] = dev_c_y[7];
        dev_c_y[7] = dev_c_y[8];
        dev_c_y[8] = dev_model[iGlobalTemp+=yStride]; // The last point of the stencil now points to the next y-slice

        // Remark on assignments just above:
        // iyTemp = iy + FAT
        // This guy points to the iy with the largest y-index needed to compute the Laplacian at the new y-position

        // Load the halos in the x-direction
        // Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
        if (threadIdx.y < FAT) {
			shared_c[threadIdx.y][izLocal] = dev_model[iGlobal-dev_nz*FAT]; // Left side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_model[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
    	}

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
    		shared_c[ixLocal][threadIdx.x] = dev_model[iGlobal-FAT]; // Up
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_model[iGlobal+BLOCK_SIZE_Z]; // Down
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads(); // Synchronise all threads within each block
    	// For a given block, we have now loaded the entire "block slice" plus the halos on both directions into the shared memory
    	// We can now compute the Laplacian value at each point of the entire block slice

        // Apply forward stepping operator
        dev_data[iGlobal] = dev_vel2Dtw2[iGlobal] * (

            dev_coeff[C0] * shared_c[ixLocal][izLocal]

            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] )
            + dev_coeff[CY1] * ( dev_c_y[3] + dev_c_y[5] )

            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] )
            + dev_coeff[CY2] * ( dev_c_y[2] + dev_c_y[6] )

            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] )
            + dev_coeff[CY3] * ( dev_c_y[1] + dev_c_y[7] )

            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] )
            + dev_coeff[CY4] * ( dev_c_y[0] + dev_c_y[8] )

        ) + 2.0 * shared_c[ixLocal][izLocal];

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
}

__global__ void derivAdjVelGpu_3D(double *dev_model, double *dev_data, double *dev_vel2Dtw2) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT]; // Current wavefield y-slice block
	__shared__ double shared_vel[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT]; // Scaled velocity y-slice block

    // Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate

    // Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1]; // Array for the current wavefield y-slice
    double dev_vel_y[2*FAT+1]; // Array for the scaled velocity y-slice

    // Number of elements in one y-slice
    long long yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

	// Load the values along the y-direction into dev_c_y (Remember: each thread has its own version of this array)
    dev_c_y[1] = dev_data[iGlobalTemp]; // iy = 0
	dev_vel_y[1] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[2] = dev_data[iGlobalTemp+=yStride]; // iy = 1
	dev_vel_y[2] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[3] = dev_data[iGlobalTemp+=yStride]; // iy = 2
	dev_vel_y[3] = dev_vel2Dtw2[iGlobalTemp];

    shared_c[ixLocal][izLocal] = dev_data[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	shared_vel[ixLocal][izLocal] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[5] = dev_data[iGlobalTemp+=yStride]; // iy = 4
	dev_vel_y[5] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[6] = dev_data[iGlobalTemp+=yStride]; // iy = 5
	dev_vel_y[6] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[7] = dev_data[iGlobalTemp+=yStride];// iy = 6
	dev_vel_y[7] = dev_vel2Dtw2[iGlobalTemp];

    dev_c_y[8] = dev_data[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 7
	dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp];

    // Loop over y
    for (long long iyGlobal=FAT; iyGlobal<dev_ny-FAT; iyGlobal++){

        // Update temporary arrays with current wavefield values along the y-axis
		dev_c_y[0] = dev_c_y[1];
		dev_vel_y[0] = dev_vel_y[1];
        dev_c_y[1] = dev_c_y[2];
		dev_vel_y[1] = dev_vel_y[2];
        dev_c_y[2] = dev_c_y[3];
		dev_vel_y[2] = dev_vel_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		dev_vel_y[3] = shared_vel[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block
        shared_c[ixLocal][izLocal] = dev_c_y[5];
		shared_vel[ixLocal][izLocal] = dev_vel_y[5]; // Load central points to shared memory (for both current slice and scaled velocity)
        dev_c_y[5] = dev_c_y[6];
		dev_vel_y[5] = dev_vel_y[6];
        dev_c_y[6] = dev_c_y[7];
		dev_vel_y[6] = dev_vel_y[7];
        dev_c_y[7] = dev_c_y[8];
		dev_vel_y[7] = dev_vel_y[8];
        dev_c_y[8] = dev_data[iGlobalTemp+=yStride];
		dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp];

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
			// Top halo
    		shared_c[ixLocal][izLocal-FAT] = dev_data[iGlobal-FAT];
            shared_vel[ixLocal][izLocal-FAT] = dev_vel2Dtw2[iGlobal-FAT];
            // Bottom halo
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_data[iGlobal+BLOCK_SIZE_Z];
    		shared_vel[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_vel2Dtw2[iGlobal+BLOCK_SIZE_Z];

    	}
        // Load the halos in the x-direction
        if (threadIdx.y < FAT) {
			// Left side
    		shared_c[ixLocal-FAT][izLocal] = dev_data[iGlobal-dev_nz*FAT];
            shared_vel[ixLocal-FAT][izLocal] = dev_vel2Dtw2[iGlobal-dev_nz*FAT];
            // Right side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_data[iGlobal+dev_nz*BLOCK_SIZE_X];
    		shared_vel[ixLocal+BLOCK_SIZE_X][izLocal] = dev_vel2Dtw2[iGlobal+dev_nz*BLOCK_SIZE_X];
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads();

		if (izGlobal == 4){
			dev_model[iGlobal] = 0.0;
		}
		if (izGlobal == 5){

			dev_model[iGlobal] = (

	            dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
	            + dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

				- dev_coeff[CZ2] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]
				- dev_coeff[CZ3] * shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1]
				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2]

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
	            + dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])

	        ) + 2.0 * shared_c[ixLocal][izLocal];

		}

		if (izGlobal == 6){

			dev_model[iGlobal] = (

	            dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
	            + dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

				- dev_coeff[CZ3] * shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1]
				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
	            + dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])

	        ) + 2.0 * shared_c[ixLocal][izLocal];

		}
		if (izGlobal == 7){

			dev_model[iGlobal] = (

				dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
	            + dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2]

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
	            + dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])

	        ) + 2.0 * shared_c[ixLocal][izLocal];
		}

		if (izGlobal > 7){

			dev_model[iGlobal] = (

	            dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
	            + dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[5] * dev_vel_y[5])

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
	            + dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[6] * dev_vel_y[6])

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[7] * dev_vel_y[7])

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])

	        ) + 2.0 * shared_c[ixLocal][izLocal];
		}
		iGlobal = iGlobal + yStride;
    }
}


		// Create two arrays (one on device, other one host)
		// int *host_array, *dev_array;
		// host_array = new int[12];
		// cuda_call(cudaMalloc((void**) &dev_array, 12*sizeof(int)));
		// for (int j=0; j<12; j++){
		// 	host_array[j]=j;
		// }
		// cuda_call(cudaMemcpy(dev_array, host_array, 12*sizeof(int), cudaMemcpyHostToDevice));
		//
		// // My thaang
		// for (int j=0; j<2; j++){
		//
		// 	kA1<<<1, 1, 0, compStream[iGpu]>>>(j, dev_array);
		//
		// 	for (int i=2; i<12; i++){
		// 		kA2<<<1, 1, 0, topStream[iGpu]>>>(i, j, dev_array);
		// 	}
		// 	cudaEventRecord(test, NULL);
		// 	cudaStreamWaitEvent(compStream[iGpu], test, 0);
		// 	kA3<<<1, 1, 0, compStream[iGpu]>>>(j, dev_array);
		// 	std::cout << "Done j = " << j << std::endl;
		// }


	//
	//
	// // We assume the source wavelet/signals already contain the second time derivative
	// // Set device number
	// // cudaSetDevice(iGpuId);
	// //
	// // // Create streams
	// // cudaStreamCreate(&compStream[iGpu]);
	// // cudaStreamCreate(&transferStream[iGpu]);
	// //
	// // // Sources geometry
	// // cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	// // cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	// // cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));
	// //
	// // // Sources geometry + signals
  	// // cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	// // cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device
	// //
	// // // Receivers geometry
	// // cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	// // cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	// // cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));
	// //
	// // // Initialize time-slices for time-stepping
  	// // cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	// // cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	// //
	// // // Initialize time-slices for transfer to host's pinned memory
  	// // cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
	// //
	// // // Initialize pinned memory
	// // cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));
	// //
	// // // Blocks for Laplacian
	// // int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	// // int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	// // dim3 dimGrid(nblockx, nblocky);
	// // dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	// //
	// // // Blocks data recording
	// // int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	// //
	// // /********************** Source wavefield computation **********************/
	// // for (int its = 0; its < host_nts-1; its++){
	// //
	// // 	// Loop within two values of its (coarse time grid)
	// // 	for (int it2 = 1; it2 < host_sub+1; it2++){
	// //
	// // 		// Compute fine time-step index
	// // 		int itw = its * host_sub + it2;
	// //
	// // 		// Step forward
	// // 		stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
	// //
	// // 		// Inject source
	// // 		injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);
	// //
	// // 		// Damp wavefields
	// // 		dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
	// //
	// // 		// Spread energy to dev_pLeft and dev_pRight
	// // 		interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
	// //
	// // 		// Extract and interpolate data
	// // 		// kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));
	// //
	// // 		// Switch pointers
	// // 		dev_temp1[iGpu] = dev_p0[iGpu];
	// // 		dev_p0[iGpu] = dev_p1[iGpu];
	// // 		dev_p1[iGpu] = dev_temp1[iGpu];
	// // 		dev_temp1[iGpu] = NULL;
	// //
	// // 	}
	// //
	// // 	/* Note: At that point pLeft [its] is ready to be transfered back to host */
	// // 	// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
	// // 	cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
	// // 	// At that point, the value of pStream has been transfered back to host pinned memory
	// //
	// // 	// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
	// // 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	// // 	// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]
	// //
	// // 	if (its>0) {
	// // 		// Standard library
	// // 		std::memcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
	// //
	// // 		// Using HostToHost
	// // 		// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	// //
	// // 	}
	// // 	// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
	// // 	// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
	// // 	cuda_call(cudaStreamSynchronize(compStream[iGpu]));
	// // 	// Asynchronous transfer of pStream => pin [its] [transfer]
	// // 	// Launch the transfer while we compute the next coarse time sample
	// // 	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));
	// //
	// // 	// Switch pointers
	// // 	dev_pTemp[iGpu] = dev_pLeft[iGpu];
	// // 	dev_pLeft[iGpu] = dev_pRight[iGpu];
	// // 	dev_pRight[iGpu] = dev_pTemp[iGpu];
	// // 	dev_pTemp[iGpu] = NULL;
  	// // 	cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	// // }
	// //
	// // // At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// // // The CPU has stored the wavefield values ranging from 0,...,nts-3
	// //
	// // // Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	// // cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
	// //
	// // // Load pLeft to pStream (value of wavefield at nts-1)
	// // cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	// //
	// // // In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	// // std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
	// // // cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-2)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	// //
	// // // Wait until pLeft -> pStream is done
	// // cuda_call(cudaStreamSynchronize(compStream[iGpu]));
	// //
	// // // At this point, pStream contains the value of the wavefield at nts-1
	// // // Transfer pStream -> pinned
	// // cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
	// //
	// // // Copy pinned -> RAM
	// // std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel,pin_wavefieldSlice[iGpu], host_nModel*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]
	// // // cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-1)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	// //
	// // /************************ Adjoint wavefield computation *******************/
	// //
	// // cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	// // cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	// // cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
	// // cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
  	// // cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
	// // cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));
	// //
	// // // Set model to zero
	// // cuda_call(cudaMemset(dev_modelBorn[iGpu], 0, host_nModel*sizeof(double)));
	// //
	// // // Allocate and copy data from host -> device
  	// // cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	// // cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device
	// //
	// // // Start propagating scattered wavefield
	// // for (int its = host_nts-2; its > -1; its--){
	// //
	// // 	// Load source wavefield for its+1
	// // 	std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+1)*host_nModel, host_nModel*sizeof(double));
	// // 	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice));
	// // 	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice));
	// //
	// // 	for (int it2 = host_sub-1; it2 > -1; it2--){
	// //
	// // 		// Step adjoint
	// // 		stepAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
	// //
	// // 		// Inject data
	// // 		interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);
	// //
	// // 		// Damp wavefields
	// // 		dampCosineEdge_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]);
	// //
	// // 		// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
	// // 		// pLeft corresponds to its, pRight corresponds to its+1
	// // 		interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
	// //
	// // 		// Switch pointers
	// // 		dev_temp1[iGpu] = dev_p0[iGpu];
	// // 		dev_p0[iGpu] = dev_p1[iGpu];
	// // 		dev_p1[iGpu] = dev_temp1[iGpu];
	// // 		dev_temp1[iGpu] = NULL;
	// //
	// // 	}
	// // 	// At that point, the receiver wavefield for its+1 is done (stored in pRight)
	// // 	// Apply imaging condition for index its+1
	// // 	imagingAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
	// //
	// // 	// Switch pointers for secondary source
	// // 	dev_pTemp[iGpu] = dev_pRight[iGpu];
	// // 	dev_pRight[iGpu] = dev_pLeft[iGpu];
	// // 	dev_pLeft[iGpu] = dev_pTemp[iGpu];
	// // 	dev_pTemp[iGpu] = NULL;
	// // 	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
	// //
	// // }
	// //
	// // // Load source wavefield for 0
	// // std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel*sizeof(double));
	// // cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice));
	// // cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice));
	// //
	// // // Finished main loop - we still have to compute imaging condition for its = 0
  	// // imagingAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
	// //
	// // // Scale model for finite-difference and secondary source coefficient
	// // kernel_exec(scaleReflectivity_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));
	// //
	// // // Copy model back to host
	// // cuda_call(cudaMemcpy(model, dev_modelBorn[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
	//
	//
	// // Reset the time slices to zero
	// // cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	// // cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	// // cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
	// // cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
  	// // cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
	// // cuda_call(cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double)));
	// //
	// // // Set model to zero
	// // cuda_call(cudaMemset(dev_modelBorn[iGpu], 0, host_nModel*sizeof(double)));
	// //
	// // // Allocate and copy data from host -> device
  	// // cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	// // cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device
	// //
	// // /************************ Streaming stuff starts **************************/
	// //
	// // // Load time-slice nts-1 from the source wavefield
	// // // From RAM -> pinned -> dev_pStream -> dev_pSourceWavefield
	// // // std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(host_nts-1)*host_nModel, host_nModel*sizeof(double));
	// // // cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(nts-1)*host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	// // // cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// // // cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
	// // // cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
	// //
	// // // At that point:
	// // // dev_pSourceWavefield contains wavefield at nts-1
	// // // pin_wavefieldSlice and dev_pStream are free to be used
	// //
	// // // Start propagating scattered wavefield
	// // for (int its = host_nts-2; its > -1; its--){
	// //
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+1)*host_nModel, host_nModel*sizeof(double));
		// // cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, compStream[iGpu]));
		// cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice));
		// // cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice));
	// //
	// // 	// Launch transfer from RAM -> pinned for time slice its
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+its*host_nModel, host_nModel*sizeof(double));
	// //
	// // 	// Wait until pStream has been transfered to pSourceWavefield
	// // 	// cuda_call(cudaStreamSynchronize(compStream[iGpu]));
	// //
	// // 	// Launch transfer from pinned -> device for time slice its
	// // 	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// //
	// // 	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, compStream[iGpu]));
	// //
	// // 	for (int it2 = host_sub-1; it2 > -1; it2--){
	// //
	// // 		// Step adjoint
	// // 		stepAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
	// //
	// // 		// Inject data
	// // 		interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);
	// //
	// // 		// Damp wavefields
	// // 		dampCosineEdge_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]);
	// //
	// // 		// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
	// // 		// pLeft corresponds to its-2, pRight corresponds to its-1
	// // 		interpFineToCoarseSlice_3D<<<dimGrid, dimBlock>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
	// //
	// // 		// Switch pointers
	// // 		dev_temp1[iGpu] = dev_p0[iGpu];
	// // 		dev_p0[iGpu] = dev_p1[iGpu];
	// // 		dev_p1[iGpu] = dev_temp1[iGpu];
	// // 		dev_temp1[iGpu] = NULL;
	// //
	// // 	}
	// // 	// At that point, the receiver wavefield for its+1 is done (stored in pRight)
	// // 	// Apply imaging condition for index its+1
	// // 	imagingAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
	// // 	// cuda_call(cudaDeviceSynchronize());
	// //
	// // 	// Wait until the new source wavefield time-slice is loaded
	// // 	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
	// // 	// At this point, the imaging condition is done for its+1
	// //
	// // 	// dev_pSourceWavefield needs to be updated to value at its
	// // 	// cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	// //
	// // 	// Switch pointers for secondary source
	// // 	dev_pTemp[iGpu] = dev_pRight[iGpu];
	// // 	dev_pRight[iGpu] = dev_pLeft[iGpu];
	// // 	dev_pLeft[iGpu] = dev_pTemp[iGpu];
	// // 	dev_pTemp[iGpu] = NULL;
	// // 	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
	// // 	// cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu]));
	// // 	// cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu]));
	// // 	// cuda_call(cudaDeviceSynchronize());
	// // }
	// //
	// //
	// // std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel*sizeof(double));
	// // // cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, compStream[iGpu]));
	// // cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice));
	// // // cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	// // cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice));
	// //
	// // // Apply imaging condition for its=0
	// // imagingAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
	// // cuda_call(cudaDeviceSynchronize());
	// // // Scale model for finite-difference and secondary source coefficient
	// // scaleReflectivity_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);
	// //
	// // // Wait until pStream(its=0) has been transfered to pSourceWavefield (its=0)
	// // // cuda_call(cudaStreamSynchronize(compStream[iGpu]));
	// //
	// // // Copy model back to host
	// // cuda_call(cudaMemcpy(model, dev_modelBorn[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
	//
	// /*************************** Memory deallocation **************************/
	// // Deallocate the array for sources/receivers' positions
    // // cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    // // cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	// // cuda_call(cudaFree(dev_dataRegDts[iGpu]));
	// //
	// // // Calls that should be moved from here when debugging is done
    // // cuda_call(cudaStreamDestroy(compStream[iGpu]));
    // // cuda_call(cudaStreamDestroy(transferStream[iGpu]));
	//
	// cudaSetDevice(iGpuId);
	//
	// // Create streams
	// cudaStreamCreate(&compStream[iGpu]);
	// cudaStreamCreate(&transferStream[iGpu]);
	//
	// // Sources geometry
	// cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	// cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	// cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));
	//
	// // Sources geometry + signals
  	// cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	// cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device
	//
	// // Receivers geometry
	// cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	// cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	// cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));
	//
	// // Initialize time-slices for time-stepping
  	// cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	// cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	//
	// // Initialize time-slices for transfer to host's pinned memory
  	// cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
	//
	// // Initialize pinned memory
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));
	//
	// // Blocks for Laplacian
	// int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	// int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	// dim3 dimGrid(nblockx, nblocky);
	// dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	//
	// // Blocks data recording
	// int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	//
	// /********************** Source wavefield computation **********************/
	// for (int its = 0; its < host_nts-1; its++){
	//
	// 	// Loop within two values of its (coarse time grid)
	// 	for (int it2 = 1; it2 < host_sub+1; it2++){
	//
	// 		// Compute fine time-step index
	// 		int itw = its * host_sub + it2;
	//
	// 		// Step forward
	// 		stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
	//
	// 		// Inject source
	// 		injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);
	//
	// 		// Damp wavefields
	// 		dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
	//
	// 		// Spread energy to dev_pLeft and dev_pRight
	// 		interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
	//
	// 		// Extract and interpolate data
	// 		// kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));
	//
	// 		// Switch pointers
	// 		dev_temp1[iGpu] = dev_p0[iGpu];
	// 		dev_p0[iGpu] = dev_p1[iGpu];
	// 		dev_p1[iGpu] = dev_temp1[iGpu];
	// 		dev_temp1[iGpu] = NULL;
	//
	// 	}
	//
	// 	/* Note: At that point pLeft [its] is ready to be transfered back to host */
	// 	// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
	// 	cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
	// 	// At that point, the value of pStream has been transfered back to host pinned memory
	//
	// 	// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
	// 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	// 	// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]
	//
	// 	if (its>0) {
	// 		// Standard library
	// 		std::memcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
	//
	// 		// Using HostToHost
	// 		// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	//
	// 	}
	// 	// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
	// 	// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
	// 	cuda_call(cudaStreamSynchronize(compStream[iGpu]));
	// 	// Asynchronous transfer of pStream => pin [its] [transfer]
	// 	// Launch the transfer while we compute the next coarse time sample
	// 	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));
	//
	// 	// Switch pointers
	// 	dev_pTemp[iGpu] = dev_pLeft[iGpu];
	// 	dev_pLeft[iGpu] = dev_pRight[iGpu];
	// 	dev_pRight[iGpu] = dev_pTemp[iGpu];
	// 	dev_pTemp[iGpu] = NULL;
  	// 	cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	// }
	//
	// // At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// // The CPU has stored the wavefield values ranging from 0,...,nts-3
	//
	// // Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
	//
	// // Load pLeft to pStream (value of wavefield at nts-1)
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	//
	// // In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	// std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
	// // cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-2)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	//
	// // Wait until pLeft -> pStream is done
	// cuda_call(cudaStreamSynchronize(compStream[iGpu]));
	//
	// // At this point, pStream contains the value of the wavefield at nts-1
	// // Transfer pStream -> pinned
	// cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
	//
	// // Copy pinned -> RAM
	// std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel,pin_wavefieldSlice[iGpu], host_nModel*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]
	// // cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-1)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	//
	// /************************ Adjoint wavefield computation *******************/
	//
	// // Reset the time slices to zero
	// cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	// cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	// cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
	// cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
  	// cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));
	//
	// // Set model to zero
	// cuda_call(cudaMemset(dev_modelBorn[iGpu], 0, host_nModel*sizeof(double)));
	//
	// // Allocate and copy data from host -> device
  	// cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	// cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device
	//
	// // Start propagating scattered wavefield
	// for (int its = host_nts-2; its > -1; its--){
	//
	// 	// Load source wavefield for its+1
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+1)*host_nModel, host_nModel*sizeof(double));
		// cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice));
		// cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice));
	//
	// 	for (int it2 = host_sub-1; it2 > -1; it2--){
	//
	// 		// Step adjoint
	// 		stepAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
	//
	// 		// Inject data
	// 		interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);
	//
	// 		// Damp wavefields
	// 		dampCosineEdge_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]);
	//
	// 		// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
	// 		// pLeft corresponds to its, pRight corresponds to its+1
	// 		interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
	//
	// 		// Switch pointers
	// 		dev_temp1[iGpu] = dev_p0[iGpu];
	// 		dev_p0[iGpu] = dev_p1[iGpu];
	// 		dev_p1[iGpu] = dev_temp1[iGpu];
	// 		dev_temp1[iGpu] = NULL;
	//
	// 	}
	// 	// At that point, the receiver wavefield for its+1 is done (stored in pRight)
	// 	// Apply imaging condition for index its+1
	// 	imagingAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
	//
	// 	// Switch pointers for secondary source
	// 	dev_pTemp[iGpu] = dev_pRight[iGpu];
	// 	dev_pRight[iGpu] = dev_pLeft[iGpu];
	// 	dev_pLeft[iGpu] = dev_pTemp[iGpu];
	// 	dev_pTemp[iGpu] = NULL;
	// 	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
	//
	// }
	//
	// // Load source wavefield for 0
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel*sizeof(double));
	// cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice));
	// cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice));
	//
	// // Finished main loop - we still have to compute imaging condition for its = 0
  	// imagingAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
	//
	// // Scale model for finite-difference and secondary source coefficient
	// kernel_exec(scaleReflectivity_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));
	//
	// // Copy model back to host
	// cuda_call(cudaMemcpy(model, dev_modelBorn[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
	//
	// /*************************** Memory deallocation **************************/
	// // Deallocate the array for sources/receivers' positions
    // cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    // cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	// cuda_call(cudaFree(dev_dataRegDts[iGpu]));
	//
    // cuda_call(cudaStreamDestroy(compStream[iGpu]));
    // cuda_call(cudaStreamDestroy(transferStream[iGpu]));

/******************************************************************************/
/****************************** Benchmark *************************************/
/******************************************************************************/
// Benchmark for imaging condition
void imagingFwd_zLoop(double *model, double *data, int iGpu, int iGpuId){

	// Set device number
	cudaSetDevice(iGpuId);

	// Allocate and copy model to dev_modelDebug
	cuda_call(cudaMalloc((void**) &dev_modelDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_modelDebug[iGpu], model, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate, copy data to dev_dataDebug and initialize it
	cuda_call(cudaMalloc((void**) &dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemset(dev_dataDebug[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));

	// Allocate slice for testing imaging kernel
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_p1[iGpu], model, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocks for wavefield extraction
	int nblockyModel = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockzModel = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGridModel(nblockyModel, nblockzModel);
	dim3 dimBlockModel(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	float ms;
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for (int iRep=0; iRep<10000; iRep++){
		kernel_exec(imagingFwdGpu_3D_zLoop<<<dimGridModel, dimBlockModel>>>(dev_modelDebug[iGpu], dev_dataDebug[iGpu], dev_p1[iGpu]));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "duration for z-loop: " << ms/1000 << " [s]" << std::endl;

	// Copy data back to host
	cuda_call(cudaMemcpy(data, dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToHost));
	std::cout << "Done zloop" << std::endl;
}

void imagingFwd_yLoop(double *model, double *data, int iGpu, int iGpuId){

	// Set device number
	cudaSetDevice(iGpuId);

	// Allocate and copy model to dev_modelDebug
	cuda_call(cudaMalloc((void**) &dev_modelDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_modelDebug[iGpu], model, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate, copy data to dev_dataDebug and initialize it
	cuda_call(cudaMalloc((void**) &dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemset(dev_dataDebug[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));

	// Allocate slice for testing imaging kernel
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_p1[iGpu], model, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocks for wavefield extraction
	int nblockyModel = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockzModel = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGridModel(nblockyModel, nblockzModel);
	dim3 dimBlockModel(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	float ms;
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Timer
	// std::clock_t start;
	// double duration;
	// start = std::clock();

	for (int iRep=0; iRep<10000; iRep++){
		kernel_exec(imagingFwdGpu_3D_yLoop<<<dimGrid, dimBlock>>>(dev_modelDebug[iGpu], dev_dataDebug[iGpu], dev_p1[iGpu]));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "duration for y-loop: " << ms/1000 << " [s]" << std::endl;

	// duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// duration*=1000;


	// Copy data back to host
	cuda_call(cudaMemcpy(data, dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToHost));
	std::cout << "Done yloop" << std::endl;
}

// void computeTomoSrcWfldDt2_3D(double *dev_sourcesIn, double *wavefield1, double *wavefield2, long long *dev_sourcesPositionsRegIn, int nSourcesRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn){
//
// 	// Initialize time-slices for time-stepping
//   	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pWavefieldSliceDt2[iGpu], 0, host_nVel*sizeof(double)));
//
// 	// Initialize time-slices for transfer to host's pinned memory
//   	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));
//
// 	// Initialize pinned memory
// 	cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nVel*sizeof(double));
//
// 	double *dummySliceLeft, *dummySliceRight;
// 	dummySliceLeft = new double[host_nVel];
// 	dummySliceRight = new double[host_nVel];
//
// 	// Compute coarse source wavefield sample at its = 0
// 	int its = 0;
//
// 	// Loop within two values of its (coarse time grid)
// 	for (int it2 = 1; it2 < host_sub+1; it2++){
//
// 		// Compute fine time-step index
// 		int itw = its * host_sub + it2;
//
// 		// Step forward
// 		stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
//
// 		// Inject source
// 		injectSourceLinear_3D<<<1, nSourcesRegIn, 0, compStreamIn>>>(dev_sourcesIn, dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);
//
// 		// Damp wavefields
// 		dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);
//
// 		// Spread energy to dev_pLeft and dev_pRight
// 		interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
//
// 		// Switch pointers
// 		dev_temp1[iGpu] = dev_p0[iGpu];
// 		dev_p0[iGpu] = dev_p1[iGpu];
// 		dev_p1[iGpu] = dev_temp1[iGpu];
// 		dev_temp1[iGpu] = NULL;
// 	}
//
// 	cudaMemcpy(dummySliceLeft, dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
// 	std::cout << "Min pLeft at its = " << its << ", = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nVel) << std::endl;
// 	std::cout << "Max pLeft at its = " << its << ", = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nVel) << std::endl;
//
// 	cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
// 	std::cout << "Min pRight at its = " << its << ", = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
// 	std::cout << "Max pRight at its = " << its << ", = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
//
//
	// Copy pDt1 (its = 0)
	// cuda_call(cudaMemcpy(dev_pDt1[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
//
// 	cuda_call(cudaStreamSynchronize(compStreamIn));
// 	cuda_call(cudaMemcpy(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
// 	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));
// 	std::memcpy(wavefield1+0*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
//
// 	// Switch coarse grid pointers
// 	dev_pTemp[iGpu] = dev_pLeft[iGpu];
// 	dev_pLeft[iGpu] = dev_pRight[iGpu];
// 	dev_pRight[iGpu] = dev_pTemp[iGpu];
// 	dev_pTemp[iGpu] = NULL;
//
// 	/************************** Main loop (its > 0) ***************************/
// 	for (int its = 1; its < host_nts-1; its++){
//
// 		// Loop within two values of its (coarse time grid)
// 		for (int it2 = 1; it2 < host_sub+1; it2++){
//
// 			// Compute fine time-step index
// 			int itw = its * host_sub + it2;
//
// 			// Step forward
// 			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
//
// 			// Inject source
// 			injectSourceLinear_3D<<<1, nSourcesRegIn, 0, compStreamIn>>>(dev_sourcesIn, dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);
//
// 			// Damp wavefields
// 			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);
//
// 			// Spread energy to dev_pLeft and dev_pRight
// 			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
//
// 			// Switch pointers
// 			dev_temp1[iGpu] = dev_p0[iGpu];
// 			dev_p0[iGpu] = dev_p1[iGpu];
// 			dev_p1[iGpu] = dev_temp1[iGpu];
// 			dev_temp1[iGpu] = NULL;
//
// 		}
//
// 		cudaMemcpy(dummySliceLeft, dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
// 		std::cout << "Min pLeft at its = " << its << ", = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nVel) << std::endl;
// 		std::cout << "Max pLeft at its = " << its << ", = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nVel) << std::endl;
//
// 		cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
// 		std::cout << "Min pRight at its = " << its << ", = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
// 		std::cout << "Max pRight at its = " << its << ", = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
//
//
// 		// std::cout << "its = " << its << ", step #2" << std::endl;
// 		// Copy pLeft (value of source wavefield at its) into pDt2
// 		// cuda_call(cudaMemcpyAsync(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));
//
// 		cuda_call(cudaStreamSynchronize(compStreamIn));
// 		cuda_call(cudaMemcpy(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
//
// 		// Compute second-order time-derivative of source wavefield at its-1
// 	    // srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pWavefieldSliceDt2[iGpu], dev_pDt0[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu]);
//
// 		// cuda_call(cudaStreamSynchronize(transferStreamIn)); // Blocks host until all issued cuda calls in transfer stream are completed
// 		// cuda_call(cudaMemcpy(dev_pStream[iGpu], dev_pWavefieldSliceDt2[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
//
// 		if (its > 1){
// 			// Copy from pinned -> RAM
// 			// std::memcpy(wavefield1+(its-2)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
// 			std::memcpy(wavefield1+(its-1)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
// 		}
// 		// std::cout << "its = " << its << ", step #7" << std::endl;
// 		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
// 		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
// 		// cuda_call(cudaStreamSynchronize(compStreamIn));
// 		// Asynchronous transfer of pStream => pin [its] [transfer]
// 		// Launch the transfer while we compute the next coarse time sample
// 		// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStreamIn));
//
// 		cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));
//
// 		// Switch pointers
// 		dev_pTemp[iGpu] = dev_pLeft[iGpu];
// 		dev_pLeft[iGpu] = dev_pRight[iGpu];
// 		dev_pRight[iGpu] = dev_pTemp[iGpu];
// 		dev_pTemp[iGpu] = NULL;
//   		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStreamIn)); // Reinitialize dev_pRight to zero (because of the += in the kernel)
//
// 		// Switch pointers for time derivative
// 		dev_pDtTemp[iGpu] = dev_pDt0[iGpu];
// 		dev_pDt0[iGpu] = dev_pDt1[iGpu];
// 		dev_pDt1[iGpu] = dev_pDt2[iGpu];
// 		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
// 		dev_pDtTemp[iGpu] = NULL;
//
// 	}
//
// 	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
// 	// The CPU has stored the wavefield values ranging from 0,...,nts-4
//
// 	// Copy pLeft (value of source wavefield at nts-1) into pDt2
// 	// cuda_call(cudaMemcpyAsync(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));
//
// 	// cuda_call(cudaMemcpy(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
// 	// cuda_call(cudaMemcpy(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
//
// 	cuda_call(cudaMemcpy(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
//
// 	std::memcpy(wavefield1+(host_nts-2)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
//
// 	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
//
// 	// Copy pinned -> RAM
// 	std::memcpy(wavefield1+(host_nts-1)*host_nModel,pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
//
// 	// Compute second-order time-derivative of source wavefield at nts-2
// 	// srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pWavefieldSliceDt2[iGpu], dev_pDt0[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu]);
//
// 	// Wait until pStream (which contains the second time derivative of the source wavefield at nts-3) has transfered value to pinned memory
// 	// cuda_call(cudaStreamSynchronize(transferStreamIn));
//
// 	// Load pWavefield into pStream (value of second time derivative of source wavefield at nts-2)
// 	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pWavefieldSliceDt2[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));
//
// 	// cuda_call(cudaMemcpy(dev_pStream[iGpu], dev_pWavefieldSliceDt2[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
//
// 	// In the meantime, copy value of wavefield at nts-3 from pinned memory to RAM
// 	// std::memcpy(wavefield1+(host_nts-3)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
// 	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-2)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
//
// 	// Wait until pWavefieldSlice -> pStream is done
// 	// cuda_call(cudaStreamSynchronize(compStreamIn));
//
// 	// Copy value of second-order time-derivative of source wavefield at nts-2
// 	// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStreamIn));
//
// 	// cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));
// 	//
// 	// // Compute second-order time-derivative of source wavefield at nts-1
// 	// // cuda_call(cudaMemsetAsync(dev_pDt0[iGpu], 0, host_nVel*sizeof(double), compStreamIn)); // Set pDt0 to
// 	// cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(double))); // Set pDt0 to zero
// 	// srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pWavefieldSliceDt2[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu], dev_pDt0[iGpu]);
// 	//
// 	// // Wait for pStream to by fully copied to pin
// 	// // cuda_call(cudaStreamSynchronize(transferStreamIn));
// 	//
// 	// // Copy second time derivative of wavefield into pStream at nts-1
// 	// // cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pWavefieldSliceDt2[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));
// 	//
// 	// cuda_call(cudaMemcpy(dev_pStream[iGpu], dev_pWavefieldSliceDt2[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
// 	//
// 	// // Copy second time derivative of wavefield at nts-2 from pin -> RAM
// 	// std::memcpy(wavefield1+(host_nts-2)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double)); // Copy second-order time-derivative of source wavefield from pin -> RAM for nts-2
// 	//
// 	// // Wait until the copy into pStream is done
// 	// // cuda_call(cudaStreamSynchronize(compStreamIn));
// 	//
// 	// // Copy to pin
// 	// // cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStreamIn));
// 	//
// 	// cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));
// 	//
// 	// // Copy second time derivative of wavefield at nts-1 from pin -> RAM
// 	// std::memcpy(wavefield1+(host_nts-1)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double)); // Copy second-order time-derivative of source wavefield from pin -> RAM for nts-2
//
// }
