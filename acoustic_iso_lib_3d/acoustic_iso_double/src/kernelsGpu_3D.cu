/******************************************************************************/
/*********************************** Injection ********************************/
/******************************************************************************/
/* Inject source: no need for a "if" statement because the number of threads = nb devices */
__global__ void injectSourceLinear_3D(float *dev_signalIn, float *dev_timeSlice, int itw, int *dev_sourcesPositionReg){
	int iThread = blockIdx.x * blockDim.x + threadIdx.x;
	dev_timeSlice[dev_sourcesPositionReg[iThread]] += dev_signalIn[iThread * dev_ntw + itw]; // Time is the fast axis
}

/* Interpolate and inject data */
__global__ void interpLinearInjectData_3D(float *dev_signalIn, float *dev_timeSlice, int its, int it2, int *dev_receiversPositionReg) {
	int iThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (iThread < dev_nReceiversReg) {
		dev_timeSlice[dev_receiversPositionReg[iThread]] += dev_signalIn[dev_nts*iThread+its] * dev_interpFilter[it2+1] + dev_signalIn[dev_nts*iThread+its+1] * dev_interpFilter[dev_hInterpFilter+it2+1];
	}
}

/******************************************************************************/
/*********************************** Extraction *******************************/
/******************************************************************************/
/* Extract and interpolate data */
__global__ void recordLinearInterpData_3D(float *dev_newTimeSlice, float *dev_signalOut, int its, int it2, int *dev_receiversPositionReg) {

	int iThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (iThread < dev_nReceiversReg) {
		// printf("dev_receiversPositionReg[iThread] = %d \n", dev_receiversPositionReg[iThread]);
		dev_signalOut[dev_nts*iThread+its]   += dev_newTimeSlice[dev_receiversPositionReg[iThread]] * dev_interpFilter[it2];
		dev_signalOut[dev_nts*iThread+its+1] += dev_newTimeSlice[dev_receiversPositionReg[iThread]] * dev_interpFilter[dev_hInterpFilter+it2];
	}
}

/* Extract source for "nonlinear adjoint" */
__global__ void recordSource_3D(float *dev_wavefield, float *dev_signalOut, int itw, int *dev_sourcesPositionReg) {
	int iThread = blockIdx.x * blockDim.x + threadIdx.x;
	dev_signalOut[dev_ntw*iThread + itw] += dev_wavefield[dev_sourcesPositionReg[iThread]];
}

/******************************************************************************/
/******************************** Damping *************************************/
/******************************************************************************/
__global__ void dampCosineEdge_3D(double *dev_p1, double *dev_p2) {

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
    int yStride = dev_nz * dev_nx;

    for (int iyGlobal=FAT; iyGlobal<dev_ny-FAT; iyGlobal++){

    	// Compute distance to the closest edge of model
    	int distToEdge = min4(izGlobal-FAT, ixGlobal-FAT, dev_nz-izGlobal-1-FAT, dev_nx-ixGlobal-1-FAT);
        distToEdge = min2(distToEdge,min2(iyGlobal-FAT,dev_ny-iyGlobal-1-FAT));

    	if (distToEdge < dev_minPad){

            // Compute global index
	        int iGlobal = iyGlobal * yStride + dev_nz * ixGlobal + izGlobal;

    		// Compute damping coefficient
    		double damp = dev_cosDampingCoeff[distToEdge];

    		// Apply damping
    		dev_p1[iGlobal] *= damp;
    		dev_p2[iGlobal] *= damp;
    	}
    }
}

/******************************************************************************/
/******************************* Forward stepper ******************************/
/******************************************************************************/
/* Forward stepper (no damping) */
__global__ void stepFwdGpu_3D(double *dev_o, double *dev_c, double *dev_n, double *dev_vel2Dtw2) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_Z+2*FAT][BLOCK_SIZE_X+2*FAT];  // Current wavefield y-slice block

    // Global coordinates for the faster two axes (z and x)
	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate

    // Local coordinates for the fastest two axes
	int izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	int ixLocal = FAT + threadIdx.y; // z-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1];

    // Number of elements in one y-slice
    int yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    int iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iyGlobal
    int iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction (Remember: each thread has its own version of the array)
    dev_c_y[1] = dev_c_y[iGlobalTemp]; // iy = 0
    dev_c_y[2] = dev_c_y[iGlobalTemp+=yStride]; // iy = 1
    dev_c_y[3] = dev_c_y[iGlobalTemp+=yStride]; // iy = 2
    shared_c[ixLocal][izLocal] = dev_c[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
    dev_c_y[5] = dev_c_y[iGlobalTemp+=yStride]; // iy = 4
    dev_c_y[6] = dev_c_y[iGlobalTemp+=yStride]; // iy = 5
    dev_c_y[7] = dev_c_y[iGlobalTemp+=yStride];// iy = 6
    dev_c_y[8] = dev_c_y[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 7

    // Loop over y
    for (int iy=FAT; iy<dev_ny-FAT; iy++){

        // Update values along the y-axis
        dev_c_y[0] = dev_c_y[1];
        dev_c_y[1] = dev_c_y[2];
        dev_c_y[2] = dev_c_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
        shared_c[ixLocal][izLocal] = dev_c_y[5]; // Store the middle one in the shared memory (it will be re-used to compute the Laplacian in the z- and x-directions)
        dev_c_y[5] = dev_c_y[6];
        dev_c_y[6] = dev_c_y[7];
        dev_c_y[7] = dev_c_y[8];
        dev_c_y[8] = dev_c[iGlobalTemp+=yStride]

        // Remark on assignments just above:
        // iyTemp = 2*FAT + (iy-FAT)
        // This guy points to the iy with the largest y-index needed to compute the Laplacian at the new y-position

        // Load the halos in the x-direction
        // Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
        if (threadIdx.y < FAT) {
    		shared_c[ixLocal-FAT][izLocal] = dev_c[iGlobal-dev_nz*FAT]; // Left side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
    	}

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
    		shared_c[ixLocal][izLocal-FAT] = dev_c[iGlobal-FAT]; // Up
    		shared_c[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c[iGlobal+BLOCK_SIZE_Z]; // Down
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads(); // Synchronise all threads within each block
    	// For a given block, we have now loaded the entire "block slice" plus the halos on both directions into the shared memory
    	// We can now compute the Laplacian value at each point of the entire block slice

        // Apply forward stepping operator
        dev_n[iGlobal] = dev_vel2Dtw2[iGlobal] * (

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

        ) + shared_c[ixLocal][izLocal] * shared_c[ixLocal][izLocal] - dev_o[iGlobal];

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
}

/******************************************************************************/
/******************************* Adjoint stepper ******************************/
/******************************************************************************/
/* Forward stepper (no damping) */
__global__ void stepAdjGpu_3D(double *dev_o, double *dev_c, double *dev_n, double *dev_vel2Dtw2) {

    // Allocate shared memory for a specific block
	__shared__ double shared_c[BLOCK_SIZE_Z+2*FAT][BLOCK_SIZE_X+2*FAT]; // Current wavefield y-slice block
	__shared__ double shared_vel[BLOCK_SIZE_Z+2*FAT][BLOCK_SIZE_X+2*FAT]; // Scaled velocity y-slice block

    // Global coordinates for the faster two axes (z and x)
	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate

    // Local coordinates for the fastest two axes
	int izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	int ixLocal = FAT + threadIdx.y; // z-coordinate on the local grid stored in shared memory

    // Allocate (on global memory?) the array that will store the wavefield values in the y-direction
    // Each thread will have its own version of this array
    // Question: is that on the global memory? -> can it fit in the register?
    // Why do we create this temporary array and not call it directly from global memory?
    double dev_c_y[2*FAT+1]; // Array for the current wavefield y-slice
    double dev_vel_y[2*FAT+1]; // Array for the scaled velocity y-slice

    // Number of elements in one y-slice
    int yStride = dev_nz * dev_nx;

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    int iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    int iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction (Remember: each thread has its own version of dev_c_y and dev_vel_y array)

    // Points from the current wavefield time-slice that will be used by the current block
    dev_c_y[1] = dev_c_y[iGlobalTemp];
    dev_c_y[2] = dev_c_y[iGlobalTemp+=yStride];
    dev_c_y[3] = dev_c_y[iGlobalTemp+=yStride];
    shared_c[ixLocal][izLocal] = dev_c[iGlobalTemp+=yStride]; // -> this one goes to shared memory because used multiple times in Laplacian computation
    dev_c_y[5] = dev_c_y[iGlobalTemp+=yStride];
    dev_c_y[6] = dev_c_y[iGlobalTemp+=yStride];
    dev_c_y[7] = dev_c_y[iGlobalTemp+=yStride];
    dev_c_y[8] = dev_c_y[iGlobalTemp+=yStride];

    // Same shit but for the scaled velocity
    dev_vel_y[1] = dev_vel2Dtw2[iGlobalTemp];
    dev_vel_y[2] = dev_vel2Dtw2[iGlobalTemp+=yStride];
    dev_vel_y[3] = dev_vel2Dtw2[iGlobalTemp+=yStride];
    shared_vel[ixLocal][izLocal] = dev_c[dev_vel2Dtw2+=yStride];
    dev_vel_y[5] = dev_vel2Dtw2[iGlobalTemp+=yStride];
    dev_vel_y[6] = dev_vel2Dtw2[iGlobalTemp+=yStride];
    dev_vel_y[7] = dev_vel2Dtw2[iGlobalTemp+=yStride];
    dev_vel_y[8] = dev_vel2Dtw2[iGlobalTemp+=yStride];

    // Loop over y
    for (int iyGlobal=FAT; iyGlobal<dev_ny-FAT; iyGlobal++){

        // Update temporary arrays with current wavefield values along the y-axis
        dev_c_y[0] = dev_c_y[1];
        dev_c_y[1] = dev_c_y[2];
        dev_c_y[2] = dev_c_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
        shared_c[ixLocal][izLocal] = dev_c_y[5]; // Load central point to shared memory
        dev_c_y[5] = dev_c_y[6];
        dev_c_y[6] = dev_c_y[7];
        dev_c_y[7] = dev_c_y[8];
        dev_c_y[8] = dev_c[iGlobalTemp+=yStride]

        // Update temporaray arrays with scaled velocity values along the y-axis
        dev_vel_y[0] = dev_vel_y[1];
        dev_vel_y[1] = dev_vel_y[2];
        dev_vel_y[2] = dev_vel_y[3];
        dev_vel_y[3] = shared_vel[ixLocal][izLocal];
        shared_vel[ixLocal][izLocal] = dev_vel_y[5]; // Load central point to shared memory
        dev_vel_y[5] = dev_vel_y[6];
        dev_vel_y[6] = dev_vel_y[7];
        dev_vel_y[7] = dev_vel_y[8];
        dev_vel_y[8] = dev_vel[iGlobalTemp+=yStride]

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

            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
            + dev_coeff[CX3] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[8] * dev_vel_y[8])

        ) + shared_c[ixLocal][izLocal] * shared_c[ixLocal][izLocal] - dev_n[iGlobal];

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
}
