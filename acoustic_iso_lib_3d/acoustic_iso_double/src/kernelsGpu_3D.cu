#include "varDeclare_3D.h"
#include <stdio.h>

/******************************************************************************/
/*********************************** Injection ********************************/
/******************************************************************************/
/* Inject source: no need for a "if" statement because the number of threads = nb devices */
__global__ void injectSourceLinear_3D(double *dev_signalIn, double *dev_timeSlice, long long itw, long long *dev_sourcesPositionReg){
	long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
	dev_timeSlice[dev_sourcesPositionReg[iThread]] += dev_signalIn[iThread * dev_ntw + itw]; // Time is the fast axis
}

/* Interpolate and inject data */
__global__ void interpLinearInjectData_3D(double *dev_signalIn, double *dev_timeSlice, long long its, long long it2, long long *dev_receiversPositionReg) {
	long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (iThread < dev_nReceiversReg) {
		dev_timeSlice[dev_receiversPositionReg[iThread]] += dev_signalIn[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
	}
}

/* Inject the secondary source for Born */
__global__ void injectSecondarySource_3D(double *dev_ssLeft, double *dev_ssRight, double *dev_p0, int indexFilter) {

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride + dev_nz * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny-FAT; iy++){
		dev_p0[iGlobal] += dev_ssLeft[iGlobal] * dev_timeInterpFilter[indexFilter] + dev_ssRight[iGlobal] * dev_timeInterpFilter[dev_hTimeInterpFilter+indexFilter];
		iGlobal+=dev_yStride;
	}
}

/******************************************************************************/
/*********************************** Extraction *******************************/
/******************************************************************************/
/* Extract and interpolate data */
__global__ void recordLinearInterpData_3D(double *dev_newTimeSlice, double *dev_signalOut, long long its, long long it2, long long *dev_receiversPositionReg) {

	long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (iThread < dev_nReceiversReg) {
		dev_signalOut[dev_nts*iThread+its]   += dev_newTimeSlice[dev_receiversPositionReg[iThread]] * dev_timeInterpFilter[it2];
		dev_signalOut[dev_nts*iThread+its+1] += dev_newTimeSlice[dev_receiversPositionReg[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
	}
}

/* Extract source for "nonlinear adjoint" */
__global__ void recordSource_3D(double *dev_wavefield, double *dev_signalOut, long long itw, long long *dev_sourcesPositionReg) {
	long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
	dev_signalOut[dev_ntw*iThread + itw] += dev_wavefield[dev_sourcesPositionReg[iThread]];
}

/******************************************************************************/
/*********************************** Scaling **********************************/
/******************************************************************************/
// Scale reflecticity by for non-extended Born
// Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2
__global__ void scaleReflectivity_3D(double *dev_model, double *dev_reflectivityScale, double *dev_vel2Dtw2) {

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride + dev_nz * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny-FAT; iy++){
		// Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2
		dev_model[iGlobal] *= dev_vel2Dtw2[iGlobal] * dev_reflectivityScale[iGlobal];
		// Move forward on the y-axis
		iGlobal+=dev_yStride;
	}
}


/******************************************************************************/
/**************************** Imaging condition *******************************/
/******************************************************************************/
// Forward non-extended
__global__ void imagingFwdGpu_3D(double *dev_model, double *dev_data, double *dev_sourceWavefieldDts) {

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
	long long iGlobal = FAT * dev_yStride + dev_nz * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny-FAT; iy++){
		dev_data[iGlobal] = dev_model[iGlobal] * dev_sourceWavefieldDts[iGlobal];
		iGlobal+=dev_yStride;
	}
}

/******************************************************************************/
/********************************* Wavefield extraction ***********************/
/******************************************************************************/
__global__ void interpFineToCoarseSlice(double *dev_timeSliceLeft, double *dev_timeSliceRight, double *dev_timeSliceFine, int its, int it2) {

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride + dev_nz * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny-FAT; iy++){
		// Spread to time-slice its
		dev_timeSliceLeft[iGlobal] += dev_timeSliceFine[iGlobal] * dev_timeInterpFilter[it2];
		// Spread to time-slice its+1
		dev_timeSliceRight[iGlobal] += dev_timeSliceFine[iGlobal] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		// Move forward on the y-axis
		iGlobal+=dev_yStride;
	}

}

/******************************************************************************/
/******************************** Damping *************************************/
/******************************************************************************/
__global__ void dampCosineEdge_3D(double *dev_p1, double *dev_p2) {

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate on the x-axis
    long long yStride = dev_nz * dev_nx;

    for (long long iyGlobal=FAT; iyGlobal<dev_ny-FAT; iyGlobal++){

    	// Compute distance to the closest edge of model (not including the fat)
		// For example, the first non fat element will have a distance of 0
    	long long distToEdge = min4(izGlobal-FAT, ixGlobal-FAT, dev_nz-izGlobal-1-FAT, dev_nx-ixGlobal-1-FAT);
        distToEdge = min2(distToEdge,min2(iyGlobal-FAT,dev_ny-iyGlobal-1-FAT));

    	if (distToEdge < dev_minPad){

            // Compute global index
	        long long iGlobal = iyGlobal * yStride + dev_nz * ixGlobal + izGlobal;

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
    dev_c_y[1] = dev_c[iGlobalTemp]; // iy = 0
    dev_c_y[2] = dev_c[iGlobalTemp+=yStride]; // iy = 1
    dev_c_y[3] = dev_c[iGlobalTemp+=yStride]; // iy = 2
    shared_c[ixLocal][izLocal] = dev_c[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
    dev_c_y[5] = dev_c[iGlobalTemp+=yStride]; // iy = 4
    dev_c_y[6] = dev_c[iGlobalTemp+=yStride]; // iy = 5
    dev_c_y[7] = dev_c[iGlobalTemp+=yStride];// iy = 6
    dev_c_y[8] = dev_c[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 7

    // Loop over y
    for (long long iy=FAT; iy<dev_ny-FAT; iy++){

        // Update values along the y-axis
        dev_c_y[0] = dev_c_y[1];
        dev_c_y[1] = dev_c_y[2];
        dev_c_y[2] = dev_c_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
        shared_c[ixLocal][izLocal] = dev_c_y[5]; // Store the middle one in the shared memory (it will be re-used to compute the Laplacian in the z- and x-directions)
        dev_c_y[5] = dev_c_y[6];
        dev_c_y[6] = dev_c_y[7];
        dev_c_y[7] = dev_c_y[8];
        dev_c_y[8] = dev_c[iGlobalTemp+=yStride]; // The last point of the stencil now points to the next y-slice

        // Remark on assignments just above:
        // iyTemp = iy + FAT
        // This guy points to the iy with the largest y-index needed to compute the Laplacian at the new y-position

        // Load the halos in the x-direction
        // Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
        if (threadIdx.y < FAT) {
    		// shared_c[ixLocal-FAT][izLocal] = dev_c[iGlobal-dev_nz*FAT]; // Left side
			shared_c[threadIdx.y][izLocal] = dev_c[iGlobal-dev_nz*FAT]; // Left side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
    	}

        // Load the halos in the z-direction
        if (threadIdx.x < FAT) {
    		shared_c[ixLocal][threadIdx.x] = dev_c[iGlobal-FAT]; // Up
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

        ) + 2.0 * shared_c[ixLocal][izLocal] - dev_o[iGlobal];

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
}

/******************************************************************************/
/******************************* Adjoint stepper ******************************/
/******************************************************************************/
/* Adjoint stepper (no damping) */
__global__ void stepAdjGpu_3D(double *dev_o, double *dev_c, double *dev_n, double *dev_vel2Dtw2) {

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

        ) + 2.0 * shared_c[ixLocal][izLocal] - dev_n[iGlobal];

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
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
    int iGlobal = FAT * dev_yStride + dev_nz * ixGlobal + izGlobal; // Global position on the cube

    for (int iy=FAT; iy<dev_ny-FAT; iy++){
		dev_data[iGlobal] = dev_model[iGlobal] * dev_sourceWavefieldDts[iGlobal];
		iGlobal+=dev_yStride;
	}
}