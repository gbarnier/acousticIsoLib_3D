#include "varDeclare_3D.h"
#include <stdio.h>


/******************************************************************************/
/*********************************** Injection ********************************/
/******************************************************************************/
__global__ void injectSecondarySourceGinsu_3D(double *dev_ssLeft, double *dev_ssRight, double *dev_p0, int indexFilter, int iGpu) {

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		dev_p0[iGlobal] += dev_ssLeft[iGlobal] * dev_timeInterpFilter[indexFilter] + dev_ssRight[iGlobal] * dev_timeInterpFilter[dev_hTimeInterpFilter+indexFilter];
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

/******************************************************************************/
/*********************************** Scaling **********************************/
/******************************************************************************/
// Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2 for the Ginsu
__global__ void scaleReflectivityGinsu_3D(double *dev_model, double *dev_reflectivityScaleIn, double *dev_vel2Dtw2In, int iGpu) {

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		// Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2
		dev_model[iGlobal] *= dev_vel2Dtw2In[iGlobal] * dev_reflectivityScaleIn[iGlobal];
		// Move forward on the y-axis
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

__global__ void scaleSecondarySourceFdGinsu_3D(double *dev_timeSlice, double *dev_vel2Dtw2In, int iGpu){

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		dev_timeSlice[iGlobal] *= dev_vel2Dtw2In[iGlobal];
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

__global__ void scaleReflectivityTauGinsu_3D(double *dev_modelExtIn, double *dev_reflectivityScaleIn, double *dev_vel2Dtw2In, long long extStrideIn, int iGpu){

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		dev_modelExtIn[iGlobal+extStrideIn] *= dev_reflectivityScaleIn[iGlobal] * dev_vel2Dtw2In[iGlobal];
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

__global__ void scaleReflectivityLinHxHyGinsu_3D(double *dev_modelExtIn, double *dev_reflectivityScaleIn, long long extStride1In, long long extStride2In, int iGpu){

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		dev_modelExtIn[extStride2In+extStride1In+iGlobal] *= dev_reflectivityScaleIn[iGlobal];
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

/******************************************************************************/
/**************************** Imaging condition *******************************/
/******************************************************************************/
__global__ void imagingFwdGinsuGpu_3D(double *dev_model, double *dev_data, double *dev_sourceWavefieldDts, int iGpu) {

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
	long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		dev_data[iGlobal] = dev_model[iGlobal] * dev_sourceWavefieldDts[iGlobal];
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

__global__ void imagingAdjGinsuGpu_3D(double *dev_model, double *dev_data, double *dev_sourceWavefieldDts, int iGpu) {

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
	long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	// Here "data" is one time-slice of the receiver wavefield (its+1)

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		dev_model[iGlobal] += dev_data[iGlobal] * dev_sourceWavefieldDts[iGlobal];
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

// Forward hx and hy
__global__ void imagingHxHyFwdGinsuGpu_3D(double *dev_model, double *dev_data, double *dev_sourceWavefieldDts, int ihx, int iExt1, int ihy, int iExt2, int iGpu) {

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
	long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the x-z slice for iy = FAT

	if ( ixGlobal-FAT >= abs(ihx) && ixGlobal <= dev_nx_ginsu[iGpu]-FAT-1-abs(ihx) ){
		for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
			if ( iy-FAT >= abs(ihy) && iy <= dev_ny_ginsu[iGpu]-FAT-1-abs(ihy) ){
				dev_data[iGlobal+ihy*dev_yStride_ginsu[iGpu]+ihx*dev_nz_ginsu[iGpu]] += dev_model[iExt2*dev_extStride_ginsu[iGpu]+iExt1*dev_nVel_ginsu[iGpu]+iGlobal] * dev_sourceWavefieldDts[iGlobal-ihy*dev_yStride_ginsu[iGpu]-ihx*dev_nz_ginsu[iGpu]];
			}
			iGlobal+=dev_yStride_ginsu[iGpu];
		}
	}
}

// Adjoint hx and hy
__global__ void imagingHxHyAdjGinsuGpu_3D(double *dev_model, double *dev_data, double *dev_sourceWavefieldDts, int ihx, long long iExt1, int ihy, long long iExt2, int iGpu) {

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
	long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the x-z slice for iy = FAT

	if ( ixGlobal-FAT >= abs(ihx) && ixGlobal <= dev_nx_ginsu[iGpu]-FAT-1-abs(ihx) ){
		for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
			if ( iy-FAT >= abs(ihy) && iy <= dev_ny_ginsu[iGpu]-FAT-1-abs(ihy) ){
				dev_model[iExt2*dev_extStride_ginsu[iGpu]+iExt1*dev_nVel_ginsu[iGpu]+iGlobal] += dev_data[iGlobal+ihy*dev_yStride_ginsu[iGpu]+ihx*dev_nz_ginsu[iGpu]] * dev_sourceWavefieldDts[iGlobal-ihy*dev_yStride_ginsu[iGpu]-ihx*dev_nz_ginsu[iGpu]];
			}
			iGlobal+=dev_yStride_ginsu[iGpu];
		}
	}
}

// Forward time-lags
__global__ void imagingTauFwdGinsuGpu_3D(double *dev_model, double *dev_data, double *dev_sourceWavefieldDts, int iExt, int iGpu) {

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
	long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		dev_data[iGlobal] += dev_model[iGlobal+iExt*dev_nVel_ginsu[iGpu]] * dev_sourceWavefieldDts[iGlobal];
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

// Forward time-lags
__global__ void imagingTauFwdGinsuGpu_32_3D(double *dev_model, double *dev_data, double *dev_sourceWavefieldDts, int iExt, int iGpu) {

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * 32 + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * 32 + threadIdx.y; // Global x-coordinate
	long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

    if (izGlobal < dev_nz_ginsu[iGpu]-FAT && ixGlobal < dev_nx_ginsu[iGpu]-FAT){

    	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
    		dev_data[iGlobal] += dev_model[iGlobal+iExt*dev_nVel_ginsu[iGpu]] * dev_sourceWavefieldDts[iGlobal];
    		iGlobal+=dev_yStride_ginsu[iGpu];
    	}
    }
}

// Adjoint time-lags Ginsu
__global__ void imagingTauAdjGinsuGpu_3D(double *dev_model, double *dev_data, double *dev_sourceWavefieldDts, int iExt, int iGpu) {

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
	long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		dev_model[iGlobal+iExt*dev_nVel_ginsu[iGpu]] += dev_data[iGlobal] * dev_sourceWavefieldDts[iGlobal];
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

// Adjoint tau for the adjoint scattered wavefield in leg 2 of tomo
__global__ void imagingTauTomoAdjGinsuGpu_3D(double *dev_model, double *dev_data, double *dev_extReflectivity, long long iExt1, int iGpu) {

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
	long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		dev_model[iGlobal] += dev_data[iGlobal] * dev_extReflectivity[iGlobal+iExt1*dev_nVel_ginsu[iGpu]];
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

// Adjoint hx and hy for the adjoint scattered wavefield in leg 2 of tomo
__global__ void imagingHxHyTomoAdjGinsuGpu_3D(double *dev_model, double *dev_data, double *dev_extReflectivity, int ihx, long long iExt1, int ihy, long long iExt2, int iGpu) {

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
	long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the x-z slice for iy = FAT

	if ( ixGlobal-FAT >= abs(ihx) && ixGlobal <= dev_nx_ginsu[iGpu]-FAT-1-abs(ihx) ){
		for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
			if ( iy-FAT >= abs(ihy) && iy <= dev_ny_ginsu[iGpu]-FAT-1-abs(ihy) ){
				dev_model[iGlobal-ihy*dev_yStride_ginsu[iGpu]-ihx*dev_nz_ginsu[iGpu]] += dev_extReflectivity[iExt2*dev_extStride_ginsu[iGpu]+iExt1*dev_nVel_ginsu[iGpu]+iGlobal] * dev_data[iGlobal+ihy*dev_yStride_ginsu[iGpu]+ihx*dev_nz_ginsu[iGpu]];
			}
			iGlobal+=dev_yStride_ginsu[iGpu];
		}
	}
}

/******************************************************************************/
/********************************* Wavefield extraction ***********************/
/******************************************************************************/
__global__ void interpFineToCoarseSliceGinsu_3D(double *dev_timeSliceLeft, double *dev_timeSliceRight, double *dev_timeSliceFine, int it2, int iGpu) {

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
		// Spread to time-slice its
		dev_timeSliceLeft[iGlobal] += dev_timeSliceFine[iGlobal] * dev_timeInterpFilter[it2];
		// Spread to time-slice its+1
		dev_timeSliceRight[iGlobal] += dev_timeSliceFine[iGlobal] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		// Move forward on the y-axis
		iGlobal+=dev_yStride_ginsu[iGpu];
	}
}

__global__ void interpFineToCoarseSliceGinsu_32_3D(double *dev_timeSliceLeft, double *dev_timeSliceRight, double *dev_timeSliceFine, int it2, int iGpu) {

	long long izGlobal = FAT + blockIdx.x * 32 + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * 32 + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

    if (izGlobal < dev_nz_ginsu[iGpu]-FAT && ixGlobal < dev_nx_ginsu[iGpu]-FAT){

    	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
    		// Spread to time-slice its
    		dev_timeSliceLeft[iGlobal] += dev_timeSliceFine[iGlobal] * dev_timeInterpFilter[it2];
    		// Spread to time-slice its+1
    		dev_timeSliceRight[iGlobal] += dev_timeSliceFine[iGlobal] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
    		// Move forward on the y-axis
    		iGlobal+=dev_yStride_ginsu[iGpu];
    	}
    }
}

/******************************************************************************/
/****************************** Time derivative *******************************/
/******************************************************************************/
__global__ void srcWfldSecondTimeDerivativeGinsu_32_3D(double *dev_wavefieldSlice, double *dev_slice0, double *dev_slice1, double *dev_slice2, int iGpu){

	long long izGlobal = FAT + blockIdx.x * 32 + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * 32 + threadIdx.y; // Global x-coordinate
    long long iGlobal = FAT * dev_yStride_ginsu[iGpu] + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal; // Global position on the cube

    if (izGlobal < dev_nz_ginsu[iGpu]-FAT && ixGlobal < dev_nx_ginsu[iGpu]-FAT){

    	for (int iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){
    		dev_wavefieldSlice[iGlobal] = dev_cSide * ( dev_slice0[iGlobal] + dev_slice2[iGlobal] ) + dev_cCenter * dev_slice1[iGlobal];
    		iGlobal+=dev_yStride_ginsu[iGpu];
    	}
    }
}

/******************************************************************************/
/******************************** Damping *************************************/
/******************************************************************************/
__global__ void dampCosineEdgeGinsu_32_3D(double *dev_p1, double *dev_p2, int iGpu) {

	long long izGlobal = FAT + blockIdx.x * 32 + threadIdx.x; // Global z-coordinate on the z-axis
	long long ixGlobal = FAT + blockIdx.y * 32 + threadIdx.y; // Global x-coordinate on the x-axis
    long long yStride = dev_nz_ginsu[iGpu] * dev_nx_ginsu[iGpu];

	if (izGlobal < dev_nz_ginsu[iGpu]-FAT && ixGlobal < dev_nx_ginsu[iGpu]-FAT){

	    for (long long iyGlobal=FAT; iyGlobal<dev_ny_ginsu[iGpu]-FAT; iyGlobal++){

	    	// Compute distance to the closest edge of model (not including the fat)
			// For example, the first non fat element will have a distance of 0
	    	long long distToEdge = min4(izGlobal-FAT, ixGlobal-FAT, dev_nz_ginsu[iGpu]-izGlobal-1-FAT, dev_nx_ginsu[iGpu]-ixGlobal-1-FAT);
	        distToEdge = min2(distToEdge,min2(iyGlobal-FAT, dev_ny_ginsu[iGpu]-iyGlobal-1-FAT));

	    	if (distToEdge < dev_minPad_ginsu[iGpu]){

	            // Compute global index
		        long long iGlobal = iyGlobal * yStride + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal;

	    		// Compute damping coefficient
	    		// double damp = dampCosArray[distToEdge];
                double damp = dev_cosDampingCoeffGinsuConstant[iGpu][distToEdge];

	    		// Apply damping
	    		dev_p1[iGlobal] *= damp;
	    		dev_p2[iGlobal] *= damp;
	    	}
	    }
	}
}

__global__ void dampCosineEdgeFreeSurfaceGinsu_3D(double *dev_p1, double *dev_p2, int iGpu) {

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate on the x-axis
    long long yStride = dev_nz_ginsu[iGpu] * dev_nx_ginsu[iGpu];

    for (long long iyGlobal=FAT; iyGlobal<dev_ny_ginsu[iGpu]-FAT; iyGlobal++){

    	// Compute distance to the closest edge of model (not including the fat)
		// For example, the first non fat element will have a distance of 0
		long long distToEdge = min4(ixGlobal-FAT, dev_nx_ginsu[iGpu]-ixGlobal-1-FAT, iyGlobal-FAT, dev_ny_ginsu[iGpu]-iyGlobal-1-FAT);
        distToEdge = min2(distToEdge, dev_nz_ginsu[iGpu]-izGlobal-1-FAT);

    	if (distToEdge < dev_minPad_ginsu[iGpu]){

            // Compute global index
	        long long iGlobal = iyGlobal * yStride + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal;

    		// Compute damping coefficient
    		double damp = dev_cosDampingCoeffGinsuConstant[iGpu][distToEdge];

    		// Apply damping
    		dev_p1[iGlobal] *= damp;
    		dev_p2[iGlobal] *= damp;
    	}
    }
}

__global__ void dampCosineEdgeFreeSurfaceGinsu_32_3D(double *dev_p1, double *dev_p2, int iGpu) {

	long long izGlobal = FAT + blockIdx.x * 32 + threadIdx.x; // Global z-coordinate on the z-axis
	long long ixGlobal = FAT + blockIdx.y * 32 + threadIdx.y; // Global x-coordinate on the x-axis
    long long yStride = dev_nz_ginsu[iGpu] * dev_nx_ginsu[iGpu];

    if (izGlobal < dev_nz_ginsu[iGpu]-FAT && ixGlobal < dev_nx_ginsu[iGpu]-FAT){

        for (long long iyGlobal=FAT; iyGlobal<dev_ny_ginsu[iGpu]-FAT; iyGlobal++){

        	// Compute distance to the closest edge of model (not including the fat)
    		// For example, the first non fat element will have a distance of 0
    		long long distToEdge = min4(ixGlobal-FAT, dev_nx_ginsu[iGpu]-ixGlobal-1-FAT, iyGlobal-FAT, dev_ny_ginsu[iGpu]-iyGlobal-1-FAT);
            distToEdge = min2(distToEdge, dev_nz_ginsu[iGpu]-izGlobal-1-FAT);

        	if (distToEdge < dev_minPad_ginsu[iGpu]){

                // Compute global index
    	        long long iGlobal = iyGlobal * yStride + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal;

        		// Compute damping coefficient
        		double damp = dev_cosDampingCoeffGinsuConstant[iGpu][distToEdge];

        		// Apply damping
        		dev_p1[iGlobal] *= damp;
        		dev_p2[iGlobal] *= damp;
        	}
        }
    }
}

/******************************************************************************/
/******************************* Forward stepper ******************************/
/******************************************************************************/
__global__ void stepFwdGinsuGpu_3D(double *dev_o, double *dev_c, double *dev_n, double *dev_vel2Dtw2, int iGpu){

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
    double dev_c_y[2*FAT];

    // Number of elements in one y-slice
    long long yStride = dev_nz_ginsu[iGpu] * dev_nx_ginsu[iGpu];

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction into dev_c_y (Remember: each thread has its own version of this array)
    dev_c_y[1] = dev_c[iGlobalTemp]; // iy = 0
    dev_c_y[2] = dev_c[iGlobalTemp+=yStride]; // iy = 1
    dev_c_y[3] = dev_c[iGlobalTemp+=yStride]; // iy = 2
    shared_c[ixLocal][izLocal] = dev_c[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
    dev_c_y[4] = dev_c[iGlobalTemp+=yStride]; // iy = 4
    dev_c_y[5] = dev_c[iGlobalTemp+=yStride]; // iy = 5
    dev_c_y[6] = dev_c[iGlobalTemp+=yStride];// iy = 6
    dev_c_y[7] = dev_c[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 7

    // Loop over y
	#pragma unroll(9)
    for (long long iy=FAT; iy<dev_ny_ginsu[iGpu]-FAT; iy++){

        // Update values along the y-axis
        dev_c_y[0] = dev_c_y[1];
        dev_c_y[1] = dev_c_y[2];
        dev_c_y[2] = dev_c_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
        shared_c[ixLocal][izLocal] = dev_c_y[4]; // Store the middle one in the shared memory (it will be re-used to compute the Laplacian in the z- and x-directions)
        dev_c_y[4] = dev_c_y[5];
        dev_c_y[5] = dev_c_y[6];
        dev_c_y[6] = dev_c_y[7];
        dev_c_y[7] = dev_c[iGlobalTemp+=yStride]; // The last point of the stencil now points to the next y-slice

        // Remark on assignments just above:
        // iyTemp = iy + FAT
        // This guy points to the iy with the largest y-index needed to compute the Laplacian at the new y-position

        // Load the halos in the x-direction
        // Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
        if (threadIdx.y < FAT) {
    			// shared_c[ixLocal-FAT][izLocal] = dev_c[iGlobal-dev_nz*FAT]; // Left side
				shared_c[threadIdx.y][izLocal] = dev_c[iGlobal-dev_nz_ginsu[iGpu]*FAT]; // Left side
    			shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c[iGlobal+dev_nz_ginsu[iGpu]*BLOCK_SIZE_X]; // Right side
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
            + dev_coeff[CY1] * ( dev_c_y[3] + dev_c_y[4] )

            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] )
            + dev_coeff[CY2] * ( dev_c_y[2] + dev_c_y[5] )

            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] )
            + dev_coeff[CY3] * ( dev_c_y[1] + dev_c_y[6] )

            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] )
            + dev_coeff[CY4] * ( dev_c_y[0] + dev_c_y[7] )

        ) + 2.0 * shared_c[ixLocal][izLocal] - dev_o[iGlobal];

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
}

__global__ void setFreeSurfaceConditionFwdGinsuGpu_3D(double *dev_c, int iGpu){

	// Global coordinates for the slowest axis
	long long iyGlobal = FAT + blockIdx.y * BLOCK_SIZE_Y + threadIdx.y; // Global y-coordinate
	long long ixGlobal = FAT + blockIdx.x * BLOCK_SIZE_X + threadIdx.x; // Global x-coordinate
	long long iGlobal = iyGlobal * dev_yStride_ginsu[iGpu] + ixGlobal * dev_nz_ginsu[iGpu];

	if (iyGlobal < dev_ny_ginsu[iGpu]-FAT){
		dev_c[iGlobal+FAT] = 0.0; // Set the value of the pressure field to zero at the free surface
		dev_c[iGlobal] = -dev_c[iGlobal+2*FAT];
		dev_c[iGlobal+1] = -dev_c[iGlobal+2*FAT-1];
		dev_c[iGlobal+2] = -dev_c[iGlobal+2*FAT-2];
		dev_c[iGlobal+3] = -dev_c[iGlobal+2*FAT-3];
	}
}

/******************************************************************************/
/******************************* Adjoint stepper ******************************/
/******************************************************************************/
__global__ void stepAdjGinsuGpu_3D(double *dev_o, double *dev_c, double *dev_n, double *dev_vel2Dtw2, int iGpu) {

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
    double dev_c_y[2*FAT]; // Array for the current wavefield y-slice
    double dev_vel_y[2*FAT]; // Array for the scaled velocity y-slice

    // Number of elements in one y-slice
    long long yStride = dev_nz_ginsu[iGpu] * dev_nx_ginsu[iGpu];

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction (Remember: each thread has its own version of dev_c_y and dev_vel_y array)
    // Points from the current wavefield time-slice that will be used by the current block
    dev_c_y[1] = dev_c[iGlobalTemp];								dev_vel_y[1] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[2] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[2] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[3] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[3] = dev_vel2Dtw2[iGlobalTemp];
	// These ones go to shared memory because used multiple times in Laplacian computation for the z- and x-directions
    shared_c[ixLocal][izLocal] = dev_c[iGlobalTemp+=yStride]; 		shared_vel[ixLocal][izLocal] = dev_vel2Dtw2[iGlobalTemp];
	dev_c_y[4] = dev_c[iGlobalTemp+=yStride];					    dev_vel_y[4] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[5] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[5] = dev_vel2Dtw2[iGlobalTemp];
	dev_c_y[6] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[6] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[7] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[7] = dev_vel2Dtw2[iGlobalTemp];

    // Loop over y
    #pragma unroll(9)
    for (long long iyGlobal=FAT; iyGlobal<dev_ny_ginsu[iGpu]-FAT; iyGlobal++){

        // Update temporary arrays with current wavefield values along the y-axis
        dev_c_y[0] = dev_c_y[1];										dev_vel_y[0] = dev_vel_y[1];
        dev_c_y[1] = dev_c_y[2];										dev_vel_y[1] = dev_vel_y[2];
        dev_c_y[2] = dev_c_y[3];										dev_vel_y[2] = dev_vel_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];		                dev_vel_y[3] = shared_vel[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
        shared_c[ixLocal][izLocal] = dev_c_y[4]; 		                shared_vel[ixLocal][izLocal] = dev_vel_y[4]; // Load central points to shared memory (for both current slice and scaled velocity)
        dev_c_y[4] = dev_c_y[5];										dev_vel_y[4] = dev_vel_y[5];
        dev_c_y[5] = dev_c_y[6];										dev_vel_y[5] = dev_vel_y[6];
        dev_c_y[6] = dev_c_y[7];										dev_vel_y[6] = dev_vel_y[7];
        dev_c_y[7] = dev_c[iGlobalTemp+=yStride];		                dev_vel_y[7] = dev_vel2Dtw2[iGlobalTemp];

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
    	    shared_c[ixLocal-FAT][izLocal] = dev_c[iGlobal-dev_nz_ginsu[iGpu]*FAT];
            shared_vel[ixLocal-FAT][izLocal] = dev_vel2Dtw2[iGlobal-dev_nz_ginsu[iGpu]*FAT];
            // Right side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c[iGlobal+dev_nz_ginsu[iGpu]*BLOCK_SIZE_X];
    		shared_vel[ixLocal+BLOCK_SIZE_X][izLocal] = dev_vel2Dtw2[iGlobal+dev_nz_ginsu[iGpu]*BLOCK_SIZE_X];
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads();

        // Apply adjoint stepping operator
        dev_o[iGlobal] = (

            dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
            + dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[4] * dev_vel_y[4])

            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
            + dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[5] * dev_vel_y[5])

            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
            + dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[6] * dev_vel_y[6])

            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[7] * dev_vel_y[7])

        ) + 2.0 * shared_c[ixLocal][izLocal] - dev_n[iGlobal];

        // Move forward one grid point in the y-direction
        iGlobal = iGlobal + yStride;

    }
}

__global__ void stepAdjFreeSurfaceGinsuGpu_3D(double *dev_o, double *dev_c, double *dev_n, double *dev_vel2Dtw2, int iGpu) {

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
    double dev_c_y[2*FAT]; // Array for the current wavefield y-slice
    double dev_vel_y[2*FAT]; // Array for the scaled velocity y-slice

    // Number of elements in one y-slice
    long long yStride = dev_nz_ginsu[iGpu] * dev_nx_ginsu[iGpu];

    // Global index of the first element at which we are going to compute the Laplacian
    // Skip the first FAT elements on the y-axis
    long long iGlobal = FAT * yStride + dev_nz_ginsu[iGpu] * ixGlobal + izGlobal;

    // Global index of the element with the smallest y-position needed to compute Laplacian at iGlobal
    long long iGlobalTemp = iGlobal - FAT * yStride;

    // Load the values along the y-direction (Remember: each thread has its own version of dev_c_y and dev_vel_y array)
    // Points from the current wavefield time-slice that will be used by the current block
    dev_c_y[1] = dev_c[iGlobalTemp];								dev_vel_y[1] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[2] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[2] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[3] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[3] = dev_vel2Dtw2[iGlobalTemp];
	// These ones go to shared memory because used multiple times in Laplacian computation for the z- and x-directions
    shared_c[ixLocal][izLocal] = dev_c[iGlobalTemp+=yStride]; 		shared_vel[ixLocal][izLocal] = dev_vel2Dtw2[iGlobalTemp];
	dev_c_y[4] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[4] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[5] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[5] = dev_vel2Dtw2[iGlobalTemp];
	dev_c_y[6] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[6] = dev_vel2Dtw2[iGlobalTemp];
    dev_c_y[7] = dev_c[iGlobalTemp+=yStride];						dev_vel_y[7] = dev_vel2Dtw2[iGlobalTemp];

    // Loop over y
    for (long long iyGlobal=FAT; iyGlobal<dev_ny_ginsu[iGpu]-FAT; iyGlobal++){

        // Update temporary arrays with current wavefield values along the y-axis
        dev_c_y[0] = dev_c_y[1];										dev_vel_y[0] = dev_vel_y[1];
        dev_c_y[1] = dev_c_y[2];										dev_vel_y[1] = dev_vel_y[2];
        dev_c_y[2] = dev_c_y[3];										dev_vel_y[2] = dev_vel_y[3];
        dev_c_y[3] = shared_c[ixLocal][izLocal];		                dev_vel_y[3] = shared_vel[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
        shared_c[ixLocal][izLocal] = dev_c_y[4]; 		                shared_vel[ixLocal][izLocal] = dev_vel_y[4]; // Load central points to shared memory (for both current slice and scaled velocity)
        dev_c_y[4] = dev_c_y[5];										dev_vel_y[4] = dev_vel_y[5];
        dev_c_y[5] = dev_c_y[6];										dev_vel_y[5] = dev_vel_y[6];
        dev_c_y[6] = dev_c_y[7];										dev_vel_y[6] = dev_vel_y[7];
        dev_c_y[7] = dev_c[iGlobalTemp+=yStride];		                dev_vel_y[7] = dev_vel2Dtw2[iGlobalTemp];

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
    		shared_c[ixLocal-FAT][izLocal] = dev_c[iGlobal-dev_nz_ginsu[iGpu]*FAT];
            shared_vel[ixLocal-FAT][izLocal] = dev_vel2Dtw2[iGlobal-dev_nz_ginsu[iGpu]*FAT];
            // Right side
    		shared_c[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c[iGlobal+dev_nz_ginsu[iGpu]*BLOCK_SIZE_X];
    		shared_vel[ixLocal+BLOCK_SIZE_X][izLocal] = dev_vel2Dtw2[iGlobal+dev_nz_ginsu[iGpu]*BLOCK_SIZE_X];
    	}

        // Wait until all threads of this block have loaded the slice y-slice into shared memory
        __syncthreads();

        // Apply adjoint stepping operator
				// if (izGlobal == 4){
				// dev_o[iGlobal] = - dev_n[iGlobal];
				// 	dev_o[iGlobal] = 0.0;
				// }
		if (izGlobal == 5){
	        dev_o[iGlobal] = (

	            dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

	            + dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
	            + dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
	            + dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[4] * dev_vel_y[4])

				- dev_coeff[CZ2] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]
				- dev_coeff[CZ3] * shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1]
				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2]

	            + dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
	            + dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
	            + dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[5] * dev_vel_y[5])

	            + dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
	            + dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
	            + dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[6] * dev_vel_y[6])

	            + dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
	            + dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
	            + dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[7] * dev_vel_y[7])

	        ) + 2.0 * shared_c[ixLocal][izLocal] - dev_n[iGlobal];
		}

		if (izGlobal == 6){

			dev_o[iGlobal] = (

				dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

				+ dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
				+ dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
				+ dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[4] * dev_vel_y[4])

				- dev_coeff[CZ3] * shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1]
				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

				+ dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
				+ dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
				+ dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[5] * dev_vel_y[5])

				+ dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
				+ dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
				+ dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[6] * dev_vel_y[6])

				+ dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
				+ dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
				+ dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[7] * dev_vel_y[7])

		    ) + 2.0 * shared_c[ixLocal][izLocal] - dev_n[iGlobal];
		}

		if (izGlobal == 7){

			dev_o[iGlobal] = (

				dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

				+ dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
				+ dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
				+ dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[4] * dev_vel_y[4])

				- dev_coeff[CZ4] * shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2]

				+ dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
				+ dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
				+ dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[5] * dev_vel_y[5])

				+ dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
				+ dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
				+ dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[6] * dev_vel_y[6])

				+ dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
				+ dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
				+ dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[7] * dev_vel_y[7])

			    ) + 2.0 * shared_c[ixLocal][izLocal] - dev_n[iGlobal];
			}

		if (izGlobal > 7){

			dev_o[iGlobal] = (

				dev_coeff[C0] * shared_c[ixLocal][izLocal] * shared_vel[ixLocal][izLocal]

				+ dev_coeff[CZ1] * ( shared_c[ixLocal][izLocal-1] * shared_vel[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] * shared_vel[ixLocal][izLocal+1] )
				+ dev_coeff[CX1] * ( shared_c[ixLocal-1][izLocal] * shared_vel[ixLocal-1][izLocal] + shared_c[ixLocal+1][izLocal] * shared_vel[ixLocal+1][izLocal])
				+ dev_coeff[CY1] * ( dev_c_y[3] * dev_vel_y[3] + dev_c_y[4] * dev_vel_y[4])

				+ dev_coeff[CZ2] * ( shared_c[ixLocal][izLocal-2] * shared_vel[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] * shared_vel[ixLocal][izLocal+2] )
				+ dev_coeff[CX2] * ( shared_c[ixLocal-2][izLocal] * shared_vel[ixLocal-2][izLocal] + shared_c[ixLocal+2][izLocal] * shared_vel[ixLocal+2][izLocal])
				+ dev_coeff[CY2] * ( dev_c_y[2] * dev_vel_y[2] + dev_c_y[5] * dev_vel_y[5])

				+ dev_coeff[CZ3] * ( shared_c[ixLocal][izLocal-3] * shared_vel[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] * shared_vel[ixLocal][izLocal+3] )
				+ dev_coeff[CX3] * ( shared_c[ixLocal-3][izLocal] * shared_vel[ixLocal-3][izLocal] + shared_c[ixLocal+3][izLocal] * shared_vel[ixLocal+3][izLocal] )
				+ dev_coeff[CY3] * ( dev_c_y[1] * dev_vel_y[1] + dev_c_y[6] * dev_vel_y[6])

				+ dev_coeff[CZ4] * ( shared_c[ixLocal][izLocal-4] * shared_vel[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] * shared_vel[ixLocal][izLocal+4] )
				+ dev_coeff[CX4] * ( shared_c[ixLocal-4][izLocal] * shared_vel[ixLocal-4][izLocal] + shared_c[ixLocal+4][izLocal] * shared_vel[ixLocal+4][izLocal] )
				+ dev_coeff[CY4] * ( dev_c_y[0] * dev_vel_y[0] + dev_c_y[7] * dev_vel_y[7])

			) + 2.0 * shared_c[ixLocal][izLocal] - dev_n[iGlobal];
		}

        iGlobal = iGlobal + yStride;
    }
}
