#include <cstring>
#include <iostream>
#include "tomoExtShotsGpuFunctions_3D.h"
#include "varDeclare_3D.h"
#include "kernelsGpu_3D.cu"
#include "cudaErrors_3D.cu"
#include <vector>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

/******************************************************************************/
/****************** Declaration of auxiliary functions ************************/
/******************************************************************************/
// Note: The implementations of these auxiliary functions are done at the bottom of the file
void computeTomoSrcWfldDt2_3D(double *dev_sourcesIn, double *wavefield1, long long *dev_sourcesPositionsRegIn, int nSourcesRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn);

void computeTomoRecWfld_3D(double *dev_dataRegDtsIn, double *wavefield2, long long *dev_receiversPositionsRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn);

// Forward + Subsurface offsets
void computeTomoLeg1HxHyFwd_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_dataRegDtsIn, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

void computeTomoLeg2HxHyFwd_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_dataRegDtsIn, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

// Forward + Time-lags
// void computeTomoLeg1TauFwd_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_dataRegDtsIn, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);
//
// void computeTomoLeg2TauFwd_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_dataRegDtsIn, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

// Adjoint + Subsurface offsets
void computeTomoLeg1HxHyAdj_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, double *dev_dataRegDtsQcIn);

void computeTomoLeg2HxHyAdj_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, double *dev_dataRegDtsQcIn);

// Adjoint + Time-lags
// void computeTomoLeg1TauAdj_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, double *dev_dataRegDtsQcIn);
//
// void computeTomoLeg2TauAdj_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, double *dev_dataRegDtsQcIn);

/******************************************************************************/
/**************************** Initialization **********************************/
/******************************************************************************/
/* Parameter settings */
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo){

	int nDevice, driver;
	cudaGetDeviceCount(&nDevice);

	if (info == 1){

		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "---------------------------- INFO FOR GPU# " << deviceNumberInfo << " ----------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;

		// Number of devices
		std::cout << "Number of requested GPUs: " << gpuList.size() << std::endl;
		std::cout << "Number of available GPUs: " << nDevice << std::endl;
		std::cout << "Id of requested GPUs: ";
		for (int iGpu=0; iGpu<gpuList.size(); iGpu++){
			if (iGpu<gpuList.size()-1){std::cout << gpuList[iGpu] << ", ";}
 			else{ std::cout << gpuList[iGpu] << std::endl;}
		}

		// Driver version
		std::cout << "Cuda driver version: " << cudaDriverGetVersion(&driver) << std::endl;

		// Get properties
		cudaDeviceProp dprop;
		cudaGetDeviceProperties(&dprop,deviceNumberInfo);

		// Display
		std::cout << "Name: " << dprop.name << std::endl;
		std::cout << "Total global memory: " << dprop.totalGlobalMem/(1024*1024*1024) << " [GB] " << std::endl;
		std::cout << "Shared memory per block: " << dprop.sharedMemPerBlock/1024 << " [kB]" << std::endl;
		std::cout << "Number of register per block: " << dprop.regsPerBlock << std::endl;
		std::cout << "Warp size: " << dprop.warpSize << " [threads]" << std::endl;
		std::cout << "Maximum pitch allowed for memory copies in bytes: " << dprop.memPitch/(1024*1024*1024) << " [GB]" << std::endl;
		std::cout << "Maximum threads per block: " << dprop.maxThreadsPerBlock << std::endl;
		std::cout << "Maximum block dimensions: " << "(" << dprop.maxThreadsDim[0] << ", " << dprop.maxThreadsDim[1] << ", " << dprop.maxThreadsDim[2] << ")" << std::endl;
		std::cout << "Maximum grid dimensions: " << "(" << dprop.maxGridSize[0] << ", " << dprop.maxGridSize[1] << ", " << dprop.maxGridSize[2] << ")" << std::endl;
		std::cout << "Total constant memory: " << dprop.totalConstMem/1024 << " [kB]" << std::endl;
		std::cout << "Number of streaming multiprocessors on device: " << dprop.multiProcessorCount << std::endl;
		if (dprop.deviceOverlap == 1) {std::cout << "Device can simultaneously perform a cudaMemcpy() and kernel execution" << std::endl;}
		if (dprop.deviceOverlap != 1) {std::cout << "Device cannot simultaneously perform a cudaMemcpy() and kernel execution" << std::endl;}
		if (dprop.canMapHostMemory == 1) { std::cout << "Device can map host memory" << std::endl; }
		if (dprop.canMapHostMemory != 1) { std::cout << "Device cannot map host memory" << std::endl; }
		if (dprop.concurrentKernels == 1) {std::cout << "Device can support concurrent kernel" << std::endl;}
		if (dprop.concurrentKernels != 1) {std::cout << "Device cannot support concurrent kernel execution" << std::endl;}

		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	// Check that the number of requested GPU is less or equal to the total number of available GPUs
	if (gpuList.size()>nDevice) {
		std::cout << "**** ERROR [getGpuInfo_3D]: Number of requested GPU greater than available GPUs ****" << std::endl;
		return false;
	}

	// Check that the GPU numbers in the list are between 0 and nGpu-1
	for (int iGpu=0; iGpu<gpuList.size(); iGpu++){
		if (gpuList[iGpu]<0 || gpuList[iGpu]>nDevice-1){
			std::cout << "**** ERROR [getGpuInfo_3D]: One of the element of the GPU Id list is not a valid GPU Id number ****" << std::endl;
			return false;
		}
	}

	return true;
}

/* Initialize GPU */
void initTomoExtGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nExt1, int nExt2, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nz = nz;
	host_nx = nx;
    host_ny = ny;
	host_yStride = nz * nx;
	host_nts = nts;
	host_dts = dts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;
    host_nExt1 = nExt1;
    host_nExt2 = nExt2;
	host_hExt1 = (nExt1-1)/2;
    host_hExt2 = (nExt2-1)/2;
    host_nModelExt = nz * nx * ny * nExt1 * nExt2;
    host_nVel = nz * nx * ny;
	host_extStride = host_nExt1 * host_nVel;
	host_leg1 = leg1;
	host_leg2 = leg2;

	// Coefficients for second-order time derivative
	host_cSide = 1.0 / (host_dts*host_dts);
	host_cCenter = -2.0 / (host_dts*host_dts);
	//
	// host_cSide = 0.0;
	// host_cCenter = 1.0;

	/**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {

		// Time slices for FD stepping
		dev_p0 = new double*[nGpu];
		dev_p1 = new double*[nGpu];
		dev_temp1 = new double*[nGpu];

		dev_pLeft = new double*[nGpu];
		dev_pRight = new double*[nGpu];
		dev_pTemp = new double*[nGpu];

		dev_pDt0 = new double*[nGpu];
		dev_pDt1 = new double*[nGpu];
		dev_pDt2 = new double*[nGpu];
		dev_pDtTemp = new double*[nGpu];

		dev_pSourceWavefield = new double*[nGpu];
		dev_pRecWavefield = new double*[nGpu];

		// Data and model
		dev_dataRegDts = new double*[nGpu];
		dev_dataRegDtsQc = new double*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new long long*[nGpu];
		dev_receiversPositionReg = new long long*[nGpu];

        // Sources signal
		dev_sourcesSignals = new double*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new double*[nGpu];

        // Reflectivity scaling
		dev_reflectivityScale = new double*[nGpu];

        // Background perturbation ("model" for tomo)
		dev_modelTomo = new double*[nGpu];

		// Extended reflectivity for tomo
		dev_extReflectivity = new double*[nGpu];

		// Debug model and data
		dev_modelDebug = new double*[nGpu];
		dev_dataDebug = new double*[nGpu];

        // Streams
		compStream = new cudaStream_t[nGpu];
		transferStream = new cudaStream_t[nGpu];
		transferStreamH2D = new cudaStream_t[nGpu];
		transferStreamD2H = new cudaStream_t[nGpu];
		pin_wavefieldSlice1 = new double*[nGpu];
		pin_wavefieldSlice2 = new double*[nGpu];
		dev_pStream = new double*[nGpu];

		// Time-lags
		dev_pSourceWavefieldTau = new double**[nGpu];
		for (int iGpu=0; iGpu<nGpu; iGpu++){
			dev_pSourceWavefieldTau[iGpu] = new double*[4*host_hExt1+1];
		}
		dev_pTempTau = new double*[nGpu];

	}
	/**************************** COMPUTE LAPLACIAN COEFFICIENTS ************************/
	double host_coeff[COEFF_SIZE] = get_coeffs((double)dz,(double)dx,(double)dy); // Stored on host

	/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
	// Time interpolation filter length / half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Subsampling parameter for time interpolation is too high ****" << std::endl;
		assert (1==2);
	}

	// Allocate and fill time interpolation filter
	double interpFilter[nInterpFilter];
	for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
		interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
		interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
		interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
		interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
	}

	/************************* COMPUTE COSINE DAMPING COEFFICIENTS **********************/
	if (minPad>=PAD_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Padding value is too high ****" << std::endl;
		assert (1==2);
	}
	double cosDampingCoeff[minPad];

	// Cosine padding
	for (int iFilter=FAT; iFilter<FAT+minPad; iFilter++){
		double arg = M_PI / (1.0 * minPad) * 1.0 * (minPad-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		cosDampingCoeff[iFilter-FAT] = arg;
	}

	// Check that the block size is consistent between parfile and "varDeclare.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare file ****" << std::endl;
		assert (1==2);
	}

	/**************************** COPY TO CONSTANT MEMORY *******************************/
	// Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_coeff, host_coeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice)); // Copy Laplacian coefficients to device

	// Time interpolation filter
	cuda_call(cudaMemcpyToSymbol(dev_nTimeInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
	cuda_call(cudaMemcpyToSymbol(dev_hTimeInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
	cuda_call(cudaMemcpyToSymbol(dev_timeInterpFilter, interpFilter, nInterpFilter*sizeof(double), 0, cudaMemcpyHostToDevice)); // Filter

	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeff, &cosDampingCoeff, minPad*sizeof(double), 0, cudaMemcpyHostToDevice)); // Array for damping
	cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(double), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula
	cuda_call(cudaMemcpyToSymbol(dev_minPad, &minPad, sizeof(int), 0, cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nz, &nz, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_nx, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ny, &ny, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nVel, &host_nVel, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_yStride, &host_yStride, sizeof(long long), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nModelExt, &host_nModelExt, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));

	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

    // Extension parameters
	cuda_call(cudaMemcpyToSymbol(dev_nExt1, &host_nExt1, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nExt2, &host_nExt2, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_hExt1, &host_hExt1, sizeof(int), 0, cudaMemcpyHostToDevice));
    cuda_call(cudaMemcpyToSymbol(dev_hExt2, &host_hExt2, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_extStride, &host_extStride, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));

	// Second order time derivative coefficients
	cuda_call(cudaMemcpyToSymbol(dev_cCenter, &host_cCenter, sizeof(double), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_cSide, &host_cSide, sizeof(double), 0, cudaMemcpyHostToDevice));

}

/* Allocation on device */
void allocateTomoExtShotsGpu_3D(double *vel2Dtw2, double *reflectivityScale, double *extReflectivity, int iGpu, int iGpuId){

	// Get GPU number
	cudaSetDevice(iGpuId);

	// Scaled velocity
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nVel*sizeof(double))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nVel*sizeof(double), cudaMemcpyHostToDevice));

    // Reflectivity scale
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nVel*sizeof(double))); // Allocate scaling for reflectivity
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nVel*sizeof(double), cudaMemcpyHostToDevice)); //

	// Allocate time slices on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nVel*sizeof(double))); // Allocate time slices on device (for the stepper)
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nVel*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pLeft[iGpu], host_nVel*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_pRight[iGpu], host_nVel*sizeof(double)));

	// Allocate time slices on device for second time derivative of source wavefield
	cuda_call(cudaMalloc((void**) &dev_pDt0[iGpu], host_nVel*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pDt1[iGpu], host_nVel*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pDt2[iGpu], host_nVel*sizeof(double)));

    // Reflectivity model
    cuda_call(cudaMalloc((void**) &dev_modelTomo[iGpu], host_nVel*sizeof(double)));

	// Allocate pinned memory on host
	cuda_call(cudaHostAlloc((void**) &pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaHostAllocDefault));
	cuda_call(cudaHostAlloc((void**) &pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double), cudaHostAllocDefault));

	// Allocate the slice where we store the wavefield slice before transfering it to the host's pinned memory
	cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], host_nVel*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nVel*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pRecWavefield[iGpu], host_nVel*sizeof(double)));

	// Allocate and copy from host to device extended reflectivity
	cuda_call(cudaMalloc((void**) &dev_extReflectivity[iGpu], host_nModelExt*sizeof(double)));
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt*sizeof(double), cudaMemcpyHostToDevice));

}

/* Deallocation on device */
void deallocateTomoExtShotsGpu_3D(int iGpu, int iGpuId){

	cudaSetDevice(iGpuId);
	cuda_call(cudaFree(dev_vel2Dtw2[iGpu]));
    cuda_call(cudaFree(dev_reflectivityScale[iGpu]));
	cuda_call(cudaFree(dev_extReflectivity[iGpu]));
	cuda_call(cudaFree(dev_p0[iGpu]));
	cuda_call(cudaFree(dev_p1[iGpu]));
    cuda_call(cudaFree(dev_pLeft[iGpu]));
    cuda_call(cudaFree(dev_pRight[iGpu]));
	cuda_call(cudaFree(dev_pDt0[iGpu]));
	cuda_call(cudaFree(dev_pDt1[iGpu]));
	cuda_call(cudaFree(dev_pDt2[iGpu]));
    cuda_call(cudaFree(dev_pSourceWavefield[iGpu]));
	cuda_call(cudaFree(dev_pRecWavefield[iGpu]));
	cuda_call(cudaFree(dev_modelTomo[iGpu]));
	cuda_call(cudaFreeHost(pin_wavefieldSlice1[iGpu]));
	cuda_call(cudaFreeHost(pin_wavefieldSlice2[iGpu]));
}

/******************************************************************************/
/************************* Tomo extended forward ******************************/
/******************************************************************************/
// Subsurface offsets
void tomoHxHyShotsFwdGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStreamH2D[iGpu]);
	cudaStreamCreate(&transferStreamD2H[iGpu]);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Sources geometry + signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2_3D(dev_sourcesSignals[iGpu], wavefield1, dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Scale model (background perturbation) by 2/v^3 x v^2dtw^2
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	for (int iExt2=0; iExt2<host_nExt2; iExt2++){
		long long extStride2 = iExt2 * host_extStride;
		for (int iExt1=0; iExt1<host_nExt1; iExt1++){
			long long extStride1 = iExt1 * host_nVel;
			scaleReflectivityLinHxHy_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2);
		}
	}

	// Allocate and initialize data to zero
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	if (host_leg1 == 1){
		std::cout << "Leg 1 fwd" << std::endl;
		// Source -> reflectivity -> model -> data
		computeTomoLeg1HxHyFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){
		std::cout << "Leg 2 fwd" << std::endl;
		// Source -> model -> reflectivity -> data
		computeTomoLeg2HxHyFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);

	}

	/**************************************************************************/
	/******************************** Data ************************************/
	/**************************************************************************/
	// Copy data to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Time-lags
void tomoTauShotsFwdGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStreamH2D[iGpu]);
	cudaStreamCreate(&transferStreamD2H[iGpu]);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Sources geometry + signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2_3D(dev_sourcesSignals[iGpu], wavefield1, dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Scale model (background perturbation) by 2/v^3 x v^2dtw^2
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	for (int iExt2=0; iExt2<host_nExt2; iExt2++){
		long long extStride2 = iExt2 * host_extStride;
		for (int iExt1=0; iExt1<host_nExt1; iExt1++){
			long long extStride1 = iExt1 * host_nVel;
			scaleReflectivityLinHxHy_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2);
		}
	}

	// Allocate and initialize data to zero
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	if (host_leg1 == 1){
		std::cout << "Leg 1 fwd" << std::endl;
		// Source -> reflectivity -> model -> data
		// computeTomoLeg1TauFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){
		std::cout << "Leg 2 fwd" << std::endl;
		// Source -> model -> reflectivity -> data
		// computeTomoLeg2TauFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);

	}

	/**************************************************************************/
	/******************************** Data ************************************/
	/**************************************************************************/
	// Copy data to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

/******************************************************************************/
/************************* Tomo extended adjoint ******************************/
/******************************************************************************/
// Subsurface offsets
void tomoHxHyShotsAdjGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId, double *dataRegDtsQc){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStreamH2D[iGpu]);
	cudaStreamCreate(&transferStreamD2H[iGpu]);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/

	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2_3D(dev_sourcesSignals[iGpu], wavefield1, dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice));

  	cuda_call(cudaMalloc((void**) &dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(double)));
	cuda_call(cudaMemset(dev_dataRegDtsQc[iGpu], 0, nReceiversReg*host_nts*sizeof(double)));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoRecWfld_3D(dev_dataRegDts[iGpu], wavefield2, dev_receiversPositionReg[iGpu], dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Set model to zero
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nVel*sizeof(double)));

	// Scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	for (int iExt2=0; iExt2<host_nExt2; iExt2++){
		long long extStride2 = iExt2 * host_extStride;
		for (int iExt1=0; iExt1<host_nExt1; iExt1++){
			long long extStride1 = iExt1 * host_nVel;
			scaleReflectivityLinHxHy_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2);
		}
	}

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	if (host_leg1 == 1){
		std::cout << "Leg 1 adj" << std::endl;
		// Source -> reflectivity -> model -> data
		computeTomoLeg1HxHyAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){
		std::cout << "Leg 2 adj" << std::endl;
		// Source -> reflectivity -> model -> data
		computeTomoLeg2HxHyAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);

	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	// QC
	cuda_call(cudaMemcpy(dataRegDtsQc, dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Time-lags
void tomoTauShotsAdjGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId, double *dataRegDtsQc){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStreamH2D[iGpu]);
	cudaStreamCreate(&transferStreamD2H[iGpu]);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/

	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2_3D(dev_sourcesSignals[iGpu], wavefield1, dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice));

  	cuda_call(cudaMalloc((void**) &dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(double)));
	cuda_call(cudaMemset(dev_dataRegDtsQc[iGpu], 0, nReceiversReg*host_nts*sizeof(double)));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoRecWfld_3D(dev_dataRegDts[iGpu], wavefield2, dev_receiversPositionReg[iGpu], dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Set model to zero
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nVel*sizeof(double)));

	// Scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	for (int iExt2=0; iExt2<host_nExt2; iExt2++){
		long long extStride2 = iExt2 * host_extStride;
		for (int iExt1=0; iExt1<host_nExt1; iExt1++){
			long long extStride1 = iExt1 * host_nVel;
			scaleReflectivityLinHxHy_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2);
		}
	}

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	if (host_leg1 == 1){
		std::cout << "Leg 1 adj" << std::endl;
		// Source -> reflectivity -> model -> data
		computeTomoLeg1HxHyAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){
		std::cout << "Leg 2 adj" << std::endl;
		// Source -> reflectivity -> model -> data
		computeTomoLeg2HxHyAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);

	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	// QC
	cuda_call(cudaMemcpy(dataRegDtsQc, dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}
/******************************************************************************/
/***************************** Auxiliary functions ****************************/
/******************************************************************************/

/***************************** Common parts ***********************************/
// Source wavefield with an additional second time derivative
void computeTomoSrcWfldDt2_3D(double *dev_sourcesIn, double *wavefield1, long long *dev_sourcesPositionsRegIn, int nSourcesRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn){

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize pinned memory
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(double));

	double *dummySliceLeft, *dummySliceRight;
	dummySliceLeft = new double[host_nVel];
	dummySliceRight = new double[host_nVel];

	// Compute coarse source wavefield sample at its = 0
	int its = 0;

	// Loop within two values of its (coarse time grid)
	for (int it2 = 1; it2 < host_sub+1; it2++){

		// Compute fine time-step index
		int itw = its * host_sub + it2;

		// Step forward
		stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

		// Inject source
		injectSourceLinear_3D<<<1, nSourcesRegIn, 0, compStreamIn>>>(dev_sourcesIn, dev_p0[iGpu], itw-1, dev_sourcesPositionsRegIn);

		// Damp wavefields
		dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

		// Spread energy to dev_pLeft and dev_pRight
		interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

		// Switch pointers
		dev_temp1[iGpu] = dev_p0[iGpu];
		dev_p0[iGpu] = dev_p1[iGpu];
		dev_p1[iGpu] = dev_temp1[iGpu];
		dev_temp1[iGpu] = NULL;
	}

	// Copy pDt1 (its=0)
	cuda_call(cudaMemcpyAsync(dev_pDt1[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

	// Switch pointers
	dev_pTemp[iGpu] = dev_pLeft[iGpu];
	dev_pLeft[iGpu] = dev_pRight[iGpu];
	dev_pRight[iGpu] = dev_pTemp[iGpu];
	dev_pTemp[iGpu] = NULL;
	cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStreamIn));

	/************************** Main loop (its > 0) ***************************/
	for (int its = 1; its < host_nts-1; its++){

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesRegIn, 0, compStreamIn>>>(dev_sourcesIn, dev_p0[iGpu], itw-1, dev_sourcesPositionsRegIn);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pLeft and dev_pRight
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Copy source wavefield value at its into pDt2
		cuda_call(cudaMemcpyAsync(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		// Compute second-order time-derivative of source wavefield at its-1
	    srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu]);
		cuda_call(cudaStreamSynchronize(compStreamIn));
		cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

		// Wait for pStream to be free
		cuda_call(cudaStreamSynchronize(transferStreamIn));
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pSourceWavefield[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its > 1){
			std::memcpy(wavefield1+(its-2)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double));
		}

		//// WHY DO YOU NEED THAT ONE ??? ////
		cuda_call(cudaStreamSynchronize(compStreamIn));

		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStreamIn));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStreamIn));

		// Switch pointers for time derivative
		dev_pDtTemp[iGpu] = dev_pDt0[iGpu];
		dev_pDt0[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;

	}

	// Copy source wavefield at nts-1 into pDt2
	cuda_call(cudaMemcpyAsync(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute second order time derivative of source wavefield at nts-2
	srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu]);

	// Wait until pStream has been transfered to host
	cuda_call(cudaStreamSynchronize(transferStreamIn));

	// Copy dev_pSourceWavefield into pStream
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pSourceWavefield[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy second order time derivative of source wavefield at nts-3 from pin -> RAM
	std::memcpy(wavefield1+(host_nts-3)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double));

	// Transfer pStream (second order time derivative of source wavefield at nts-2) to pin
	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStreamIn));

	// In the meantime, reset pDt0 and compute second order time-derivative at nts-1
	cuda_call(cudaMemsetAsync(dev_pDt0[iGpu], 0, host_nVel*sizeof(double), compStreamIn));
	srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu], dev_pDt0[iGpu]);

	// Wait until pStream has been fully transfered to pin (derivative of source wavefield at nts-2)
	cuda_call(cudaStreamSynchronize(transferStreamIn));

	// Copy source derivative from pin -> RAM
	std::memcpy(wavefield1+(host_nts-2)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double));

	// Copy source derivative at nts-1
	cuda_call(cudaMemcpy(pin_wavefieldSlice1[iGpu], dev_pSourceWavefield[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));
	std::memcpy(wavefield1+(host_nts-1)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double));

}

// Receiver wavefield
void computeTomoRecWfld_3D(double *dev_dataRegDtsIn, double *wavefield2, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn){

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize pinned memory
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(double));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject data
			interpLinearInjectData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_dataRegDtsIn, dev_p0[iGpu], its, it2, dev_receiversPositionRegIn);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Wait until pStream has been transfered
		cuda_call(cudaStreamSynchronize(transferStreamIn));

		// Copy pRight (contains wavefield at its+1) into pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy wavefield at its+1 from pin -> RAM
		if (its < host_nts-2) {
			std::memcpy(wavefield2+(its+2)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double));

		}

		// Wait until pStream has been updated
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Transfer pStream -> pin (at its+1)
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStreamIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel*sizeof(double), compStreamIn));

	}

	// At this point, pin contains receiver wavefield at its=1

	// Wait until pStream has been transfered
	cuda_call(cudaStreamSynchronize(transferStreamIn));

 	// Copy wavefield at its=1 from pin -> RAM
	std::memcpy(wavefield2+host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double));

	// Transfer pStream -> pin (at its=0)
	cuda_call(cudaMemcpy(pin_wavefieldSlice1[iGpu], dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy wavefield at its=0 from pin -> RAM
	std::memcpy(wavefield2, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double));

}

/******************************** Forward *************************************/
// Source -> reflectivity -> model -> data
void computeTomoLeg1HxHyFwd_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_dataRegDtsIn, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #1 *************************/
	/*************** Source -> reflectivity -> scattered wavefield ************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(double));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(double));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield1 slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(double));
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Launch transfer of wavefield2 slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2);

			// Extract data
			// recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Wait until the scattered wavefield has been transfered to pin
		cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

		// Asynchronous copy of dev_pDt1 => dev_pDt0 [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));
		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(double), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Transfer pDt0 -> pin
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));

	/**************************************************************************/
	/*************************** Second part of leg #1 ************************/
	/***************** Scattered wavefield -> model -> data *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// // Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu],wavefield2+(its+2)*host_nVel, host_nVel*sizeof(double));
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionRegIn);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}
}

// Source -> model -> reflectivity -> data
void computeTomoLeg2HxHyFwd_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_dataRegDtsIn, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #2 *************************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(double));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(double));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+1)*host_nVel, host_nVel*sizeof(double));
		cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2);

			// Extract data
			// recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Asynchronous copy of dev_pDt1 => dev_pDt0 (scattered wavefield at its)
		cuda_call(cudaMemcpy(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
		cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));
		std::memcpy(wavefield2+its*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));

	}

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice));
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));

	/**************************************************************************/
	/*************************** Second part of leg #2 ************************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu]);

	// imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);
	// scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu]);


	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(its+1)*host_nVel, host_nVel*sizeof(double));
		cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		// imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
		// scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);


		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionRegIn);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	}
}

/******************************** Adjoint *************************************/
// Source -> reflectivity -> model <- data
void computeTomoLeg1HxHyAdj_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, double *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRecWavefield[iGpu], 0, host_nVel*sizeof(double)));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu]);
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Copy source wavefield slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Copy receiver wavefield slice from RAM -> pinned for time its = 0 -> transfer to pDt0
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy source wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){

			// Copy wavefield1 slice its+2 from RAM -> pin
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(double));

			// Wait until dev_pStream is ready to be used
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Load wavefield slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Copy wavefield2 slice its+1 from RAM -> pin
		std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(its+1)*host_nVel, host_nVel*sizeof(double));

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its+1
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2);

			// Extract data
			// recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsQcIn, its, it2, dev_receiversPositionRegIn);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// At this point, pDt1 contains the value of the scattered wavefield at its
		// The imaging condition can be done for its

		// Apply imaging condition at its
		imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu]);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));

	}

	// Copy receiver wavefield value at nts-1 from pDt0 -> pRecWavefield
	cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute imaging condition at its = nts-1
	imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu]);

	// Scale model for finite-difference and secondary source coefficient
	// scaleReflectivity_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

}

// Source -> reflectivity -> model <- data
void computeTomoLeg2HxHyAdj_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, double *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRecWavefield[iGpu], 0, host_nVel*sizeof(double)));

	// Copy receiver wavefield time-slice its = nts-1
	// From RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(host_nts-1)*host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRecWavefield[iGpu], dev_vel2Dtw2[iGpu]);

	// Compute secondary source for its = nts-1
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_pRecWavefield[iGpu], dev_extReflectivityIn, ihx, iExt1, ihy, iExt2);
		}
	}

	// Copy receiver wavefield slice from RAM -> pinned for time nts-2 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(host_nts-2)*host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Copy source wavefield slice from RAM -> pinned for time its = nts-1 -> transfer to pDt0
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(host_nts-1)*host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy receiver wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its > 0){

			// Copy receiver wavefield slice its-1 from RAM -> pin
			std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(its-1)*host_nVel, host_nVel*sizeof(double));

			// Wait until dev_pStream is ready to be used
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Load receiver wavefield slice its-1 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Copy source wavefield slice its from RAM -> pin
		std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+its*host_nVel, host_nVel*sizeof(double));

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRecWavefield[iGpu], dev_vel2Dtw2[iGpu]);

		// Compute secondary source for its = nts-1
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRecWavefield[iGpu], dev_extReflectivityIn, ihx, iExt1, ihy, iExt2);
			}
		}

		// Start subloop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step forward
			stepAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2+1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply imaging condition at its+1
		imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu]);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(double)));

	}

	// Copy receiver wavefield value at its = 0 from pStream -> pSourceWavefield
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute imaging condition at its = 0
	imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu]);

}
