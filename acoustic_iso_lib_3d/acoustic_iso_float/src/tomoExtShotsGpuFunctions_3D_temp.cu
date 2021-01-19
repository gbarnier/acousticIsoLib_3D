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

/**************************** No free surface *********************************/

// Note: The implementations of these auxiliary functions are done at the bottom of the file
void computeTomoSrcWfldDt2_3D(float *dev_sourcesIn, float *wavefield1, long long *dev_sourcesPositionsRegIn, int nSourcesRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn);

void computeTomoRecWfld_3D(float *dev_dataRegDtsIn, float *wavefield2, long long *dev_receiversPositionsRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn);

// Forward + Subsurface offsets
void computeTomoLeg1HxHyFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

void computeTomoLeg2HxHyFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

// Forward + Time-lags
void computeTomoLeg1TauFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

void computeTomoLeg2TauFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

// Adjoint + Subsurface offsets
void computeTomoLeg1HxHyAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn);

void computeTomoLeg2HxHyAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn);

// Adjoint + Time-lags
void computeTomoLeg1TauAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn);

void computeTomoLeg2TauAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn);

/******************************* Free surface *********************************/
void computeTomoFsSrcWfldDt2_3D(float *dev_sourcesIn, float *wavefield1, long long *dev_sourcesPositionsRegIn, int nSourcesRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn);

void computeTomoFsRecWfld_3D(float *dev_dataRegDtsIn, float *wavefield2, long long *dev_receiversPositionsRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn);

// Forward + Subsurface offsets
void computeTomoFsLeg1HxHyFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

void computeTomoFsLeg2HxHyFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

// Forward + Time-lags
void computeTomoFsLeg1TauFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

void computeTomoFsLeg2TauFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int nblockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn);

// Adjoint + Subsurface offsets
void computeTomoFsLeg1HxHyAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn);

void computeTomoFsLeg2HxHyAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn);

// Adjoint + Time-lags
void computeTomoFsLeg1TauAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn);

void computeTomoFsLeg2TauAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn);

/******************************************************************************/
/**************************** Initialization **********************************/
/******************************************************************************/
// Parameter settings
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

// Initialize GPU
void initTomoExtGpu_3D(float dz, float dx, float dy, int nz, int nx, int ny, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, std::string extension, int nExt1, int nExt2, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc){

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
	host_extension = extension;
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
	// host_cSide = 0.0;
	// host_cCenter = 1.0;

	/**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {

		// Time slices for FD stepping
		dev_p0 = new float*[nGpu];
		dev_p1 = new float*[nGpu];
		dev_temp1 = new float*[nGpu];

		dev_pLeft = new float*[nGpu];
		dev_pRight = new float*[nGpu];
		dev_pTemp = new float*[nGpu];

		dev_pDt0 = new float*[nGpu];
		dev_pDt1 = new float*[nGpu];
		dev_pDt2 = new float*[nGpu];
		dev_pDtTemp = new float*[nGpu];

		dev_pRecWavefield = new float*[nGpu];

		// Data and model
		dev_dataRegDts = new float*[nGpu];
		dev_dataRegDtsQc = new float*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new long long*[nGpu];
		dev_receiversPositionReg = new long long*[nGpu];

        // Sources signal
		dev_sourcesSignals = new float*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new float*[nGpu];

        // Reflectivity scaling
		dev_reflectivityScale = new float*[nGpu];

        // Background perturbation ("model" for tomo)
		dev_modelTomo = new float*[nGpu];

		// Extended reflectivity for tomo
		dev_extReflectivity = new float*[nGpu];

		// Debug model and data
		dev_modelDebug = new float*[nGpu];
		dev_dataDebug = new float*[nGpu];

        // Streams
		compStream = new cudaStream_t[nGpu];
		transferStream = new cudaStream_t[nGpu];
		transferStreamH2D = new cudaStream_t[nGpu];
		transferStreamD2H = new cudaStream_t[nGpu];
		pin_wavefieldSlice1 = new float*[nGpu];
		pin_wavefieldSlice2 = new float*[nGpu];
		dev_pStream = new float*[nGpu];

		// if (host_extension == "offset"){
		dev_pSourceWavefield = new float*[nGpu];
		// }

		// Time-lags
		if (host_extension == "time"){
			dev_pSourceWavefieldTau = new float**[nGpu];
			for (int iGpu=0; iGpu<nGpu; iGpu++){
				dev_pSourceWavefieldTau[iGpu] = new float*[4*host_hExt1+1];
			}
			dev_pTempTau = new float*[nGpu];
		}
	}
	/**************************** COMPUTE LAPLACIAN COEFFICIENTS ************************/
	float host_coeff[COEFF_SIZE] = get_coeffs((float)dz,(float)dx,(float)dy); // Stored on host

	/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
	// Time interpolation filter length / half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		throw std::runtime_error("**** ERROR [tomoExtShotsGpuFunctions_3D]: Subsampling parameter for time interpolation is too high ****");
	}

	// Allocate and fill time interpolation filter
	float interpFilter[nInterpFilter];
	for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
		interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
		interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
		interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(float(host_ntw)/float(host_nts)));
		interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(float(host_ntw)/float(host_nts)));
	}

	/************************* COMPUTE COSINE DAMPING COEFFICIENTS **********************/
	if (minPad>=PAD_MAX){
		throw std::runtime_error("**** ERROR [tomoExtShotsGpuFunctions_3D]: Padding value is too high ****");
	}
	float cosDampingCoeff[minPad];

	// Cosine padding
	for (int iFilter=FAT; iFilter<FAT+minPad; iFilter++){
		float arg = M_PI / (1.0 * minPad) * 1.0 * (minPad-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		cosDampingCoeff[iFilter-FAT] = arg;
	}

	// Check that the block size is consistent between parfile and "varDeclare.h"
	if (blockSize != BLOCK_SIZE) {
		throw std::runtime_error("**** ERROR [tomoExtShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare file ****");
	}

	/**************************** COPY TO CONSTANT MEMORY *******************************/
	// Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_coeff, host_coeff, COEFF_SIZE*sizeof(float), 0, cudaMemcpyHostToDevice)); // Copy Laplacian coefficients to device

	// Time interpolation filter
	cuda_call(cudaMemcpyToSymbol(dev_nTimeInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
	cuda_call(cudaMemcpyToSymbol(dev_hTimeInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
	cuda_call(cudaMemcpyToSymbol(dev_timeInterpFilter, interpFilter, nInterpFilter*sizeof(float), 0, cudaMemcpyHostToDevice)); // Filter

	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeff, &cosDampingCoeff, minPad*sizeof(float), 0, cudaMemcpyHostToDevice)); // Array for damping
	cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(float), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula
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
	cuda_call(cudaMemcpyToSymbol(dev_cCenter, &host_cCenter, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_cSide, &host_cSide, sizeof(float), 0, cudaMemcpyHostToDevice));

}

// Allocation on device (no Ginsu)
void allocateTomoExtShotsGpu_3D(float *vel2Dtw2, float *reflectivityScale, float *extReflectivity, int iGpu, int iGpuId){

	// Get GPU number
	cudaSetDevice(iGpuId);

	// Scaled velocity
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nVel*sizeof(float))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nVel*sizeof(float), cudaMemcpyHostToDevice));

    // Reflectivity scale
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nVel*sizeof(float))); // Allocate scaling for reflectivity
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Allocate time slices on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nVel*sizeof(float))); // Allocate time slices on device (for the stepper)
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nVel*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_pLeft[iGpu], host_nVel*sizeof(float)));
    cuda_call(cudaMalloc((void**) &dev_pRight[iGpu], host_nVel*sizeof(float)));

	// Allocate time slices on device for second time derivative of source wavefield
	cuda_call(cudaMalloc((void**) &dev_pDt0[iGpu], host_nVel*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_pDt1[iGpu], host_nVel*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_pDt2[iGpu], host_nVel*sizeof(float)));

    // Reflectivity model
    cuda_call(cudaMalloc((void**) &dev_modelTomo[iGpu], host_nVel*sizeof(float)));

	// Allocate the slice where we store the wavefield slice before transfering it to the host's pinned memory
	cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], host_nVel*sizeof(float)));
	// cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nVel*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_pRecWavefield[iGpu], host_nVel*sizeof(float)));

	// Allocate and copy from host to device extended reflectivity
	cuda_call(cudaMalloc((void**) &dev_extReflectivity[iGpu], host_nModelExt*sizeof(float)));
	// cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt*sizeof(float), cudaMemcpyHostToDevice));

	// if (host_extension == "offset"){
	cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nVel*sizeof(float)));
	// }

	// Allocate the arrays that will contain the source wavefield slices for time-lags
	if (host_extension == "time"){
		for (int iExt=0; iExt<4*host_hExt1+1; iExt++){
			cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(float)));
		}
	}
}

// Allocate pinned normal
void allocatePinnedTomoExtGpu_3D(int nzWavefield, int nxWavefield, int nyWavefield, int ntsWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	// Get GPU number
	cudaSetDevice(iGpuId);

	host_nWavefieldSpace = nzWavefield * nxWavefield * nyWavefield;

	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {
		pin_wavefieldSlice1 = new float*[nGpu];
		pin_wavefieldSlice2 = new float*[nGpu];
	}
	// Allocate pinned memory on host
	cuda_call(cudaHostAlloc((void**) &pin_wavefieldSlice1[iGpu], host_nWavefieldSpace*ntsWavefield*sizeof(float), cudaHostAllocDefault));
	cuda_call(cudaHostAlloc((void**) &pin_wavefieldSlice2[iGpu], host_nWavefieldSpace*ntsWavefield*sizeof(float), cudaHostAllocDefault));
}

// Init Ginsu
void initTomoExtGinsuGpu_3D(float dz, float dx, float dy, int nts, float dts, int sub, int blockSize, float alphaCos, std::string extension, int nExt1, int nExt2, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nts = nts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;
	host_extension = extension;
	host_nExt1 = nExt1;
	host_nExt2 = nExt2;
	host_hExt1 = (nExt1-1)/2;
	host_hExt2 = (nExt2-1)/2;

	/********************* ALLOCATE ARRAYS OF ARRAYS **************************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {

		// Time slices for FD stepping
		dev_p0 = new float*[nGpu];
		dev_p1 = new float*[nGpu];
		dev_temp1 = new float*[nGpu];

		dev_pLeft = new float*[nGpu];
		dev_pRight = new float*[nGpu];
		dev_pTemp = new float*[nGpu];

		// Data and model
		dev_dataRegDts = new float*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new long long*[nGpu];
		dev_receiversPositionReg = new long long*[nGpu];

        // Sources signal
		dev_sourcesSignals = new float*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new float*[nGpu];

        // Reflectivity scaling
		dev_reflectivityScale = new float*[nGpu];

        // Reflectivity
		dev_modelBornExt = new float*[nGpu];

		// Debug model and data
		dev_modelDebug = new float*[nGpu];
		dev_dataDebug = new float*[nGpu];

        // Streams
		compStream = new cudaStream_t[nGpu];
		transferStream = new cudaStream_t[nGpu];
		transferStreamH2D = new cudaStream_t[nGpu];
		dev_pStream = new float*[nGpu];

		// Subsurface offsets
		if (host_extension == "offset"){
			dev_pSourceWavefield = new float*[nGpu];
		}

		// Time-lags
		if (host_extension == "time"){
			dev_pSourceWavefieldTau = new float**[nGpu];
			for (int iGpu=0; iGpu<nGpu; iGpu++){
				std::cout << "Allocating wavefield slices Ginsu" << std::endl;
				dev_pSourceWavefieldTau[iGpu] = new float*[4*host_hExt1+1];
			}
			dev_pTempTau = new float*[nGpu];
		}

	}

	/**************************** COMPUTE LAPLACIAN COEFFICIENTS ************************/
	float host_coeff[COEFF_SIZE] = get_coeffs((float)dz,(float)dx,(float)dy); // Stored on host

	/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
	// Time interpolation filter length / half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		std::cout << "**** ERROR [BornShotsGpuFunctions_3D]: Subsampling parameter for time interpolation is too high ****" << std::endl;
		throw std::runtime_error("");
	}

	// Allocate and fill time interpolation filter
	float interpFilter[nInterpFilter];
	for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
		interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
		interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
		interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(float(host_ntw)/float(host_nts)));
		interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(float(host_ntw)/float(host_nts)));
	}

	// Check that the block size is consistent between parfile and "varDeclare.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [BornShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare file ****" << std::endl;
		throw std::runtime_error("");
	}

	/**************************** COPY TO CONSTANT MEMORY *******************************/
	// Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_coeff, host_coeff, COEFF_SIZE*sizeof(float), 0, cudaMemcpyHostToDevice)); // Copy Laplacian coefficients to device

	// Time interpolation filter
	cuda_call(cudaMemcpyToSymbol(dev_nTimeInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
	cuda_call(cudaMemcpyToSymbol(dev_hTimeInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
	cuda_call(cudaMemcpyToSymbol(dev_timeInterpFilter, interpFilter, nInterpFilter*sizeof(float), 0, cudaMemcpyHostToDevice)); // Filter

	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(float), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

}

// Allocate Ginsu
void allocateSetTomoExtGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, float alphaCos, float *vel2Dtw2, float *reflectivityScale, int iGpu, int iGpuId){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nz_ginsu[iGpu] = nz;
	host_nx_ginsu[iGpu] = nx;
    host_ny_ginsu[iGpu] = ny;
	host_yStride_ginsu[iGpu] = nz * nx;
	host_nVel_ginsu[iGpu] = nz * nx * ny;
	host_nModelExt_ginsu[iGpu] = host_nVel_ginsu[iGpu]*host_nExt1*host_nExt2;
	host_minPad_ginsu[iGpu] = minPad;
	host_extStride_ginsu[iGpu] = host_nExt1 * host_nVel_ginsu[iGpu];

	/******************** COMPUTE COSINE DAMPING COEFFICIENTS *****************/
	if (minPad>=PAD_MAX){
		std::cout << "**** ERROR [tomoExtShotsGpuFunctions_3D]: Padding value is too high ****" << std::endl;
		throw std::runtime_error("");
	}

	// Allocate array to store damping coefficients on host
	float host_cosDampingCoeffGinsuTemp[host_minPad_ginsu[iGpu]];

	// Compute array coefficients
	for (int iFilter=FAT; iFilter<FAT+host_minPad_ginsu[iGpu]; iFilter++){
		float arg = M_PI / (1.0 * host_minPad_ginsu[iGpu]) * 1.0 * (host_minPad_ginsu[iGpu]-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		host_cosDampingCoeffGinsuTemp[iFilter-FAT] = arg;
	}

	// Check that the block size is consistent between parfile and "varDeclare_3D.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [tomoExtShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare.h file ****" << std::endl;
		throw std::runtime_error("");
	}

	/********************** COPY TO CONSTANT MEMORY ***************************/
	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeffGinsuConstant, &host_cosDampingCoeffGinsuTemp, host_minPad_ginsu[iGpu]*sizeof(float), iGpu*PAD_MAX*sizeof(float), cudaMemcpyHostToDevice)); // Array for damping
	cuda_call(cudaMemcpyToSymbol(dev_minPad_ginsu, &host_minPad_ginsu[iGpu], sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nz_ginsu, &nz, sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_nx_ginsu, &nx, sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_ny_ginsu, &ny, sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // Copy model size to device

	cuda_call(cudaMemcpyToSymbol(dev_nExt1, &host_nExt1, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nExt2, &host_nExt2, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_hExt1, &host_hExt1, sizeof(int), 0, cudaMemcpyHostToDevice));
    cuda_call(cudaMemcpyToSymbol(dev_hExt2, &host_hExt2, sizeof(int), 0, cudaMemcpyHostToDevice));

	cuda_call(cudaMemcpyToSymbol(dev_yStride_ginsu, &host_yStride_ginsu[iGpu], sizeof(long long), iGpu*sizeof(long long), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nVel_ginsu, &host_nVel_ginsu[iGpu], sizeof(unsigned long long), iGpu*sizeof(unsigned long long), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nModelExt_ginsu, &host_nModelExt_ginsu[iGpu], sizeof(unsigned long long), iGpu*sizeof(unsigned long long), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_extStride_ginsu, &host_extStride_ginsu[iGpu], sizeof(long long), iGpu*sizeof(long long), cudaMemcpyHostToDevice));

	// Allocate and copy scaled velocity model to device
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Allocate time slices on device for the FD stepping
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
    cuda_call(cudaMalloc((void**) &dev_pRight[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));

	// Allocate time slices on device for the FD stepping
	cuda_call(cudaMalloc((void**) &dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_pDt2[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));

	// Allocate model on device
	cuda_call(cudaMalloc((void**) &dev_modelTomo[iGpu], host_nVel*sizeof(float)));

	// Reflectivity scaling
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

    // Reflectivity model
    cuda_call(cudaMalloc((void**) &dev_modelBornExt[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(float)));

	// Allocate the slice where we store the wavefield slice before transfering it to the host's pinned memory
	cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_pRecWavefield[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));

	// Allocate and copy from host to device extended reflectivity
	cuda_call(cudaMalloc((void**) &dev_extReflectivity[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(float)));

	// Allocate the arrays that will contain the source wavefield slices for time-lags
	if (host_extension == "time"){
		for (int iExt=0; iExt<4*host_hExt1+1; iExt++){
			// std::cout << "iExt = " << iExt << std::endl;
			cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel_ginsu[iGpu]*sizeof(float)));
		}
	}
	if (host_extension == "offset"){
		cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	}
}

// Deallocation on device
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
	cuda_call(cudaFree(dev_pRecWavefield[iGpu]));
	cuda_call(cudaFree(dev_modelTomo[iGpu]));
	cuda_call(cudaFree(dev_pStream[iGpu]));

	// Deallocate the arrays that will contain the source wavefield slices for time-lags
	if (host_extension == "time"){
		for (int iExt=0; iExt<4*host_hExt1+1; iExt++){
			cuda_call(cudaFree(dev_pSourceWavefieldTau[iGpu][iExt]));
		}
	}
	if (host_extension == "offset"){
		cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nVel*sizeof(float)));
	}
}

// Deallocate pinned memory
void deallocatePinnedTomoExtShotsGpu_3D(int iGpu, int iGpuId){
	// Set GPU number
	cudaSetDevice(iGpuId);
	cuda_call(cudaFreeHost(pin_wavefieldSlice1[iGpu]));
	cuda_call(cudaFreeHost(pin_wavefieldSlice2[iGpu]));
}

/******************************************************************************/
/************************* Tomo extended forward ******************************/
/******************************************************************************/
// Time-lags
void tomoTauShotsFwdGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

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
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Scale model (background perturbation) by 2/v^3 x v^2dtw^2
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Scale = 2.0 * 1/v^3 * v^2 * dtw^2
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt=0; iExt<host_nExt1; iExt++){
		long long extStride = iExt * host_nVel;
		scaleReflectivityTau_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride);
	}

	// Allocate and initialize data to zero
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	if (host_leg1 == 1){

		// std::cout << "Leg 1 fwd time" << std::endl;

		// Source -> reflectivity -> model -> data
		computeTomoLeg1TauFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){

		// std::cout << "Leg 2 fwd time" << std::endl;

		// Source -> model -> reflectivity -> data
		computeTomoLeg2TauFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);

	}

	/**************************************************************************/
	/******************************** Data ************************************/
	/**************************************************************************/
	// Copy data to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Subsurface offsets
void tomoHxHyShotsFwdGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, float *wavefield1, float *wavefield2, int iGpu, int iGpuId){

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
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

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
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Scale model (background perturbation) by 2/v^3 x v^2dtw^2
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy and scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt2=0; iExt2<host_nExt2; iExt2++){
		long long extStride2 = iExt2 * host_extStride;
		for (int iExt1=0; iExt1<host_nExt1; iExt1++){
			long long extStride1 = iExt1 * host_nVel;
			scaleReflectivityLinHxHy_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2);
		}
	}

	// Allocate and initialize data to zero
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	// float *dummyData, *dummyModel, *dummyReflectivity;
	// dummyData = new float[nReceiversReg*host_nts];
	// dummyModel = new float[host_nVel];
	// dummyReflectivity = new float[host_nModelExt];
	// cuda_call(cudaMemcpy(dummyReflectivity, dev_extReflectivity[iGpu], host_nModelExt*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "Min value reflectivity inside = " << *std::min_element(dummyReflectivity,dummyReflectivity+host_nModelExt) << std::endl;
	// std::cout << "Max value reflectivity inside = " << *std::max_element(dummyReflectivity,dummyReflectivity+host_nModelExt) << std::endl;

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	if (host_leg1 == 1){

		// cuda_call(cudaMemcpy(dummyModel, dev_modelTomo[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "Min value model inside = " << *std::min_element(dummyModel,dummyModel+host_nVel) << std::endl;
		// std::cout << "Max value model inside = " << *std::max_element(dummyModel,dummyModel+host_nVel) << std::endl;

		// Source -> reflectivity -> model -> data
		computeTomoLeg1HxHyFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);

		// cuda_call(cudaMemcpy(dummyData, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "Min value data dummy inside = " << *std::min_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
		// std::cout << "Max value data dummt inside = " << *std::max_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;

	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){

		// std::cout << "Leg 2 fwd offset" << std::endl;

		// Source -> model -> reflectivity -> data
		computeTomoLeg2HxHyFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);

	}

	/**************************************************************************/
	/******************************** Data ************************************/
	/**************************************************************************/
	// Copy data to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Subsurface offsets + free surface
void tomoHxHyFsShotsFwdGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, float *wavefield1, float *wavefield2, int iGpu, int iGpuId){

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
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFs(nblocky, nblockz);
	dim3 dimBlockFs(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoFsSrcWfldDt2_3D(dev_sourcesSignals[iGpu], wavefield1, dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGridFs, dimBlockFs, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Scale model (background perturbation) by 2/v^3 x v^2dtw^2
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt2=0; iExt2<host_nExt2; iExt2++){
		long long extStride2 = iExt2 * host_extStride;
		for (int iExt1=0; iExt1<host_nExt1; iExt1++){
			long long extStride1 = iExt1 * host_nVel;
			scaleReflectivityLinHxHy_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2);
		}
	}

	// Allocate and initialize data to zero
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	if (host_leg1 == 1){

		// std::cout << "Leg 1 fwd offset" << std::endl;

		// Source -> reflectivity -> model -> data
		computeTomoFsLeg1HxHyFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGridFs, dimBlockFs, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){

		// std::cout << "Leg 2 fwd offset" << std::endl;

		// Source -> model -> reflectivity -> data
		computeTomoFsLeg2HxHyFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGridFs, dimBlockFs, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);

	}

	/**************************************************************************/
	/******************************** Data ************************************/
	/**************************************************************************/
	// Copy data to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Time-lags + free surface
void tomoTauFsShotsFwdGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, float *wavefield1, float *wavefield2, int iGpu, int iGpuId){

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
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFs(nblocky, nblockz);
	dim3 dimBlockFs(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoFsSrcWfldDt2_3D(dev_sourcesSignals[iGpu], wavefield1, dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGridFs, dimBlockFs, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Scale model (background perturbation) by 2/v^3 x v^2dtw^2
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Scale = 2.0 * 1/v^3 * v^2 * dtw^2
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt=0; iExt<host_nExt1; iExt++){
		long long extStride = iExt * host_nVel;
		scaleReflectivityTau_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride);
	}

	// Allocate and initialize data to zero
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device

  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	if (host_leg1 == 1){

		// std::cout << "Leg 1 fwd time" << std::endl;

		// Source -> reflectivity -> model -> data
		computeTomoFsLeg1TauFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGridFs, dimBlockFs, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){

		// std::cout << "Leg 2 fwd time" << std::endl;

		// Source -> model -> reflectivity -> data
		computeTomoFsLeg2TauFwd_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGridFs, dimBlockFs, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);

	}

	/**************************************************************************/
	/******************************** Data ************************************/
	/**************************************************************************/
	// Copy data to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

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
void tomoHxHyShotsAdjGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, float *wavefield1, float *wavefield2, int iGpu, int iGpuId, float *dataRegDtsQc){

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
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

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
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

  	cuda_call(cudaMalloc((void**) &dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(float)));
	cuda_call(cudaMemset(dev_dataRegDtsQc[iGpu], 0, nReceiversReg*host_nts*sizeof(float)));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoRecWfld_3D(dev_dataRegDts[iGpu], wavefield2, dev_receiversPositionReg[iGpu], dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Set model to zero
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nVel*sizeof(float)));

	// Copy and scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt*sizeof(float), cudaMemcpyHostToDevice));
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

		// Source -> reflectivity -> model -> data
		computeTomoLeg1HxHyAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){

		// Source -> reflectivity -> model -> data
		computeTomoLeg2HxHyAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);

	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// QC
	cuda_call(cudaMemcpy(dataRegDtsQc, dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Subsurface offsets + free surface
void tomoHxHyFsShotsAdjGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, float *wavefield1, float *wavefield2, int iGpu, int iGpuId, float *dataRegDtsQc){

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
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFs(nblocky, nblockz);
	dim3 dimBlockFs(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/

	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoFsSrcWfldDt2_3D(dev_sourcesSignals[iGpu], wavefield1, dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGridFs, dimBlockFs, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

  	cuda_call(cudaMalloc((void**) &dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(float)));
	cuda_call(cudaMemset(dev_dataRegDtsQc[iGpu], 0, nReceiversReg*host_nts*sizeof(float)));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoFsRecWfld_3D(dev_dataRegDts[iGpu], wavefield2, dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGridFs, dimBlockFs, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Set model to zero
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nVel*sizeof(float)));

	// Scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt*sizeof(float), cudaMemcpyHostToDevice));
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

		// std::cout << "Leg 1 adj offset" << std::endl;

		// Source -> reflectivity -> model -> data
		computeTomoFsLeg1HxHyAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGridFs, dimBlockFs, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){

		// std::cout << "Leg 2 adj offset" << std::endl;

		// Source -> reflectivity -> model -> data
		computeTomoFsLeg2HxHyAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGridFs, dimBlockFs, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);

	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// QC
	cuda_call(cudaMemcpy(dataRegDtsQc, dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Time-lags
void tomoTauShotsAdjGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, float *wavefield1, float *wavefield2, int iGpu, int iGpuId, float *dataRegDtsQc){

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
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

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
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

  	cuda_call(cudaMalloc((void**) &dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(float)));
	cuda_call(cudaMemset(dev_dataRegDtsQc[iGpu], 0, nReceiversReg*host_nts*sizeof(float)));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoRecWfld_3D(dev_dataRegDts[iGpu], wavefield2, dev_receiversPositionReg[iGpu], dimGrid, dimBlock, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Set model to zero
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nVel*sizeof(float)));

	// Scale = 2.0 * 1/v^3 * v^2 * dtw^2
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt=0; iExt<host_nExt1; iExt++){
		long long extStride = iExt * host_nVel;
		scaleReflectivityTau_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride);
	}

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	if (host_leg1 == 1){

		// std::cout << "Leg 1 adj time" << std::endl;

		// Source -> reflectivity -> model -> data
		computeTomoLeg1TauAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){

		// std::cout << "Leg 2 adj time" << std::endl;

		// Source -> reflectivity -> model -> data
		computeTomoLeg2TauAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);

	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// QC
	cuda_call(cudaMemcpy(dataRegDtsQc, dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Time-lags + free surface
void tomoTauFsShotsAdjGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, float *wavefield1, float *wavefield2, int iGpu, int iGpuId, float *dataRegDtsQc){

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
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFs(nblocky, nblockz);
	dim3 dimBlockFs(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/

	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoFsSrcWfldDt2_3D(dev_sourcesSignals[iGpu], wavefield1, dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGridFs, dimBlockFs, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

  	cuda_call(cudaMalloc((void**) &dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(float)));
	cuda_call(cudaMemset(dev_dataRegDtsQc[iGpu], 0, nReceiversReg*host_nts*sizeof(float)));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoFsRecWfld_3D(dev_dataRegDts[iGpu], wavefield2, dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGridFs, dimBlockFs, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Set model to zero
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nVel*sizeof(float)));

	// Scale = 2.0 * 1/v^3 * v^2 * dtw^2
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt=0; iExt<host_nExt1; iExt++){
		long long extStride = iExt * host_nVel;
		scaleReflectivityTau_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride);
	}

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	if (host_leg1 == 1){

		// std::cout << "Leg 1 adj time" << std::endl;

		// Source -> reflectivity -> model -> data
		computeTomoFsLeg1TauAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGridFs, dimBlockFs, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	if (host_leg2 == 1){

		// std::cout << "Leg 2 adj time" << std::endl;

		// Source -> reflectivity -> model -> data
		computeTomoFsLeg2TauAdj_3D(dev_modelTomo[iGpu], wavefield1, wavefield2, dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGridFs, dimBlockFs, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData, dev_dataRegDtsQc[iGpu]);

	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// QC
	cuda_call(cudaMemcpy(dataRegDtsQc, dev_dataRegDtsQc[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

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
void computeTomoSrcWfldDt2_3D(float *dev_sourcesIn, float *wavefield1, long long *dev_sourcesPositionsRegIn, int nSourcesRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn){

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float)));

	// float *dummySliceLeft, *dummySliceRight;
	// dummySliceLeft = new float[host_nVel];
	// dummySliceRight = new float[host_nVel];

	// Compute coarse source wavefield sample at its = 0
	int its = 0;

	// Loop within the first two values of its (coarse time grid)
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

	// Copy pDt1 (its = 0)
	cuda_call(cudaMemcpyAsync(dev_pDt1[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Switch pointers
	dev_pTemp[iGpu] = dev_pLeft[iGpu];
	dev_pLeft[iGpu] = dev_pRight[iGpu];
	dev_pRight[iGpu] = dev_pTemp[iGpu];
	dev_pTemp[iGpu] = NULL;
	cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

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
		cuda_call(cudaMemcpyAsync(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Compute second-order time-derivative of source wavefield at its-1
	    srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu]);
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Wait for pStream to be free
		cuda_call(cudaStreamSynchronize(transferStreamIn));

		// Copy second time derivative of source wavefield at its-1 to pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// if (its > 1){
		// 	std::memcpy(wavefield1+(its-2)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));
		// }

		// Wait for pStream to be ready
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Copy second time derivative of source wavefield from device -> pinned memory for time sample its-1
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu]+(its-1)*host_nVel, dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamIn));

		// cuda_call(cudaMemcpy(dummySliceRight, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value source = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
		// std::cout << "Max value source = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for time derivative
		dev_pDtTemp[iGpu] = dev_pDt0[iGpu];
		dev_pDt0[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;

	}

	// Copy source wavefield at nts-1 into pDt2
	cuda_call(cudaMemcpy(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute second-order time-derivative of source wavefield at nts-2
	srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu]);

	// Transfer dev_pSourceWavefield (second-order time-derivative of source wavefield at nts-2) to pinned memory
	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu]+(host_nts-2)*host_nVel, dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, compStreamIn));

	// Reset pDt0 and compute second-order time-derivative at nts-1
	cuda_call(cudaMemsetAsync(dev_pDt0[iGpu], 0, host_nVel*sizeof(float), compStreamIn));
	srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu], dev_pDt0[iGpu]);

	// Transfer dev_pSourceWavefield (second-order time-derivative of source wavefield at nts-1) to pinned memory
	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu]+(host_nts-1)*host_nVel, dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, compStreamIn));

}

// Receiver wavefield
void computeTomoRecWfld_3D(float *dev_dataRegDtsIn, float *wavefield2, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn){

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));

	// Initialize pinned memory
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));

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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pRight[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy wavefield at its+1 from pin -> RAM
		if (its < host_nts-2) {
			std::memcpy(wavefield2+(its+2)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));

		}

		// Wait until pStream has been updated
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Transfer pStream -> pin (at its+1)
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

	}

	// At this point, pin contains receiver wavefield at its=1

	// Wait until pStream has been transfered
	cuda_call(cudaStreamSynchronize(transferStreamIn));

 	// Copy wavefield at its=1 from pin -> RAM
	std::memcpy(wavefield2+host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));

	// Transfer pStream -> pin (at its=0)
	cuda_call(cudaMemcpy(pin_wavefieldSlice1[iGpu], dev_pRight[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy wavefield at its=0 from pin -> RAM
	std::memcpy(wavefield2, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));

}

/********************** Forward - Subsurface offsets **************************/
// Source -> reflectivity -> model -> data
void computeTomoLeg1HxHyFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	float *dummySliceLeft, *dummySliceRight, *dummyRef;
	dummySliceLeft = new float[host_nVel];
	dummySliceRight = new float[host_nVel];
	dummyRef = new float[host_nModelExt];

	/**************************************************************************/
	/*************************** First part of leg #1 *************************/
	/*************** Source -> reflectivity -> scattered wavefield ************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = 0" << std::endl;
	// std::cout << "Min value wavefield = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
	// std::cout << "Max value wavefield = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

	// cuda_call(cudaMemcpy(dummyRef, dev_extReflectivityIn, host_nModelExt*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "Min value dummyRef = " << *std::min_element(dummyRef,dummyRef+host_nModelExt) << std::endl;
	// std::cout << "Max value dummyRef = " << *std::max_element(dummyRef,dummyRef+host_nModelExt) << std::endl;

	// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = 0" << std::endl;
	// std::cout << "Min value pLeft before = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
	// std::cout << "Max value pLeft before = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

	// std::cout << "host_hExt2 = " << host_hExt2 << std::endl;
	// std::cout << "host_hExt1 = " << host_hExt1 << std::endl;

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			// std::cout << "iExt2 = " << iExt2 << std::endl;
			// std::cout << "iExt1 = " << iExt1 << std::endl;
			imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}

	cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = 0" << std::endl;
	// std::cout << "Min value pLeft after = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
	// std::cout << "Max value pLeft after = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield1 slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Launch transfer of wavefield2 slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
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

		// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value source = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
		// std::cout << "Max value source = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

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
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));
		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Transfer pDt0 -> pin
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	/**************************************************************************/
	/*************************** Second part of leg #1 ************************/
	/***************** Scattered wavefield -> model -> data *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

	// // Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));



	// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = 0" << std::endl;
	// std::cout << "Min value pSource = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
	// std::cout << "Max value pSource = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value pSource = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
		// std::cout << "Max value pSource = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu],wavefield2+(its+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
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
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}
}

// Source -> model -> reflectivity -> data
void computeTomoLeg2HxHyFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #2 *************************/
	/******************** Source -> model -> scattered wavefield **************/
	/**************************************************************************/

	float *dummySliceLeft, *dummySliceRight;
	dummySliceLeft = new float[host_nVel];
	dummySliceRight = new float[host_nVel];

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield1 slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Launch transfer of wavefield2 slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

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

		// Wait until the scattered wavefield has been transfered to pin
		cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

		// Asynchronous copy of dev_pDt1 => dev_pDt0 [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));
		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Transfer pDt0 -> pin
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	/**************************************************************************/
	/*************************** Second part of leg #2 ************************/
	/*************** Scattered wavefield -> reflectivity -> data **************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

	// Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

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

	// QC
	// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = 0" << std::endl;
	// std::cout << "Min value pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
	// std::cout << "Max value pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu],wavefield2+(its+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// QC
		cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value pRight = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
		// std::cout << "Max value pRight = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}
}

/************************** Forward - Time-lags *******************************/
// Source -> reflectivity -> model -> data
void computeTomoLeg1TauFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #1 *************************/
	/*************** Source -> reflectivity -> scattered wavefield ************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(float))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// Copy slice from RAM -> pinned
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+iExt*host_nVel, host_nVel*sizeof(float));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	/****************************** its = 0 ***********************************/
	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Transfer slice 2*host_hExt1+1 from RAM to pinned
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(2*host_hExt1+1)*host_nVel, host_nVel*sizeof(float));

	// Launch transfer
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	// Otherwise, transfer slice its = 1 -> pSourceWavefieldTau
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	/****************************** Main loops ********************************/

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of the propagation
		if (its < 2*host_hExt1-1) {
			// Transfer slice from RAM to pinned
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Transfer slice (its+2)+2*host_hExt1 from RAM to pinned
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// Last part of the propagation
		else {

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

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
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));
		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
		}

		if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][0];
			for (int iExt1=0; iExt1<4*host_hExt1; iExt1++){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1+1];
			}
			dev_pSourceWavefieldTau[iGpu][4*host_hExt1] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until the transfer from pinned -> pStream is completed
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}
	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Transfer pDt0 -> pin
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	/**************************************************************************/
	/*************************** Second part of leg #1 ************************/
	/***************** Scattered wavefield -> model -> data *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

	// Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu],wavefield2+(its+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
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
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}
}

// Source -> model -> reflectivity -> data
void computeTomoLeg2TauFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #2 *************************/
	/***************** Source -> model -> scattered wavefield *****************/
	/**************************************************************************/

	float *dummySliceLeft, *dummySliceRight;
	dummySliceLeft = new float[host_nVel];
	dummySliceRight = new float[host_nVel];

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// QC
	// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = 0" << std::endl;
	// std::cout << "Min value pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
	// std::cout << "Max value pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value pRight = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
		// std::cout << "Max value pTight = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

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
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));
		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Transfer pDt0 -> pin
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	/**************************************************************************/
	/*************************** Second part of leg #2 ************************/
	/*************** Scattered wavefield -> reflectivity -> data **************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(float))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// Copy slice from RAM -> pinned
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+iExt*host_nVel, host_nVel*sizeof(float));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

		}
	}

	/****************************** its = 0 ***********************************/
	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Transfer slice 2*host_hExt1+1 from RAM to pinned
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(2*host_hExt1+1)*host_nVel, host_nVel*sizeof(float));

	// Launch transfer
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
	}

	// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = " << its << std::endl;
	// std::cout << "Min value pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
	// std::cout << "Max value pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	/****************************** Main loops ********************************/

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of the propagation
		if (its < 2*host_hExt1-1) {

			// Transfer slice from RAM to pinned
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield2+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Transfer slice (its+2)+2*host_hExt1 from RAM to pinned
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield2+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// Last part of the propagation
		else {

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value pRight = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
		// std::cout << "Max value pTight = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}

		if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][0];
			for (int iExt1=0; iExt1<4*host_hExt1; iExt1++){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1+1];
			}
			dev_pSourceWavefieldTau[iGpu][4*host_hExt1] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until the transfer from pinned -> pStream is completed
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}
	}

	// Wait for all jobs on GPU to be done
	cuda_call(cudaDeviceSynchronize());

}

/********************* Adjoint - Subsurface offset ****************************/
// Source -> reflectivity -> model <- data
void computeTomoLeg1HxHyAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRecWavefield[iGpu], 0, host_nVel*sizeof(float)));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

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
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Copy receiver wavefield slice from RAM -> pinned for time its = 0 -> transfer to pDt0
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy source wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){

			// Copy wavefield1 slice its+2 from RAM -> pin
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(float));

			// Wait until dev_pStream is ready to be used
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Load wavefield slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Copy wavefield2 slice its+1 from RAM -> pin
		std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(its+1)*host_nVel, host_nVel*sizeof(float));

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its+1
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

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
		cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));

	}

	// Copy receiver wavefield value at nts-1 from pDt0 -> pRecWavefield
	cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute imaging condition at its = nts-1
	imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu]);

	// Scale model for finite-difference and secondary source coefficient
	// scaleReflectivity_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

}

// Source -> model <- reflectivity <- data
void computeTomoLeg2HxHyAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRecWavefield[iGpu], 0, host_nVel*sizeof(float)));

	// Copy receiver wavefield time-slice its = nts-1
	// From RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(host_nts-1)*host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

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
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(host_nts-2)*host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Copy source wavefield slice from RAM -> pinned for time its = nts-1 -> transfer to pDt0
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(host_nts-1)*host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy receiver wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its > 0){

			// Copy receiver wavefield slice its-1 from RAM -> pin
			std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(its-1)*host_nVel, host_nVel*sizeof(float));

			// Wait until dev_pStream is ready to be used
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Load receiver wavefield slice its-1 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Copy source wavefield slice its from RAM -> pin
		std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+its*host_nVel, host_nVel*sizeof(float));

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRecWavefield[iGpu], dev_vel2Dtw2[iGpu]);

		// Compute secondary source for its + 1
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRecWavefield[iGpu], dev_extReflectivityIn, ihx, iExt1, ihy, iExt2);
			}
		}

		// Start subloop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
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
		cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));

	}

	// Copy receiver wavefield value at its = 0 from pStream -> pSourceWavefield
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute imaging condition at its = 0
	imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu]);

}

/************************* Adjoint - Time-lags ********************************/
// Source -> reflectivity -> model <- data
void computeTomoLeg1TauAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(float))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// Copy slice from RAM -> pinned
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+iExt*host_nVel, host_nVel*sizeof(float));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	/****************************** its = 0 ***********************************/
	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Transfer slice 2*host_hExt1+1 from RAM to pinned
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(2*host_hExt1+1)*host_nVel, host_nVel*sizeof(float));

	// Launch transfer
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	// Otherwise, transfer slice its = 1 -> pSourceWavefieldTau
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	// Copy receiver wavefield slice from RAM -> pinned for time its = 0 -> transfer to pDt0
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	/****************************** Main loops ********************************/

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
		// cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of the propagation
		if (its < 2*host_hExt1-1) {
			// Transfer slice from RAM to pinned
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Transfer slice (its+2)+2*host_hExt1 from RAM to pinned
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// Last part of the propagation
		else {

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}


		// Copy wavefield2 slice its+1 from RAM -> pin
		std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(its+1)*host_nVel, host_nVel*sizeof(float));

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its+1
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		// cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

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

		// Load receiver wavefield at its
		// std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+its*host_nVel, host_nVel*sizeof(float));
		// cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
		// cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

		// Apply imaging condition at its
		imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu]);

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
		}

		if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][0];
			for (int iExt1=0; iExt1<4*host_hExt1; iExt1++){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1+1];
			}
			dev_pSourceWavefieldTau[iGpu][4*host_hExt1] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until the transfer from pinned -> pStream is completed
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}
	}

	// Copy receiver wavefield value at nts-1 from pDt0 -> pRecWavefield
	cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	// cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

	// Compute imaging condition at its = nts-1
	imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu]);

	// Load receiver wavefield at nts-1
	// std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(host_nts-1)*host_nVel, host_nVel*sizeof(float));
	// cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
	// cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

	// Compute imaging condition at its = nts-1
	// imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu]);

}

// Source -> model <- reflectivity <- data
void computeTomoLeg2TauAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=4*host_hExt1; iExt>-1; iExt--){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(float))); // Useless

		// Load the source time-slices from its = 4*hExt1,...,2*hExt1 (included)
		if (iExt > 2*host_hExt1-1){

			// Copy slice from RAM -> pinned
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(host_nts-1+iExt-4*host_hExt1)*host_nVel, host_nVel*sizeof(float));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	// The last time slice loaded from the receiver wavefield is nts-1-2*hExt1
	// The index of the temporary wavefield for this slice is 2*host_hExt1

	/****************************** its = nts-1 *******************************/

	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = nts-1
	int its = host_nts-1;
	iExtMin = -its/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = (host_nts-1-its)/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Copy slice nts-2-2*host_hExt1 from RAM to pinned
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(host_nts-2-2*host_hExt1)*host_nVel, host_nVel*sizeof(float));

	// Transfre slice nts-2-2*host_hExt1 from RAM to pStream
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = nts-1
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){

		int iSlice = 2*host_hExt1 - host_nts + 1 + its + 2*iExt;
		imagingTauTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice nts-2-2*host_hExt1 only if hExt1 > 0
	// Otherwise, transfer slice its = nts-2 -> pSourceWavefieldTau[0]
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][2*host_hExt1-1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	// At this point:
	// - The imaging condition at its = nts-1 is done (secondary source in pRight)
	// - Time-slice its = nts-2-2*host_hExt1 is loaded into dev_pSourceWavefieldTau[iGpu][2*host_hExt1-1]
	// - The imaging at its = nts-2 is ready
	/****************************** Main loop *********************************/

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Lower bound for imaging condition at its
		iExtMin = -its/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its
		iExtMax = (host_nts-1-its)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of adjoint propagation
		if (its > host_nts-2*host_hExt1-1){

			// Copy slice its-2*host_hExt-1 from RAM -> pin
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(its-2*host_hExt1-1)*host_nVel, host_nVel*sizeof(float));

			// std::cout << "Copying receiver slice # " << its-2*host_hExt1-1 << " from RAM -> Pin" << std::endl;

			// Wait until compStream has done copying wavefield value from pStream -> dev_pSourceWavefield
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Copy slice its from pin -> pStream
			// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
			cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

			// Imaging condition for its
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = 2*host_hExt1 - host_nts + 1 + its + 2*iExt;
				imagingTauTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt);
			}
			// At this point, the secondary source has been computed at
			// its = nts-1 and its = nts-2
			// So we can propagate the adjoint scattered wavefield from nts-1 to nts-2

		// Middle part of adjoint propagation
		} else if (its <= host_nts-2*host_hExt1-1 && its >= 2*host_hExt1+1){

			// Copy slice its-2*host_hExt-1 from RAM -> pin
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(its-2*host_hExt1-1)*host_nVel, host_nVel*sizeof(float));

			// Wait until compStream has done copying wavefield value from pStream -> dev_pSourceWavefield
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Copy slice its from pin -> pStream
			// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
			cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

			// Imaging condition for its
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = 2*iExt;
				imagingTauTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt);
			}

		// Last part of adjoint propagation
		} else {

			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = its + 2*(iExt-host_hExt1);
				imagingTauTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt);
			}
		}

		// Start subloop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
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

		// Load source wavefield at its+1 from RAM -> pDt0
		std::memcpy(pin_wavefieldSlice2[iGpu], wavefield1+(its+1)*host_nVel, host_nVel*sizeof(float));
		cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

		// Apply imaging condition at its+1
		imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu]);

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		// cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel*sizeof(float), compStreamIn));
		cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));

		// First part of adjoint propagation
		if (its > host_nts-2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Compute index on the temporary receiver wavefield array
			int iSlice = 2*host_hExt1-host_nts+its;

			// Copy new wavefield slice from pStream -> pSourceWavefieldTau
			// cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][iSlice], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iSlice], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

		// Middle part of adjoint propagation
		} else if (its <= host_nts-2*host_hExt1-1 && its >= 2*host_hExt1+1){

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][4*host_hExt1];
			for (int iExt1=4*host_hExt1; iExt1>0; iExt1--){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1-1];
			}
			dev_pSourceWavefieldTau[iGpu][0] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			// cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));
		}

	}

	// Load source wavefield for its = 0
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield1, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));
	// Apply imaging condition at its = 0
	imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu]);

}

/******************************************************************************/
/***************************** Auxiliary functions ****************************/
/********************************* Free surface *******************************/
/******************************************************************************/

/***************************** Common parts ***********************************/
// Source wavefield with an additional second time derivative
void computeTomoFsSrcWfldDt2_3D(float *dev_sourcesIn, float *wavefield1, long long *dev_sourcesPositionsRegIn, int nSourcesRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn){

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel*sizeof(float)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));

	// Initialize pinned memory
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));

	// Compute coarse source wavefield sample at its = 0
	int its = 0;

	// Loop within two values of its (coarse time grid)
	for (int it2 = 1; it2 < host_sub+1; it2++){

		// Compute fine time-step index
		int itw = its * host_sub + it2;

		// Apply free surface condition for Laplacian
		setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

		// Step forward
		stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

		// Inject source
		injectSourceLinear_3D<<<1, nSourcesRegIn, 0, compStreamIn>>>(dev_sourcesIn, dev_p0[iGpu], itw-1, dev_sourcesPositionsRegIn);

		// Damp wavefields
		dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

		// Spread energy to dev_pLeft and dev_pRight
		interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

		// Switch pointers
		dev_temp1[iGpu] = dev_p0[iGpu];
		dev_p0[iGpu] = dev_p1[iGpu];
		dev_p1[iGpu] = dev_temp1[iGpu];
		dev_temp1[iGpu] = NULL;
	}

	// Copy pDt1 (its=0)
	cuda_call(cudaMemcpyAsync(dev_pDt1[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Switch pointers
	dev_pTemp[iGpu] = dev_pLeft[iGpu];
	dev_pLeft[iGpu] = dev_pRight[iGpu];
	dev_pRight[iGpu] = dev_pTemp[iGpu];
	dev_pTemp[iGpu] = NULL;
	cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

	/************************** Main loop (its > 0) ***************************/
	for (int its = 1; its < host_nts-1; its++){

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface condition for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesRegIn, 0, compStreamIn>>>(dev_sourcesIn, dev_p0[iGpu], itw-1, dev_sourcesPositionsRegIn);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pLeft and dev_pRight
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Copy source wavefield value at its into pDt2
		cuda_call(cudaMemcpyAsync(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Compute second-order time-derivative of source wavefield at its-1
	    srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu]);
		cuda_call(cudaStreamSynchronize(compStreamIn));
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

		// Wait for pStream to be free
		cuda_call(cudaStreamSynchronize(transferStreamIn));
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its > 1){
			std::memcpy(wavefield1+(its-2)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));
		}

		//// WHY DO YOU NEED THAT ONE ??? ////
		cuda_call(cudaStreamSynchronize(compStreamIn));

		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamIn));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for time derivative
		dev_pDtTemp[iGpu] = dev_pDt0[iGpu];
		dev_pDt0[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;

	}

	// Copy source wavefield at nts-1 into pDt2
	cuda_call(cudaMemcpyAsync(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute second order time derivative of source wavefield at nts-2
	srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu]);

	// Wait until pStream has been transfered to host
	cuda_call(cudaStreamSynchronize(transferStreamIn));

	// Copy dev_pSourceWavefield into pStream
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy second order time derivative of source wavefield at nts-3 from pin -> RAM
	std::memcpy(wavefield1+(host_nts-3)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));

	// Transfer pStream (second order time derivative of source wavefield at nts-2) to pin
	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamIn));

	// In the meantime, reset pDt0 and compute second order time-derivative at nts-1
	cuda_call(cudaMemsetAsync(dev_pDt0[iGpu], 0, host_nVel*sizeof(float), compStreamIn));
	srcWfldSecondTimeDerivative_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu], dev_pDt0[iGpu]);

	// Wait until pStream has been fully transfered to pin (derivative of source wavefield at nts-2)
	cuda_call(cudaStreamSynchronize(transferStreamIn));

	// Copy source derivative from pin -> RAM
	std::memcpy(wavefield1+(host_nts-2)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));

	// Copy source derivative at nts-1
	cuda_call(cudaMemcpy(pin_wavefieldSlice1[iGpu], dev_pSourceWavefield[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	std::memcpy(wavefield1+(host_nts-1)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));

}

// Receiver wavefield
void computeTomoFsRecWfld_3D(float *dev_dataRegDtsIn, float *wavefield2, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn){

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));

	// Initialize pinned memory
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjFreeSurfaceGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject data
			interpLinearInjectData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_dataRegDtsIn, dev_p0[iGpu], its, it2, dev_receiversPositionRegIn);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pRight[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy wavefield at its+1 from pin -> RAM
		if (its < host_nts-2) {
			std::memcpy(wavefield2+(its+2)*host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));

		}

		// Wait until pStream has been updated
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Transfer pStream -> pin (at its+1)
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

	}

	// At this point, pin contains receiver wavefield at its=1

	// Wait until pStream has been transfered
	cuda_call(cudaStreamSynchronize(transferStreamIn));

 	// Copy wavefield at its=1 from pin -> RAM
	std::memcpy(wavefield2+host_nVel, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));

	// Transfer pStream -> pin (at its=0)
	cuda_call(cudaMemcpy(pin_wavefieldSlice1[iGpu], dev_pRight[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy wavefield at its=0 from pin -> RAM
	std::memcpy(wavefield2, pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float));

}

/********************** Forward - Subsurface offsets **************************/
// Source -> reflectivity -> model -> data
void computeTomoFsLeg1HxHyFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #1 *************************/
	/*************** Source -> reflectivity -> scattered wavefield ************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

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
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield1 slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Launch transfer of wavefield2 slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
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

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

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
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));
		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Transfer pDt0 -> pin
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	/**************************************************************************/
	/*************************** Second part of leg #1 ************************/
	/***************** Scattered wavefield -> model -> data *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

	// // Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu],wavefield2+(its+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}
}

// Source -> model -> reflectivity -> data
void computeTomoFsLeg2HxHyFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #2 *************************/
	/******************** Source -> model -> scattered wavefield **************/
	/**************************************************************************/

	float *dummySliceLeft, *dummySliceRight;
	dummySliceLeft = new float[host_nVel];
	dummySliceRight = new float[host_nVel];

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield1 slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Launch transfer of wavefield2 slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

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
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));
		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Transfer pDt0 -> pin
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	/**************************************************************************/
	/*************************** Second part of leg #2 ************************/
	/*************** Scattered wavefield -> reflectivity -> data **************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

	// Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

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

	// QC
	cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = 0" << std::endl;
	// std::cout << "Min value pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
	// std::cout << "Max value pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu],wavefield2+(its+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// QC
		cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value pRight = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
		// std::cout << "Max value pRight = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}
}

/************************** Forward - Time-lags *******************************/
// Source -> reflectivity -> model -> data
void computeTomoFsLeg1TauFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #1 *************************/
	/*************** Source -> reflectivity -> scattered wavefield ************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(float))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// Copy slice from RAM -> pinned
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+iExt*host_nVel, host_nVel*sizeof(float));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	/****************************** its = 0 ***********************************/
	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Transfer slice 2*host_hExt1+1 from RAM to pinned
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(2*host_hExt1+1)*host_nVel, host_nVel*sizeof(float));

	// Launch transfer
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	// Otherwise, transfer slice its = 1 -> pSourceWavefieldTau
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	/****************************** Main loops ********************************/

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of the propagation
		if (its < 2*host_hExt1-1) {
			// Transfer slice from RAM to pinned
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Transfer slice (its+2)+2*host_hExt1 from RAM to pinned
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// Last part of the propagation
		else {

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface condition for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

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
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));
		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
		}

		if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][0];
			for (int iExt1=0; iExt1<4*host_hExt1; iExt1++){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1+1];
			}
			dev_pSourceWavefieldTau[iGpu][4*host_hExt1] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until the transfer from pinned -> pStream is completed
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}
	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Transfer pDt0 -> pin
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	/**************************************************************************/
	/*************************** Second part of leg #1 ************************/
	/***************** Scattered wavefield -> model -> data *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

	// Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu],wavefield2+(its+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface condition for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}
}

// Source -> model -> reflectivity -> data
void computeTomoFsLeg2TauFwd_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #2 *************************/
	/***************** Source -> model -> scattered wavefield *****************/
	/**************************************************************************/

	float *dummySliceLeft, *dummySliceRight;
	dummySliceLeft = new float[host_nVel];
	dummySliceRight = new float[host_nVel];

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// QC
	// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = 0" << std::endl;
	// std::cout << "Min value pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
	// std::cout << "Max value pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value pRight = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
		// std::cout << "Max value pTight = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface condition for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

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
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));
		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Transfer pDt0 -> pin
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float));

	/**************************************************************************/
	/*************************** Second part of leg #2 ************************/
	/*************** Scattered wavefield -> reflectivity -> data **************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(float))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// Copy slice from RAM -> pinned
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+iExt*host_nVel, host_nVel*sizeof(float));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

		}
	}

	/****************************** its = 0 ***********************************/
	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Transfer slice 2*host_hExt1+1 from RAM to pinned
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(2*host_hExt1+1)*host_nVel, host_nVel*sizeof(float));

	// Launch transfer
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
	}

	// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = " << its << std::endl;
	// std::cout << "Min value pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
	// std::cout << "Max value pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	/****************************** Main loops ********************************/

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of the propagation
		if (its < 2*host_hExt1-1) {

			// Transfer slice from RAM to pinned
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float));
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield2+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Transfer slice (its+2)+2*host_hExt1 from RAM to pinned
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield2+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// Last part of the propagation
		else {

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value pRight = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;
		// std::cout << "Max value pTight = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel) << std::endl;

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface condition for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}

		if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][0];
			for (int iExt1=0; iExt1<4*host_hExt1; iExt1++){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1+1];
			}
			dev_pSourceWavefieldTau[iGpu][4*host_hExt1] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until the transfer from pinned -> pStream is completed
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}
	}

	// Wait for all jobs on GPU to be done
	cuda_call(cudaDeviceSynchronize());

}

/********************* Adjoint - Subsurface offset ****************************/
// Source -> reflectivity -> model <- data
void computeTomoFsLeg1HxHyAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRecWavefield[iGpu], 0, host_nVel*sizeof(float)));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

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
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Copy receiver wavefield slice from RAM -> pinned for time its = 0 -> transfer to pDt0
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy source wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){

			// Copy wavefield1 slice its+2 from RAM -> pin
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(float));

			// Wait until dev_pStream is ready to be used
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Load wavefield slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Copy wavefield2 slice its+1 from RAM -> pin
		std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(its+1)*host_nVel, host_nVel*sizeof(float));

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its+1
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

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

			// Apply free surface condition for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

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
		cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));

	}

	// Copy receiver wavefield value at nts-1 from pDt0 -> pRecWavefield
	cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute imaging condition at its = nts-1
	imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu]);

	// Scale model for finite-difference and secondary source coefficient
	// scaleReflectivity_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

}

// Source -> model <- reflectivity <- data
void computeTomoFsLeg2HxHyAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRecWavefield[iGpu], 0, host_nVel*sizeof(float)));

	// Copy receiver wavefield time-slice its = nts-1
	// From RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(host_nts-1)*host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

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
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(host_nts-2)*host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Copy source wavefield slice from RAM -> pinned for time its = nts-1 -> transfer to pDt0
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(host_nts-1)*host_nVel, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy receiver wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its > 0){

			// Copy receiver wavefield slice its-1 from RAM -> pin
			std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(its-1)*host_nVel, host_nVel*sizeof(float));

			// Wait until dev_pStream is ready to be used
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Load receiver wavefield slice its-1 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Copy source wavefield slice its from RAM -> pin
		std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+its*host_nVel, host_nVel*sizeof(float));

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRecWavefield[iGpu], dev_vel2Dtw2[iGpu]);

		// Compute secondary source for its + 1
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRecWavefield[iGpu], dev_extReflectivityIn, ihx, iExt1, ihy, iExt2);
			}
		}

		// Start subloop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjFreeSurfaceGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2+1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

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
		cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));

	}

	// Copy receiver wavefield value at its = 0 from pStream -> pSourceWavefield
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute imaging condition at its = 0
	imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu]);

}

/************************* Adjoint - Time-lags ********************************/
// Source -> reflectivity -> model <- data
void computeTomoFsLeg1TauAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(float))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// Copy slice from RAM -> pinned
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+iExt*host_nVel, host_nVel*sizeof(float));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	/****************************** its = 0 ***********************************/
	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Transfer slice 2*host_hExt1+1 from RAM to pinned
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(2*host_hExt1+1)*host_nVel, host_nVel*sizeof(float));

	// Launch transfer
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	// Otherwise, transfer slice its = 1 -> pSourceWavefieldTau
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	// Copy receiver wavefield slice from RAM -> pinned for time its = 0 -> transfer to pDt0
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

	/****************************** Main loops ********************************/

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
		// cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of the propagation
		if (its < 2*host_hExt1-1) {
			// Transfer slice from RAM to pinned
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Transfer slice (its+2)+2*host_hExt1 from RAM to pinned
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(float), cudaMemcpyHostToHost, transferStreamH2DIn));

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// Last part of the propagation
		else {

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}


		// Copy wavefield2 slice its+1 from RAM -> pin
		std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(its+1)*host_nVel, host_nVel*sizeof(float));

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its+1
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		// cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface condition for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFsIn, dimBlockFsIn, 0, compStreamIn>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

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

		// Load receiver wavefield at its
		// std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+its*host_nVel, host_nVel*sizeof(float));
		// cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
		// cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

		// Apply imaging condition at its
		imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu]);

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(float), compStreamIn));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
		}

		if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][0];
			for (int iExt1=0; iExt1<4*host_hExt1; iExt1++){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1+1];
			}
			dev_pSourceWavefieldTau[iGpu][4*host_hExt1] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until the transfer from pinned -> pStream is completed
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}
	}

	// Copy receiver wavefield value at nts-1 from pDt0 -> pRecWavefield
	cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	// cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

	// Compute imaging condition at its = nts-1
	imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu]);

	// Load receiver wavefield at nts-1
	// std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(host_nts-1)*host_nVel, host_nVel*sizeof(float));
	// cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
	// cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

	// Compute imaging condition at its = nts-1
	// imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu]);

}

// Source -> model <- reflectivity <- data
void computeTomoFsLeg2TauAdj_3D(float *dev_modelTomoIn, float *wavefield1, float *wavefield2, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGridFsIn, dim3 dimBlockFsIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, float *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(float)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(float));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(float));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=4*host_hExt1; iExt>-1; iExt--){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(float))); // Useless

		// Load the source time-slices from its = 4*hExt1,...,2*hExt1 (included)
		if (iExt > 2*host_hExt1-1){

			// Copy slice from RAM -> pinned
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(host_nts-1+iExt-4*host_hExt1)*host_nVel, host_nVel*sizeof(float));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	// The last time slice loaded from the receiver wavefield is nts-1-2*hExt1
	// The index of the temporary wavefield for this slice is 2*host_hExt1

	/****************************** its = nts-1 *******************************/

	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = nts-1
	int its = host_nts-1;
	iExtMin = -its/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = (host_nts-1-its)/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Copy slice nts-2-2*host_hExt1 from RAM to pinned
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(host_nts-2-2*host_hExt1)*host_nVel, host_nVel*sizeof(float));

	// Transfre slice nts-2-2*host_hExt1 from RAM to pStream
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = nts-1
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){

		int iSlice = 2*host_hExt1 - host_nts + 1 + its + 2*iExt;
		imagingTauTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice nts-2-2*host_hExt1 only if hExt1 > 0
	// Otherwise, transfer slice its = nts-2 -> pSourceWavefieldTau[0]
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][2*host_hExt1-1], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	// At this point:
	// - The imaging condition at its = nts-1 is done (secondary source in pRight)
	// - Time-slice its = nts-2-2*host_hExt1 is loaded into dev_pSourceWavefieldTau[iGpu][2*host_hExt1-1]
	// - The imaging at its = nts-2 is ready
	/****************************** Main loop *********************************/

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Lower bound for imaging condition at its
		iExtMin = -its/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its
		iExtMax = (host_nts-1-its)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of adjoint propagation
		if (its > host_nts-2*host_hExt1-1){

			// Copy slice its-2*host_hExt-1 from RAM -> pin
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(its-2*host_hExt1-1)*host_nVel, host_nVel*sizeof(float));

			// std::cout << "Copying receiver slice # " << its-2*host_hExt1-1 << " from RAM -> Pin" << std::endl;

			// Wait until compStream has done copying wavefield value from pStream -> dev_pSourceWavefield
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Copy slice its from pin -> pStream
			// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
			cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

			// Imaging condition for its
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = 2*host_hExt1 - host_nts + 1 + its + 2*iExt;
				imagingTauTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt);
			}
			// At this point, the secondary source has been computed at
			// its = nts-1 and its = nts-2
			// So we can propagate the adjoint scattered wavefield from nts-1 to nts-2

		// Middle part of adjoint propagation
		} else if (its <= host_nts-2*host_hExt1-1 && its >= 2*host_hExt1+1){

			// Copy slice its-2*host_hExt-1 from RAM -> pin
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(its-2*host_hExt1-1)*host_nVel, host_nVel*sizeof(float));

			// Wait until compStream has done copying wavefield value from pStream -> dev_pSourceWavefield
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Copy slice its from pin -> pStream
			// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
			cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));

			// Imaging condition for its
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = 2*iExt;
				imagingTauTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt);
			}

		// Last part of adjoint propagation
		} else {

			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = its + 2*(iExt-host_hExt1);
				imagingTauTomoAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt);
			}
		}

		// Start subloop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjFreeSurfaceGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2+1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Load source wavefield at its+1 from RAM -> pDt0
		std::memcpy(pin_wavefieldSlice2[iGpu], wavefield1+(its+1)*host_nVel, host_nVel*sizeof(float));
		cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

		// Apply imaging condition at its+1
		imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu]);

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		// cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel*sizeof(float), compStreamIn));
		cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(float)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(float)));

		// First part of adjoint propagation
		if (its > host_nts-2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Compute index on the temporary receiver wavefield array
			int iSlice = 2*host_hExt1-host_nts+its;

			// Copy new wavefield slice from pStream -> pSourceWavefieldTau
			// cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][iSlice], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iSlice], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));

		// Middle part of adjoint propagation
		} else if (its <= host_nts-2*host_hExt1-1 && its >= 2*host_hExt1+1){

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][4*host_hExt1];
			for (int iExt1=4*host_hExt1; iExt1>0; iExt1--){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1-1];
			}
			dev_pSourceWavefieldTau[iGpu][0] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			// cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));
		}

	}

	// Load source wavefield for its = 0
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield1, host_nVel*sizeof(float));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(float), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToDevice));
	// Apply imaging condition at its = 0
	imagingAdjGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu]);

}
