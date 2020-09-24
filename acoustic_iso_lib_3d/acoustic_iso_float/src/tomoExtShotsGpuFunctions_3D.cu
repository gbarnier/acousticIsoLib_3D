#include <cstring>
#include <iostream>
#include "tomoExtShotsGpuFunctions_3D.h"
#include "varDeclare_3D.h"
#include "kernelsGpu_3D.cu"
#include "kernelsGinsuGpu_3D.cu"
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
#include "tomoExtShotsGpuFunctionsAuxNormal_3D.cu" // Auxiliary functions for normal modeling
#include "tomoExtShotsGpuFunctionsAuxFs_3D.cu" // Auxiliary functions for free surface
#include "tomoExtShotsGpuFunctionsAuxNormalGinsu_3D.cu" // Auxiliary functions for normal modeling + Ginsu
#include "tomoExtShotsGpuFunctionsAuxFsGinsu_3D.cu" // Auxiliary functions for free surface + Ginsu

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
		// dev_dataRegDtsQc = new float*[nGpu];

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
void initTomoExtGinsuGpu_3D(float dz, float dx, float dy, int nts, float dts, int sub, int blockSize, float alphaCos, std::string extension, int nExt1, int nExt2, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nts = nts;
	host_dts = dts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;
	host_extension = extension;
	host_nExt1 = nExt1;
	host_nExt2 = nExt2;
	host_hExt1 = (nExt1-1)/2;
	host_hExt2 = (nExt2-1)/2;
	host_leg1 = leg1;
	host_leg2 = leg2;

	// Coefficients for second-order time derivative
	host_cSide = 1.0 / (host_dts*host_dts);
	host_cCenter = -2.0 / (host_dts*host_dts);

	/********************** ALLOCATE ARRAYS OF ARRAYS *************************/
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
		dev_pStream = new float*[nGpu];
		dev_pRecWavefield = new float*[nGpu];

		// Subsurface offsets
		dev_pSourceWavefield = new float*[nGpu];

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

	// Second order time derivative coefficients
	cuda_call(cudaMemcpyToSymbol(dev_cCenter, &host_cCenter, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_cSide, &host_cSide, sizeof(float), 0, cudaMemcpyHostToDevice));


}

// Allocate Ginsu
void allocateSetTomoExtGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, float alphaCos, float *vel2Dtw2, float *reflectivityScale, float *extReflectivity, int iGpu, int iGpuId){

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

	// Allocate time slices on device for second time derivative of source wavefield
	cuda_call(cudaMalloc((void**) &dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_pDt2[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));

	// Reflectivity scaling
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

    // Reflectivity model
    cuda_call(cudaMalloc((void**) &dev_modelTomo[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));

	// Allocate the slice where we store the wavefield slice before transfering it to the host's pinned memory
	cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));

	// Allocate temporary slice for receiver wavefield before transfering it to the host's pinned memory
	cuda_call(cudaMalloc((void**) &dev_pRecWavefield[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));

	// Allocate and copy from host to device extended reflectivity
	cuda_call(cudaMalloc((void**) &dev_extReflectivity[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(float)));

	// Allocate the arrays that will contain the source wavefield slices for time-lags
	if (host_extension == "time"){
		for (int iExt=0; iExt<4*host_hExt1+1; iExt++){
			cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel_ginsu[iGpu]*sizeof(float)));
		}
	}
	// if (host_extension == "offset"){
	cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nVel_ginsu[iGpu]*sizeof(float)));
	// }
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
	cuda_call(cudaFree(dev_pSourceWavefield[iGpu]));
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

/******************************** Normal **************************************/
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

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

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
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 fwd time" << std::endl;
		computeTomoLeg1TauFwd_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> model -> reflectivity -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 fwd time" << std::endl;
		computeTomoLeg2TauFwd_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
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

// Time-lags + Ginsu
void tomoTauShotsFwdGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	float *dummySliceRight,*dummyRef;
	dummySliceRight = new float[host_nVel_ginsu[iGpu]];
	dummyRef = new float[host_nModelExt_ginsu[iGpu]];

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	// std::cout << "Test 1 Ginsu" << std::endl;
	computeTomoSrcWfldDt2Ginsu_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);
	// std::cout << "Test 2 Ginsu" << std::endl;
	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Scale model (background perturbation) by 2/v^3 x v^2dtw^2
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	//////////////////////////////// Debug /////////////////////////////////
    // cuda_call(cudaMemcpy(dummySliceRight, dev_reflectivityScale[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "Min value dev_reflectivityScale = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
    // std::cout << "Max value dev_reflectivityScale = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
    ////////////////////////////////////////////////////////////////////////

	//////////////////////////////// Debug /////////////////////////////////
    // cuda_call(cudaMemcpy(dummyRef, dev_extReflectivity[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "host_nModelExt_ginsu[iGpu] = " << host_nModelExt_ginsu[iGpu] << std::endl;
	// std::cout << "host_nz_ginsu[iGpu] = " << host_nz_ginsu[iGpu] << std::endl;
	// std::cout << "host_nx_ginsu[iGpu] = " << host_nx_ginsu[iGpu] << std::endl;
	// std::cout << "host_ny_ginsu[iGpu] = " << host_ny_ginsu[iGpu] << std::endl;
	// std::cout << "host_nExt1 = " << host_nExt1 << std::endl;
    // std::cout << "Min value extReflectivity = " << *std::min_element(extReflectivity,extReflectivity+host_nModelExt_ginsu[iGpu]) << std::endl;
    // std::cout << "Max value extReflectivity = " << *std::max_element(extReflectivity,extReflectivity+host_nModelExt_ginsu[iGpu]) << std::endl;
    ////////////////////////////////////////////////////////////////////////

	// Scale = 2.0 * 1/v^3 * v^2 * dtw^2
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt=0; iExt<host_nExt1; iExt++){
		long long extStride = iExt * host_nVel_ginsu[iGpu];
		scaleReflectivityTauGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
	}

	//////////////////////////////// Debug /////////////////////////////////
    // cuda_call(cudaMemcpy(dummyRef, dev_extReflectivity[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "Min value dev_extReflectivity = " << *std::min_element(dummyRef,dummyRef+host_nModelExt_ginsu[iGpu]) << std::endl;
    // std::cout << "Max value dev_extReflectivity = " << *std::max_element(dummyRef,dummyRef+host_nModelExt_ginsu[iGpu]) << std::endl;
    ////////////////////////////////////////////////////////////////////////

	// Allocate and initialize data to zero
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 fwd time" << std::endl;
		computeTomoLeg1TauFwdGinsu_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> model -> reflectivity -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 fwd time" << std::endl;
		computeTomoLeg2TauFwdGinsu_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
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
void tomoHxHyShotsFwdGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

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

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 fwd offset" << std::endl;
		computeTomoLeg1HxHyFwd_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> model -> reflectivity -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 fwd offset" << std::endl;
		computeTomoLeg2HxHyFwd_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
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

// Subsurface offsets + Ginsu
void tomoHxHyShotsFwdGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2Ginsu_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Scale model (background perturbation) by 2/v^3 x v^2dtw^2
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Copy and scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt2=0; iExt2<host_nExt2; iExt2++){
		long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
		for (int iExt1=0; iExt1<host_nExt1; iExt1++){
			long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
			scaleReflectivityLinHxHyGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2, iGpu);
		}
	}

	// Allocate and initialize data to zero
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 fwd offset" << std::endl;
		computeTomoLeg1HxHyFwdGinsu_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> model -> reflectivity -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 fwd offset" << std::endl;
		computeTomoLeg2HxHyFwdGinsu_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
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

/****************************** Free surface **********************************/
// Time-lags
void tomoTauShotsFwdFsGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2Fs_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

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
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 fwd time" << std::endl;
		computeTomoLeg1TauFwdFs_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> model -> reflectivity -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 fwd time" << std::endl;
		computeTomoLeg2TauFwdFs_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
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

// Time-lags + Ginsu
void tomoTauShotsFwdFsGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny_ginsu[iGpu]-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2FsGinsu_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Scale model (background perturbation) by 2/v^3 x v^2dtw^2
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Scale = 2.0 * 1/v^3 * v^2 * dtw^2
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt=0; iExt<host_nExt1; iExt++){
		long long extStride = iExt * host_nVel_ginsu[iGpu];
		scaleReflectivityTauGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
	}

	// Allocate and initialize data to zero
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 fwd time" << std::endl;
		computeTomoLeg1TauFwdFsGinsu_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> model -> reflectivity -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 fwd time" << std::endl;
		computeTomoLeg2TauFwdFsGinsu_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
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
void tomoHxHyShotsFwdFsGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2Fs_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

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

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 fwd offset" << std::endl;
		computeTomoLeg1HxHyFwdFs_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> model -> reflectivity -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 fwd offset" << std::endl;
		computeTomoLeg2HxHyFwdFs_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
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

// Subsurface offsets + Ginsu
void tomoHxHyShotsFwdFsGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny_ginsu[iGpu]-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2FsGinsu_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Scale model (background perturbation) by 2/v^3 x v^2dtw^2
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Copy and scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt2=0; iExt2<host_nExt2; iExt2++){
		long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
		for (int iExt1=0; iExt1<host_nExt1; iExt1++){
			long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
			scaleReflectivityLinHxHyGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2, iGpu);
		}
	}

	// Allocate and initialize data to zero
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 fwd offset" << std::endl;
		computeTomoLeg1HxHyFwdFsGinsu_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> model -> reflectivity -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 fwd offset" << std::endl;
		computeTomoLeg2HxHyFwdFsGinsu_3D(dev_modelTomo[iGpu], dev_dataRegDts[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, nblockData, iGpu, compStream[iGpu], transferStreamH2D[iGpu], transferStreamD2H[iGpu]);
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

/******************************** Normal **************************************/
// Time-lags
void tomoTauShotsAdjGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoRecWfld_3D(dev_dataRegDts[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

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
	// Source -> reflectivity -> model <- data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 adj time" << std::endl;
		computeTomoLeg1TauAdj_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 adj time" << std::endl;
		computeTomoLeg2TauAdj_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Time-lags + Ginsu
void tomoTauShotsAdjGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2Ginsu_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoRecWfldGinsu_3D(dev_dataRegDts[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Set model to zero
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Scale = 2.0 * 1/v^3 * v^2 * dtw^2
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt=0; iExt<host_nExt1; iExt++){
		long long extStride = iExt * host_nVel_ginsu[iGpu];
		scaleReflectivityTauGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
	}

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model <- data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 adj time" << std::endl;
		computeTomoLeg1TauAdjGinsu_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 adj time" << std::endl;
		computeTomoLeg2TauAdjGinsu_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Subsurface offsets
void tomoHxHyShotsAdjGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/

	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoRecWfld_3D(dev_dataRegDts[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

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
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 adj offset" << std::endl;
		computeTomoLeg1HxHyAdj_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 adj offset" << std::endl;
		computeTomoLeg2HxHyAdj_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Subsurface offsets + Ginsu
void tomoHxHyShotsAdjGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/

	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2Ginsu_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoRecWfldGinsu_3D(dev_dataRegDts[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Set model to zero
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Copy and scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt2=0; iExt2<host_nExt2; iExt2++){
		long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
		for (int iExt1=0; iExt1<host_nExt1; iExt1++){
			long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
			scaleReflectivityLinHxHyGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2, iGpu);
		}
	}

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 adj offset" << std::endl;
		computeTomoLeg1HxHyAdjGinsu_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 adj offset" << std::endl;
		computeTomoLeg2HxHyAdjGinsu_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

/****************************** Free surface **********************************/
// Time-lags
void tomoTauShotsAdjFsGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2Fs_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoRecWfldFs_3D(dev_dataRegDts[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

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
	// Source -> reflectivity -> model <- data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 adj time" << std::endl;
		computeTomoLeg1TauAdjFs_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 adj time" << std::endl;
		computeTomoLeg2TauAdjFs_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Time-lags + Ginsu
void tomoTauShotsAdjFsGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny_ginsu[iGpu]-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/
	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2FsGinsu_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

	// Compute receiver wavefield and store it into wavefield2 on RAM
	computeTomoRecWfldFsGinsu_3D(dev_dataRegDts[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Set model to zero
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Scale = 2.0 * 1/v^3 * v^2 * dtw^2
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt=0; iExt<host_nExt1; iExt++){
		long long extStride = iExt * host_nVel_ginsu[iGpu];
		scaleReflectivityTauGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
	}

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model <- data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 adj time" << std::endl;
		computeTomoLeg1TauAdjFsGinsu_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 adj time" << std::endl;
		computeTomoLeg2TauAdjFsGinsu_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Subsurface offsets
void tomoHxHyShotsAdjFsGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/

	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2Fs_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

	// Compute receiver wavefield and store it into pinned memory
	computeTomoRecWfldFs_3D(dev_dataRegDts[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

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
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 adj offset" << std::endl;
		computeTomoLeg1HxHyAdjFs_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 adj offset" << std::endl;
		computeTomoLeg2HxHyAdjFs_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}

// Subsurface offsets + Ginsu
void tomoHxHyShotsAdjFsGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

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
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny_ginsu[iGpu]-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	/**************************************************************************/
	/****************************** Source ************************************/
	/**************************************************************************/

	// The wavelet already contains the second time derivative
	// Compute source wavefield with an additional second-order time derivative
	computeTomoSrcWfldDt2FsGinsu_3D(dev_sourcesSignals[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/****************************** Receiver **********************************/
	/**************************************************************************/
	// Allocate data at coarse time-sampling on device and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
  	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

	// Compute receiver wavefield and store it into pinned memory
	computeTomoRecWfldFsGinsu_3D(dev_dataRegDts[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, nblockData, iGpu, compStream[iGpu], transferStreamD2H[iGpu]);

	/**************************************************************************/
	/************************* Preliminary steps ******************************/
	/**************************************************************************/
	// Set model to zero
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Copy and scale extended reflectivity by 2/v^3 (linearization of wave-equation)
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nModelExt_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
	for (int iExt2=0; iExt2<host_nExt2; iExt2++){
		long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
		for (int iExt1=0; iExt1<host_nExt1; iExt1++){
			long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
			scaleReflectivityLinHxHyGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2, iGpu);
		}
	}

	/**************************************************************************/
	/******************************** Leg #1 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg1 == 1){
		// std::cout << "Leg 1 adj offset" << std::endl;
		computeTomoLeg1HxHyAdjFsGinsu_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, dimGridFreeSurface, dimBlockFreeSurface, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Leg #2 **********************************/
	/**************************************************************************/
	// Source -> reflectivity -> model -> data
	if (host_leg2 == 1){
		// std::cout << "Leg 2 adj offset" << std::endl;
		computeTomoLeg2HxHyAdjFsGinsu_3D(dev_modelTomo[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, dimGrid32, dimBlock32, iGpu, compStream[iGpu], transferStreamH2D[iGpu], nblockData);
	}

	/**************************************************************************/
	/******************************** Model ***********************************/
	/**************************************************************************/
	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Copy model to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));

	/**************************** Deallocation ********************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));

}
