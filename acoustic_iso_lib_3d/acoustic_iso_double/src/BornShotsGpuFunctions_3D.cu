#include <cstring>
#include <iostream>
#include "BornShotsGpuFunctions_3D.h"
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
#include <thread>  //THREAD test

/****************************************************************************************/
/******************************* Set GPU propagation parameters *************************/
/****************************************************************************************/
// GPU info
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

// Init normal
void initBornGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nz = nz;
	host_nx = nx;
    host_ny = ny;
	host_nModel = nz * nx * ny;
	host_yStride = nz * nx;
	host_nts = nts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;

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

		// Data and model
		dev_dataRegDts = new double*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new long long*[nGpu];
		dev_receiversPositionReg = new long long*[nGpu];

        // Sources signal
		dev_sourcesSignals = new double*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new double*[nGpu];

        // Reflectivity scaling
		dev_reflectivityScale = new double*[nGpu];

        // Reflectivity
		dev_modelBorn = new double*[nGpu];

		// Debug model and data
		dev_modelDebug = new double*[nGpu];
		dev_dataDebug = new double*[nGpu];

        // Streams
		compStream = new cudaStream_t[nGpu];
		transferStream = new cudaStream_t[nGpu];
		// pin_wavefieldSlice = new double*[nGpu];
		dev_pStream = new double*[nGpu];
		dev_pSourceWavefield = new double*[nGpu];

	}

	/**************************** COMPUTE LAPLACIAN COEFFICIENTS ************************/
	double host_coeff[COEFF_SIZE] = get_coeffs((double)dz,(double)dx,(double)dy); // Stored on host

	/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
	// Time interpolation filter length / half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		throw std::runtime_error("**** ERROR [BornShotsGpuFunctions_3D]: Subsampling parameter for time interpolation is too high ****");
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
		throw std::runtime_error("**** ERROR [BornShotsGpuFunctions_3D]: Padding value is too high ****");
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
		throw std::runtime_error("**** ERROR [BornShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare file ****");
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
	cuda_call(cudaMemcpyToSymbol(dev_yStride, &host_yStride, sizeof(long long), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nModel, &host_nModel, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));

	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

}

// Allocate normal
void allocateBornShotsGpu_3D(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId){

	// Get GPU number
	cudaSetDevice(iGpuId);

	// Scaled velocity
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nModel*sizeof(double))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nModel*sizeof(double), cudaMemcpyHostToDevice));

    // Reflectivity scale
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nModel*sizeof(double))); // Allocate scaling for reflectivity
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nModel*sizeof(double), cudaMemcpyHostToDevice)); //

	// Allocate time slices on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nModel*sizeof(double))); // Allocate time slices on device (for the stepper)
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nModel*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pLeft[iGpu], host_nModel*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_pRight[iGpu], host_nModel*sizeof(double)));

    // Reflectivity model
    cuda_call(cudaMalloc((void**) &dev_modelBorn[iGpu], host_nModel*sizeof(double)));

	// Allocate the slice where we store the wavefield slice before transfering it to the host's pinned memory
	cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], host_nModel*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nModel*sizeof(double)));

}

// Allocate pinned normal
void allocatePinnedBornGpu_3D(int nzWavefield, int nxWavefield, int nyWavefield, int ntsWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	// Get GPU number
	cudaSetDevice(iGpuId);

	host_nWavefieldSpace = nzWavefield * nxWavefield * nyWavefield;

	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {
		pin_wavefieldSlice = new double*[nGpu];
	}
	// Allocate pinned memory on host
	// std::cout << "Allocating wavefield on pinned memory" << std::endl;
	// std::clock_t start;
	// double duration;
	// start = std::clock();
	cuda_call(cudaHostAlloc((void**) &pin_wavefieldSlice[iGpu], host_nWavefieldSpace*ntsWavefield*sizeof(double), cudaHostAllocDefault));
	// duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "Duration for allocation: " << duration << std::endl;
	// std::cout << "Done allocating wavefield on pinned memory" << std::endl;
}

// Init Ginsu
void initBornGinsuGpu_3D(double dz, double dx, double dy, int nts, double dts, int sub, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nts = nts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;

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

		// Data and model
		dev_dataRegDts = new double*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new long long*[nGpu];
		dev_receiversPositionReg = new long long*[nGpu];

        // Sources signal
		dev_sourcesSignals = new double*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new double*[nGpu];

        // Reflectivity scaling
		dev_reflectivityScale = new double*[nGpu];

        // Reflectivity
		dev_modelBorn = new double*[nGpu];

		// Debug model and data
		dev_modelDebug = new double*[nGpu];
		dev_dataDebug = new double*[nGpu];

        // Streams
		compStream = new cudaStream_t[nGpu];
		transferStream = new cudaStream_t[nGpu];
		// pin_wavefieldSlice = new double*[nGpu];
		dev_pStream = new double*[nGpu];
		dev_pSourceWavefield = new double*[nGpu];

	}

	/**************************** COMPUTE LAPLACIAN COEFFICIENTS ************************/
	double host_coeff[COEFF_SIZE] = get_coeffs((double)dz,(double)dx,(double)dy); // Stored on host

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
	double interpFilter[nInterpFilter];
	for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
		interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
		interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
		interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
		interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
	}

	// Check that the block size is consistent between parfile and "varDeclare.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [BornShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare file ****" << std::endl;
		throw std::runtime_error("");
	}

	/**************************** COPY TO CONSTANT MEMORY *******************************/
	// Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_coeff, host_coeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice)); // Copy Laplacian coefficients to device

	// Time interpolation filter
	cuda_call(cudaMemcpyToSymbol(dev_nTimeInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
	cuda_call(cudaMemcpyToSymbol(dev_hTimeInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
	cuda_call(cudaMemcpyToSymbol(dev_timeInterpFilter, interpFilter, nInterpFilter*sizeof(double), 0, cudaMemcpyHostToDevice)); // Filter

	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(double), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

}

// Allocate Ginsu
void allocateSetBornGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, double alphaCos, double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nz_ginsu[iGpu] = nz;
	host_nx_ginsu[iGpu] = nx;
    host_ny_ginsu[iGpu] = ny;
	host_nModel_ginsu[iGpu] = nz * nx * ny;
	host_yStride_ginsu[iGpu] = nz * nx;
	host_minPad_ginsu[iGpu] = minPad;

	/******************** COMPUTE COSINE DAMPING COEFFICIENTS *****************/
	if (minPad>=PAD_MAX){
		std::cout << "**** ERROR [BornShotsGpuFunctions_3D]: Padding value is too high ****" << std::endl;
		throw std::runtime_error("");
	}

	// Allocate array to store damping coefficients on host
	double host_cosDampingCoeffGinsuTemp[host_minPad_ginsu[iGpu]];

	// Compute array coefficients
	for (int iFilter=FAT; iFilter<FAT+host_minPad_ginsu[iGpu]; iFilter++){
		double arg = M_PI / (1.0 * host_minPad_ginsu[iGpu]) * 1.0 * (host_minPad_ginsu[iGpu]-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		host_cosDampingCoeffGinsuTemp[iFilter-FAT] = arg;
	}

	// Check that the block size is consistent between parfile and "varDeclare_3D.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [BornShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare.h file ****" << std::endl;
		throw std::runtime_error("");
	}

	/********************** COPY TO CONSTANT MEMORY ***************************/
	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeffGinsuConstant, &host_cosDampingCoeffGinsuTemp, host_minPad_ginsu[iGpu]*sizeof(double), iGpu*PAD_MAX*sizeof(double), cudaMemcpyHostToDevice)); // Array for damping
	cuda_call(cudaMemcpyToSymbol(dev_minPad_ginsu, &host_minPad_ginsu[iGpu], sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nz_ginsu, &nz, sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_nx_ginsu, &nx, sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_ny_ginsu, &ny, sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_yStride_ginsu, &host_yStride_ginsu[iGpu], sizeof(long long), iGpu*sizeof(long long), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nModel_ginsu, &host_nModel_ginsu[iGpu], sizeof(unsigned long long), iGpu*sizeof(unsigned long long), cudaMemcpyHostToDevice));

	// Allocate and copy scaled velocity model to device
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate time slices on device for the FD stepping
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_pRight[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)));

	// Reflectivity scaling
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice));

    // Reflectivity model
    cuda_call(cudaMalloc((void**) &dev_modelBorn[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)));

	// Allocate the slice where we store the wavefield slice before transfering it to the host's pinned memory
	cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)));

}

// Deallocate
void deallocateBornShotsGpu_3D(int iGpu, int iGpuId){
	cudaSetDevice(iGpuId);
	cuda_call(cudaFree(dev_vel2Dtw2[iGpu]));
    cuda_call(cudaFree(dev_reflectivityScale[iGpu]));
	cuda_call(cudaFree(dev_p0[iGpu]));
	cuda_call(cudaFree(dev_p1[iGpu]));
    cuda_call(cudaFree(dev_pLeft[iGpu]));
    cuda_call(cudaFree(dev_pRight[iGpu]));
    cuda_call(cudaFree(dev_modelBorn[iGpu]));
	cuda_call(cudaFree(dev_pStream[iGpu]));
	cuda_call(cudaFree(dev_pSourceWavefield[iGpu]));
	// cuda_call(cudaFreeHost(pin_wavefieldSlice[iGpu]));
}

// Deallocate pinned memory
void deallocatePinnedBornShotsGpu_3D(int iGpu, int iGpuId){
	// Set GPU number
	cudaSetDevice(iGpuId);
	cuda_call(cudaFreeHost(pin_wavefieldSlice[iGpu]));
}

/******************************************************************************/
/****************************** Born forward **********************************/
/******************************************************************************/
// Normal
void BornShotsFwdGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);

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

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));

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

	// Remove when done with debug
	// cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	// cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Timer
	std::clock_t start;
	double duration;
	start = std::clock();

	// std::clock_t start1;
	// double throughput;

	// double *dummySliceLeft, *dummySliceRight;
	// dummySliceLeft = new double[host_nModel];
	// dummySliceRight = new double[host_nModel];

	/********************** Source wavefield computation **********************/
	for (int its = 0; its < host_nts-1; its++){

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
			// dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
			dampCosineEdge_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pLeft and dev_pRight
			// interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
			interpFineToCoarseSlice_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Extract and interpolate data
			// kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// QC
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
		// std::cout << "Max value pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;

		// cudaMemcpy(dummySliceLeft, dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Min pLeft at its = " << its << ", = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
		// std::cout << "Max pLeft at its = " << its << ", = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
		//
		// cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Min pRight at its = " << its << ", = " << *std::min_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
		// std::cout << "Max pRight at its = " << its << ", = " << *std::max_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;

		/* Note: At that point pLeft [its] is ready to be transfered back to host */
		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
		// Uncomment when done
		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
		// At that point, the value of pStream has been transfered back to host pinned memory

		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		// Uncomment when done
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	//wait until second thread is done copying pin2 to RAM
		// 	// ptemp = pin2;
		// 	// pin2 = pin1;
		// 	// pin1 = ptemp;
		//
		// 	// Uncomment when done
		// 	// start1 = std::clock();
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
		// 	// std::copy(pin_wavefieldSlice[iGpu], pin_wavefieldSlice[iGpu]+host_nModel, srcWavefieldDts+(its-1)*host_nModel);
		// 	// mempcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
		// 	// throughput = host_nModel*sizeof(double)/((std::clock() - start1) / (double) CLOCKS_PER_SEC);
		// 	// throughput = throughput/(1024.0*1024.0*1024.0);
		// 	// std::cout << "throughput pinned to pageable: " << throughput << " [GB/s] at its= " << its << std::endl;
		// 	// copy from pin2 to pageable RAM
		//
		//
		// 	// std::cout << "its = " << its-1 << std::endl;
		// 	// std::memcpy(dummySliceLeft, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
		// 	// std::cout << "Min value = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
		// 	// std::cout << "Max value = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
		//
		// 	// Using HostToHost
		// 	//cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)

		// Uncomment when done
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));


		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample

		// Uncomment when done
		// if (its>0) {
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nModel, dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));
		// }
		// copy to pin1 from DEVICE

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Load pLeft to pStream (value of wavefield at nts-1)
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	//
	// // In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	// std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-2)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	// std::cout << "its = " << host_nts-2 << std::endl;
	// std::memcpy(dummySliceLeft, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
	// std::cout << "Min value = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
	// std::cout << "Max value = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel, dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
	// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nModel, dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

	// duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration source fwd: " << duration << std::endl;

	// std::cout << "Source wavefield min = " << *std::min_element(pin_wavefieldSlice[iGpu],pin_wavefieldSlice[iGpu]+host_nVel_ginsu[iGpu]*host_nts) << std::endl;
	// std::cout << "Source wavefield max = " << *std::max_element(pin_wavefieldSlice[iGpu],pin_wavefieldSlice[iGpu]+host_nVel_ginsu[iGpu]*host_nts) << std::endl;

	/********************** Scattered wavefield computation *******************/
	// std::clock_t start1;
	// double duration1;
	// start1 = std::clock();
	// // Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));


	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelBorn[iGpu], model, host_nModel*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2
	// Should this be done on the CPU to avoid allocating an additional time-slice on the GPU?
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	/************************ Streaming stuff starts **************************/

	// Copy wavefield time-slice its = 0: pinned -> dev_pStream -> dev_pSourceWavefield
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// // At that point:
	// // dev_pSourceWavefield contains wavefield at its=1
	// // pin_wavefieldSlice and dev_pStream are free to be used
	// // dev_pLeft (secondary source at its = 0) is computed
	// std::cout << "Starting scattered wavefield" << std::endl;
	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nModel, host_nModel*sizeof(double)); // -> this should be done with transfer stream
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2)*host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value pRight = " << *std::min_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
		// std::cout << "Max value pTight = " << *std::max_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;

		// std::cout << "its = " << its << std::endl;
		// cudaMemcpy(dummyModel, dev_pRight[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy pRight min = " << *std::min_element(dummyModel,dummyModel+host_nModel) << std::endl;
		// std::cout << "Dummy pRight max = " << *std::max_element(dummyModel,dummyModel+host_nModel) << std::endl;

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			// dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
			dampCosineEdge_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// cudaMemcpy(dummyModel, dev_p0[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost);
			// std::cout << "Dummy p0 min = " << *std::min_element(dummyModel,dummyModel+host_nModel) << std::endl;
			// std::cout << "Dummy p0 max = " << *std::max_element(dummyModel,dummyModel+host_nModel) << std::endl;

			// Extract data
			recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu]));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	}

	// duration1 = (std::clock() - start1) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration scatterd: " << duration1 << std::endl;
	// Copy data back to host

	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Ginsu
void BornShotsFwdGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);

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

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Initialize pinned memory
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double));

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

	// double *dummySliceLeft, *dummySliceRight;
	// dummySliceLeft = new double[host_nModel_ginsu[iGpu]];
	// dummySliceRight = new double[host_nModel_ginsu[iGpu]];

	// Timer
	std::clock_t start;
	double duration;
	start = std::clock();

	/********************** Source wavefield computation **********************/
	for (int its = 0; its < host_nts-1; its++){

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// cudaMemcpy(dummySliceLeft, dev_p0[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost);
			// std::cout << "Min p0 at its = " << its << ", = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nModel_ginsu[iGpu]) << std::endl;
			// std::cout << "Max p0 at its = " << its << ", = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nModel_ginsu[iGpu]) << std::endl;

			// Damp wavefields
			// dampCosineEdgeGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);
			dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pLeft and dev_pRight
			// interpFineToCoarseSliceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// QC
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
		// std::cout << "Max value pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;

		// cudaMemcpy(dummySliceLeft, dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Min pLeft at its = " << its << ", = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nModel_ginsu[iGpu]) << std::endl;
		// std::cout << "Max pLeft at its = " << its << ", = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nModel_ginsu[iGpu]) << std::endl;
		//
		// cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Min pRight at its = " << its << ", = " << *std::min_element(dummySliceRight,dummySliceRight+host_nModel_ginsu[iGpu]) << std::endl;
		// std::cout << "Max pRight at its = " << its << ", = " << *std::max_element(dummySliceRight,dummySliceRight+host_nModel_ginsu[iGpu]) << std::endl;

		/* Note: At that point pLeft [its] is ready to be transfered back to host */
		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
		// At that point, the value of pStream has been transfered back to host pinned memory

		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nModel_ginsu[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));
		//
		// 	// Using HostToHost
		// 	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nModel_ginsu[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration source Ginsu: " << duration << std::endl;

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Load pLeft to pStream (value of wavefield at nts-1)
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	//
	// // In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	// std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel_ginsu[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	// cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));
	//
	// // Copy pinned -> RAM
	// std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel_ginsu[iGpu],pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel_ginsu[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/********************** Scattered wavefield computation *******************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double));

	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelBorn[iGpu], model, host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2
	// Should this be done on the CPU to avoid allocating an additional time-slice on the GPU?
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	/************************ Streaming stuff starts **************************/

	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Copy wavefield time-slice its = 0: RAM -> pinned -> dev_pStream -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel_ginsu[iGpu]*sizeof(double));
	// // cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// double *dummyModel;
	// dummyModel = new double[host_nModel_ginsu[iGpu]];
	//
	// double *dummyData;
	// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], iGpu);

	// QC
	// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// At that point:
	// dev_pSourceWavefield contains wavefield at its=1
	// pin_wavefieldSlice and dev_pStream are free to be used
	// dev_pLeft (secondary source at its = 0) is computed
	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nModel, host_nModel*sizeof(double)); // -> this should be done with transfer stream
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
			// cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2)*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], iGpu);

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Extract data
			recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Free surface
void BornShotsFwdFreeSurfaceGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);

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

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));

	// Initialize pinned memory
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));

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

	/********************** Source wavefield computation **********************/
	for (int its = 0; its < host_nts-1; its++){

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
			// dampCosineEdgeFreeSurface_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
			dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pLeft and dev_pRight
			interpFineToCoarseSlice_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		/* Note: At that point pLeft [its] is ready to be transfered back to host */
		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
		// At that point, the value of pStream has been transfered back to host pinned memory

		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
		//
		// 	// Using HostToHost
		// 	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
		//
		// }
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));

		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nModel, dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Load pLeft to pStream (value of wavefield at nts-1)
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	//
	// // In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	// std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-2)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel, dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	// cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	// std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel,pin_wavefieldSlice[iGpu], host_nModel*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]
	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-1)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));

	/********************** Scattered wavefield computation *******************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));

	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelBorn[iGpu], model, host_nModel*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2
	// Should this be done on the CPU to avoid allocating an additional time-slice on the GPU?
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	/************************ Streaming stuff starts **************************/

	// Copy wavefield time-slice its = 0: RAM -> pinned -> dev_pStream -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel*sizeof(double));
	// // cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));


	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+host_nModel, host_nModel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// At that point:
	// dev_pSourceWavefield contains wavefield at its=1
	// pin_wavefieldSlice and dev_pStream are free to be used
	// dev_pLeft (secondary source at its = 0) is computed

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nModel, host_nModel*sizeof(double)); // -> this should be done with transfer stream
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
			// cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2)*host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu]));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Free surface + Ginsu
void BornShotsFwdFreeSurfaceGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);

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

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Initialize pinned memory
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double));

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

	double *dummySliceLeft, *dummySliceRight;
	dummySliceLeft = new double[host_nModel_ginsu[iGpu]];
	dummySliceRight = new double[host_nModel_ginsu[iGpu]];

	/********************** Source wavefield computation **********************/
	for (int its = 0; its < host_nts-1; its++){

		// std::cout << "its = " << its << std::endl;

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu], iGpu);

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
			// dampCosineEdgeFreeSurfaceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);
			dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pLeft and dev_pRight
			// interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// QC
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

		/* Note: At that point pLeft [its] is ready to be transfered back to host */
		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
		// At that point, the value of pStream has been transfered back to host pinned memory

		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nModel_ginsu[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));
		//
		// 	// Using HostToHost
		// 	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nModel_ginsu[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));
		// std::cout << "Ginsu Min value pLeft = " << *std::min_element(pin_wavefieldSlice[iGpu]+its*host_nModel_ginsu[iGpu],pin_wavefieldSlice[iGpu]+(its+1)*host_nModel_ginsu[iGpu]) << std::endl;
		// std::cout << "Ginsu Max value pLeft = " << *std::max_element(pin_wavefieldSlice[iGpu]+its*host_nModel_ginsu[iGpu],pin_wavefieldSlice[iGpu]+(its+1)*host_nModel_ginsu[iGpu]) << std::endl;

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Load pLeft to pStream (value of wavefield at nts-1)
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	//
	// // In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	// std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel_ginsu[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel_ginsu[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	// cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));
	//
	// // Copy pinned -> RAM
	// std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel_ginsu[iGpu],pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]

	/********************** Scattered wavefield computation *******************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double));

	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelBorn[iGpu], model, host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2
	// Should this be done on the CPU to avoid allocating an additional time-slice on the GPU?
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	/************************ Streaming stuff starts **************************/

	// Copy wavefield time-slice its = 0: RAM -> pinned -> dev_pStream -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel_ginsu[iGpu]*sizeof(double));
	// // cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
	//
	// double *dummyModel;
	// dummyModel = new double[host_nModel_ginsu[iGpu]];
	//
	// double *dummyData;
	// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], iGpu);

	// QC
	// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// At that point:
	// dev_pSourceWavefield contains wavefield at its=1
	// pin_wavefieldSlice and dev_pStream are free to be used
	// dev_pLeft (secondary source at its = 0) is computed

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nModel, host_nModel*sizeof(double)); // -> this should be done with transfer stream
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
			// cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2)*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], iGpu);

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu], iGpu);

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
			// dampCosineEdgeFreeSurfaceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);
			dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Extract data
			recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

/******************************************************************************/
/****************************** Born adjoint **********************************/
/******************************************************************************/
// Normal
void BornShotsAdjGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);

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

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));

	// Initialize pinned memory
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));

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

	// Timer
	std::clock_t start;
	double duration;
	start = std::clock();

	/********************** Source wavefield computation **********************/
	for (int its = 0; its < host_nts-1; its++){
		// std::cout << "its = " << its << std::endl;
		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
			dampCosineEdge_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pLeft and dev_pRight
			// interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
			interpFineToCoarseSlice_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Extract and interpolate data
			// kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		/* Note: At that point pLeft [its] is ready to be transfered back to host */
		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
		// At that point, the value of pStream has been transfered back to host pinned memory

		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// // At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]
		//
		// if (its > 0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
		//
		// 	// Using HostToHost
		// 	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample

		// std::cout << "its = " << its << std::endl;
		// std::cout << "Before" << std::endl;
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nModel, dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));
		// std::cout << "After" << std::endl;

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration source Ginsu: " << duration << std::endl;
	// std::cout << "Born 2" << std::endl;
	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Load pLeft to pStream (value of wavefield at nts-1)
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	//
	// // In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	// std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-2)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel, dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration source adj: " << duration << std::endl;

	/************************ Adjoint wavefield computation *******************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));

	// Set model to zero
	cuda_call(cudaMemset(dev_modelBorn[iGpu], 0, host_nModel*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	/************************ Streaming stuff starts **************************/
	// Copy source wavefield slice nts-1 from RAM -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(host_nts-1)*host_nModel, host_nModel*sizeof(double));
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy slice its from RAM -> pinned
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+its*host_nModel, host_nModel*sizeof(double));

		// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));

		// Copy slice its from pin -> pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+its*host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject data
			interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Damp wavefields
			// dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
			dampCosineEdge_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			// interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
			interpFineToCoarseSlice_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply imaging condition at its+1
		imagingAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

		// Copy source wavefield slice its to dev_pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu]));

	}

	// Apply imaging condition for its=0
	imagingAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBorn[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Ginsu
void BornShotsAdjGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);

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

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Initialize pinned memory
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double));

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

	// Timer
	std::clock_t start;
	double duration;
	start = std::clock();

	/********************** Source wavefield computation **********************/
	for (int its = 0; its < host_nts-1; its++){

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
			// dampCosineEdgeGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);
			dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pLeft and dev_pRight
			// interpFineToCoarseSliceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		/* Note: At that point pLeft [its] is ready to be transfered back to host */
		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
		// At that point, the value of pStream has been transfered back to host pinned memory

		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its > 0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nModel_ginsu[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nModel_ginsu[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration source Ginsu: " << duration << std::endl;

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Load pLeft to pStream (value of wavefield at nts-1)
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	//
	// // In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	// std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel_ginsu[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	// cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM

	// std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel_ginsu[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]
	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-1)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel_ginsu[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/************************ Adjoint wavefield computation *******************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	// cuda_call(cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Set model to zero
	cuda_call(cudaMemset(dev_modelBorn[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	/************************ Streaming stuff starts **************************/
	// Copy source wavefield slice nts-1 from RAM -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(host_nts-1)*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));
	// cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy slice its from RAM -> pinned
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+its*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));

		// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));

		// Copy slice its from pin -> pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+its*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject data
			interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply imaging condition at its+1
		imagingAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], iGpu);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

		// Copy source wavefield slice its to dev_pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

	}

	// Apply imaging condition for its=0
	imagingAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], iGpu);

	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBorn[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Free surface
void BornShotsAdjFreeSurfaceGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);

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

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));

	// Initialize pinned memory
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));

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

	/********************** Source wavefield computation **********************/
	for (int its = 0; its < host_nts-1; its++){

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
			// dampCosineEdgeFreeSurface_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
			dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pLeft and dev_pRight
			interpFineToCoarseSlice_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Extract and interpolate data
			// kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		/* Note: At that point pLeft [its] is ready to be transfered back to host */
		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
		// At that point, the value of pStream has been transfered back to host pinned memory

		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
		//
		// 	// Using HostToHost
		// 	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nModel, dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Load pLeft to pStream (value of wavefield at nts-1)
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	//
	// // In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	// std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-2)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel, dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	// std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel,pin_wavefieldSlice[iGpu], host_nModel*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]
	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-1)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));

	/************************ Adjoint wavefield computation *******************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
	// cuda_call(cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double)));

	// Set model to zero
	cuda_call(cudaMemset(dev_modelBorn[iGpu], 0, host_nModel*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	/************************ Streaming stuff starts **************************/

	// Copy source wavefield slice nts-1 from RAM -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(host_nts-1)*host_nModel, host_nModel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy slice its from RAM -> pinned
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+its*host_nModel, host_nModel*sizeof(double));
		// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Copy slice its from pin -> pStream
		// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		// Copy slice its from pin -> pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+its*host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjFreeSurfaceGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject data
			interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Damp wavefields
			// dampCosineEdgeFreeSurface_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
			dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			interpFineToCoarseSlice_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply imaging condition at its+1
		imagingAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

		// Copy source wavefield slice its to dev_pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu]));

	}

	// Apply imaging condition for its=0
	imagingAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpyAsync(model, dev_modelBorn[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, compStream[iGpu]));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Free surface + Ginsu
void BornShotsAdjFreeSurfaceGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);

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

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Initialize pinned memory
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double));

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

	/********************** Source wavefield computation **********************/
	for (int its = 0; its < host_nts-1; its++){

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu], iGpu);

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
			// dampCosineEdgeFreeSurfaceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);
			dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pLeft and dev_pRight
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		/* Note: At that point pLeft [its] is ready to be transfered back to host */
		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
		// At that point, the value of pStream has been transfered back to host pinned memory

		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its > 0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nModel_ginsu[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nModel_ginsu[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Load pLeft to pStream (value of wavefield at nts-1)
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	//
	// // In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	// std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel_ginsu[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel_ginsu[iGpu], dev_pLeft[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM

	// std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel_ginsu[iGpu],pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]
	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-1)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));

	/************************ Adjoint wavefield computation *******************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	// cuda_call(cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Set model to zero
	cuda_call(cudaMemset(dev_modelBorn[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	/************************ Streaming stuff starts **************************/
	// Copy source wavefield slice nts-1 from RAM -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(host_nts-1)*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));
	// cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// // Copy slice its from RAM -> pinned
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+its*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double));

		// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));

		// Copy slice its from pin -> pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+its*host_nModel_ginsu[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjFreeSurfaceGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject data
			interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Damp wavefields
			// dampCosineEdgeFreeSurfaceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);
			dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply imaging condition at its+1
		imagingAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], iGpu);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

		// Copy source wavefield slice its to dev_pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

	}

	// Apply imaging condition for its=0
	imagingAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], iGpu);

	// Scale model for finite-difference and secondary source coefficient
	scaleReflectivityGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBorn[iGpu], host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Debug shit
// void BornShotsAdjNoStreamGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, int iGpu, int iGpuId){
//
//
// 	// We assume the source wavelet/signals already contain the second time derivative
// 	// Set device number
// 	cudaSetDevice(iGpuId);
//
// 	// Create streams
// 	cudaStreamCreate(&compStream[iGpu]);
// 	cudaStreamCreate(&transferStream[iGpu]);
//
// 	// Sources geometry
// 	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
// 	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
// 	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));
//
// 	// Sources geometry + signals
//   	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
// 	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device
//
// 	// Receivers geometry
// 	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
// 	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
// 	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));
//
// 	// Initialize time-slices for time-stepping
//   	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
//
// 	// Initialize time-slices for transfer to host's pinned memory
//   	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
//
// 	// Initialize pinned memory
// 	cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));
//
// 	// Blocks for Laplacian
// 	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
// 	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
// 	dim3 dimGrid(nblockx, nblocky);
// 	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
//
// 	// Blocks data recording
// 	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
//
// 	/********************** Source wavefield computation **********************/
// 	for (int its = 0; its < host_nts-1; its++){
//
// 		// Loop within two values of its (coarse time grid)
// 		for (int it2 = 1; it2 < host_sub+1; it2++){
//
// 			// Compute fine time-step index
// 			int itw = its * host_sub + it2;
//
// 			// Step forward
// 			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
//
// 			// Inject source
// 			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);
//
// 			// Damp wavefields
// 			dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
//
// 			// Spread energy to dev_pLeft and dev_pRight
// 			interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
//
// 			// Extract and interpolate data
// 			// kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));
//
// 			// Switch pointers
// 			dev_temp1[iGpu] = dev_p0[iGpu];
// 			dev_p0[iGpu] = dev_p1[iGpu];
// 			dev_p1[iGpu] = dev_temp1[iGpu];
// 			dev_temp1[iGpu] = NULL;
//
// 		}
//
// 		/* Note: At that point pLeft [its] is ready to be transfered back to host */
// 		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
// 		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
// 		// At that point, the value of pStream has been transfered back to host pinned memory
//
// 		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
// 		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
// 		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]
//
// 		if (its>0) {
// 			// Standard library
// 			std::memcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
//
// 			// Using HostToHost
// 			// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
//
// 		}
// 		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
// 		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
// 		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
// 		// Asynchronous transfer of pStream => pin [its] [transfer]
// 		// Launch the transfer while we compute the next coarse time sample
// 		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));
//
// 		// Switch pointers
// 		dev_pTemp[iGpu] = dev_pLeft[iGpu];
// 		dev_pLeft[iGpu] = dev_pRight[iGpu];
// 		dev_pRight[iGpu] = dev_pTemp[iGpu];
// 		dev_pTemp[iGpu] = NULL;
//   		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
// 	}
//
// 	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
// 	// The CPU has stored the wavefield values ranging from 0,...,nts-3
//
// 	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
// 	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
//
// 	// Load pLeft to pStream (value of wavefield at nts-1)
// 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
//
// 	// In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
// 	std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
// 	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-2)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
//
// 	// Wait until pLeft -> pStream is done
// 	cuda_call(cudaStreamSynchronize(compStream[iGpu]));
//
// 	// At this point, pStream contains the value of the wavefield at nts-1
// 	// Transfer pStream -> pinned
// 	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
//
// 	// Copy pinned -> RAM
// 	std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel,pin_wavefieldSlice[iGpu], host_nModel*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]
// 	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-1)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
//
// 	/************************ Adjoint wavefield computation *******************/
//
// 	// Reset the time slices to zero
// 	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
// 	cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));
//
// 	// Set model to zero
// 	cuda_call(cudaMemset(dev_modelBorn[iGpu], 0, host_nModel*sizeof(double)));
//
// 	// Allocate and copy data from host -> device
//   	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
// 	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device
//
// 	// Start propagating scattered wavefield
// 	for (int its = host_nts-2; its > -1; its--){
//
// 		// Load source wavefield for its+1
// 		std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+1)*host_nModel, host_nModel*sizeof(double));
// 		cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice));
// 		cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice));
//
// 		for (int it2 = host_sub-1; it2 > -1; it2--){
//
// 			// Step adjoint
// 			stepAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
//
// 			// Inject data
// 			interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);
//
// 			// Damp wavefields
// 			dampCosineEdge_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]);
//
// 			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
// 			// pLeft corresponds to its, pRight corresponds to its+1
// 			interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
//
// 			// Switch pointers
// 			dev_temp1[iGpu] = dev_p0[iGpu];
// 			dev_p0[iGpu] = dev_p1[iGpu];
// 			dev_p1[iGpu] = dev_temp1[iGpu];
// 			dev_temp1[iGpu] = NULL;
//
// 		}
// 		// At that point, the receiver wavefield for its+1 is done (stored in pRight)
// 		// Apply imaging condition for index its+1
// 		imagingAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
//
// 		// Switch pointers for secondary source
// 		dev_pTemp[iGpu] = dev_pRight[iGpu];
// 		dev_pRight[iGpu] = dev_pLeft[iGpu];
// 		dev_pLeft[iGpu] = dev_pTemp[iGpu];
// 		dev_pTemp[iGpu] = NULL;
// 		cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
//
// 	}
//
// 	// Load source wavefield for 0
// 	std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel*sizeof(double));
// 	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice));
// 	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice));
//
// 	// Finished main loop - we still have to compute imaging condition for its = 0
//   	imagingAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
//
// 	// Scale model for finite-difference and secondary source coefficient
// 	kernel_exec(scaleReflectivity_3D<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));
//
// 	// Copy model back to host
// 	cuda_call(cudaMemcpy(model, dev_modelBorn[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
//
// 	/*************************** Memory deallocation **************************/
// 	// Deallocate the array for sources/receivers' positions
//     cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
//     cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
// 	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
//
// 	// Destroy the streams
//     cuda_call(cudaStreamDestroy(compStream[iGpu]));
//     cuda_call(cudaStreamDestroy(transferStream[iGpu]));
//
// }










/****************************** Born Forward Thread test**********************************/


// void BornShotsFwdGpu_3D_Threads(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, int iGpu, int iGpuId){
//
// 	// We assume the source wavelet/signals already contain the second time derivative
// 	// Set device number
// 	cudaSetDevice(iGpuId);
//
// 	// Create streams
// 	cudaStreamCreate(&compStream[iGpu]);
// 	cudaStreamCreate(&transferStream[iGpu]);
//
// 	// Sources geometry
// 	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
// 	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
// 	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));
//
// 	// Sources geometry + signals
//   	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
// 	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device
//
// 	// Receivers geometry
// 	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
// 	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
// 	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));
//
// 	// Initialize time-slices for time-stepping
//   	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
//
// 	// Initialize time-slices for transfer to host's pinned memory
//   	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
//
// 	// Initialize pinned memory
// 	cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));
// 	//Allocating second pinned memory slice
// 	double * pin2; //TESTING
// 	cuda_call(cudaHostAlloc((void**) &pin2, host_nModel*sizeof(double), cudaHostAllocDefault)); //TESTING
// 	cudaMemset(pin2, 0, host_nModel*sizeof(double)); //TESTING
//
// 	// Blocks for Laplacian
// 	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
// 	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
// 	dim3 dimGrid(nblockx, nblocky);
// 	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
//
// 	// Blocksize = 32
// 	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
// 	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
// 	dim3 dimGrid32(nblockx32, nblocky32);
// 	dim3 dimBlock32(32, 32);
//
// 	// Blocks data recording
// 	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
//
// 	// Remove when done with debug
// 	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
//   	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device
//
// 	// Timer
// 	std::clock_t start;
// 	double duration;
// 	start = std::clock();
//
// 	std::clock_t start1;
// 	double duration1;
//
// 	std::cout << "Declaring thread" << std::endl;
// 	std::thread pinthread; //TESTING
//
// 	/********************** Source wavefield computation **********************/
// 	for (int its = 0; its < host_nts-1; its++){
//
// 		// Loop within two values of its (coarse time grid)
// 		for (int it2 = 1; it2 < host_sub+1; it2++){
//
// 			// Compute fine time-step index
// 			int itw = its * host_sub + it2;
//
// 			// Step forward
// 			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
//
// 			// Inject source
// 			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);
//
// 			// Damp wavefields
// 			// dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
// 			dampCosineEdge_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
//
// 			// Spread energy to dev_pLeft and dev_pRight
// 			// interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
// 			interpFineToCoarseSlice_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
//
// 			// Extract and interpolate data
// 			// kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));
//
// 			// Switch pointers
// 			dev_temp1[iGpu] = dev_p0[iGpu];
// 			dev_p0[iGpu] = dev_p1[iGpu];
// 			dev_p1[iGpu] = dev_temp1[iGpu];
// 			dev_temp1[iGpu] = NULL;
//
// 		}
//
// 		// QC
// 		// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
// 		// std::cout << "its = " << its << std::endl;
// 		// std::cout << "Min value pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
// 		// std::cout << "Max value pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
//
// 		// cudaMemcpy(dummySliceLeft, dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost);
// 		// std::cout << "Min pLeft at its = " << its << ", = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
// 		// std::cout << "Max pLeft at its = " << its << ", = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
// 		//
// 		// cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost);
// 		// std::cout << "Min pRight at its = " << its << ", = " << *std::min_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
// 		// std::cout << "Max pRight at its = " << its << ", = " << *std::max_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
//
// 		/* Note: At that point pLeft [its] is ready to be transfered back to host */
// 		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
// 		// Uncomment when done
// 		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
// 		// At that point, the value of pStream has been transfered back to host pinned memory
//
// 		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
// 		// Uncomment when done
// 		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
// 		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]
//
// 		if (its>0) {
// 			// Standard library
// 			//wait until second thread is done copying pin2 to RAM
// 			dev_temp1[iGpu] = pin2;
// 			pin2 = pin_wavefieldSlice[iGpu];
// 			pin_wavefieldSlice[iGpu] = dev_temp1[iGpu];
//
// 			// Uncomment when done
// 			// std::memcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
// 			// mempcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
// 			if (its>1){
// 				pinthread.join();
// 				duration1 = (std::clock() - start1) / (double) CLOCKS_PER_SEC;
// 				std::cout << "duration memcpy: " << duration1 << std::endl;
// 			}
// 			start1 = std::clock();
// 			pinthread = std::thread(std::memcpy, srcWavefieldDts+(its-1)*host_nModel, pin2, host_nModel*sizeof(double));
// 			// copy from pin2 to pageable RAM
//
//
// 			// std::cout << "its = " << its-1 << std::endl;
// 			// std::memcpy(dummySliceLeft, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
// 			// std::cout << "Min value = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
// 			// std::cout << "Max value = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
//
// 			// Using HostToHost
// 			//cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
//
// 		}
// 		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
// 		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
//
// 		// Uncomment when done
// 		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
//
//
// 		// Asynchronous transfer of pStream => pin [its] [transfer]
// 		// Launch the transfer while we compute the next coarse time sample
//
// 		// Uncomment when done
// 		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));
// 		// copy to pin1 from DEVICE
//
//
// 		// Switch pointers
// 		dev_pTemp[iGpu] = dev_pLeft[iGpu];
// 		dev_pLeft[iGpu] = dev_pRight[iGpu];
// 		dev_pRight[iGpu] = dev_pTemp[iGpu];
// 		dev_pTemp[iGpu] = NULL;
//   		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
// 	}
//
// 	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
// 	std::cout << "duration source: " << duration << std::endl;
//
// 	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
// 	// The CPU has stored the wavefield values ranging from 0,...,nts-3
//
// 	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
// 	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
//
// 	// Load pLeft to pStream (value of wavefield at nts-1)
// 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
//
// 	// In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
// 	std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
// 	pinthread.join(); //TESTING
// 	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-2)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
// 	// std::cout << "its = " << host_nts-2 << std::endl;
// 	// std::memcpy(dummySliceLeft, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
// 	// std::cout << "Min value = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
// 	// std::cout << "Max value = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
//
// 	// Wait until pLeft -> pStream is done
// 	cuda_call(cudaStreamSynchronize(compStream[iGpu]));
//
// 	// At this point, pStream contains the value of the wavefield at nts-1
// 	// Transfer pStream -> pinned
// 	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
//
// 	// Copy pinned -> RAM
// 	std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel,pin_wavefieldSlice[iGpu], host_nModel*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]
// 	// std::cout << "its = " << host_nts-1 << std::endl;
// 	// std::memcpy(dummySliceLeft, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
// 	// std::cout << "Min value = " << *std::min_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
// 	// std::cout << "Max value = " << *std::max_element(dummySliceLeft,dummySliceLeft+host_nModel) << std::endl;
//
// 	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-1)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
// 	// std::cout << "Done source wavefield" << std::endl;
// 	/********************** Scattered wavefield computation *******************/
// 	start = std::clock();
// 	// Reset the time slices to zero
// 	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nModel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nModel*sizeof(double)));
//   	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));
// 	cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));
//
// 	// Copy model to device
// 	cuda_call(cudaMemcpy(dev_modelBorn[iGpu], model, host_nModel*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device
//
// 	// Allocate and initialize data
//   	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
//   	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device
//
// 	// Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2
// 	// Should this be done on the CPU to avoid allocating an additional time-slice on the GPU?
// 	scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);
//
// 	/************************ Streaming stuff starts **************************/
//
// 	// Copy wavefield time-slice its = 0: RAM -> pinned -> dev_pStream -> dev_pSourceWavefield
// 	std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel*sizeof(double));
// 	// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nModel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
// 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
// 	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
// 	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
//
// 	// double *dummyModel;
// 	// dummyModel = new double[host_nModel];
//
// 	// double *dummyData;
// 	// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];
//
// 	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
// 	imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);
//
// 	// QC
// 	// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
// 	// std::cout << "its = 0" << std::endl;
// 	// std::cout << "Min value pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
// 	// std::cout << "Max value pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
//
// 	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
// 	std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+host_nModel, host_nModel*sizeof(double));
// 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
// 	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
//
// 	// At that point:
// 	// dev_pSourceWavefield contains wavefield at its=1
// 	// pin_wavefieldSlice and dev_pStream are free to be used
// 	// dev_pLeft (secondary source at its = 0) is computed
// 	// std::cout << "Starting scattered wavefield" << std::endl;
// 	// Start propagating scattered wavefield
// 	for (int its = 0; its < host_nts-1; its++){
//
// 		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
// 		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
//
// 		if (its < host_nts-2){
// 			// Copy wavefield slice its+2 from RAM > dev_pStream
// 			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nModel, host_nModel*sizeof(double)); // -> this should be done with transfer stream
// 			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nModel, host_nModel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
// 			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
// 			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
// 		}
//
// 		// Compute secondary source for first coarse time index (its+1) with compute stream
// 		imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
//
// 		// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
// 		// std::cout << "its = " << its << std::endl;
// 		// std::cout << "Min value pRight = " << *std::min_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
// 		// std::cout << "Max value pTight = " << *std::max_element(dummySliceRight,dummySliceRight+host_nModel) << std::endl;
//
// 		// std::cout << "its = " << its << std::endl;
// 		// cudaMemcpy(dummyModel, dev_pRight[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost);
// 		// std::cout << "Dummy pRight min = " << *std::min_element(dummyModel,dummyModel+host_nModel) << std::endl;
// 		// std::cout << "Dummy pRight max = " << *std::max_element(dummyModel,dummyModel+host_nModel) << std::endl;
//
// 		for (int it2 = 1; it2 < host_sub+1; it2++){
//
// 			// Step forward
// 			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
//
// 			// Inject secondary source sample itw-1
// 			injectSecondarySource_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);
//
// 			// Damp wavefields
// 			// dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
// 			dampCosineEdge_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
//
// 			// cudaMemcpy(dummyModel, dev_p0[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost);
// 			// std::cout << "Dummy p0 min = " << *std::min_element(dummyModel,dummyModel+host_nModel) << std::endl;
// 			// std::cout << "Dummy p0 max = " << *std::max_element(dummyModel,dummyModel+host_nModel) << std::endl;
//
// 			// Extract data
// 			recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);
//
// 			// Switch pointers
// 			dev_temp1[iGpu] = dev_p0[iGpu];
// 			dev_p0[iGpu] = dev_p1[iGpu];
// 			dev_p1[iGpu] = dev_temp1[iGpu];
// 			dev_temp1[iGpu] = NULL;
//
// 		}
//
// 		// cudaMemcpy(dummyData, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost);
// 		// std::cout << "Dummy data min = " << *std::min_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
// 		// std::cout << "Dummy data max = " << *std::max_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
// 		// std::cout << "---------------------" << std::endl;
//
// 		// Switch pointers for secondary source
// 		dev_pTemp[iGpu] = dev_pLeft[iGpu];
// 		dev_pLeft[iGpu] = dev_pRight[iGpu];
// 		dev_pRight[iGpu] = dev_pTemp[iGpu];
// 		dev_pTemp[iGpu] = NULL;
// 		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu]));
//
// 		// Wait until the transfer from pinned -> pStream is completed
// 		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
//
// 	}
// 	// duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
// 	// std::cout << "duration scatterd: " << duration << std::endl;
// 	// Copy data back to host
// 	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
//
// 	/*************************** Memory deallocation **************************/
// 	// Deallocate the array for sources/receivers' positions
//     cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
//     cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
// 	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
//     cuda_call(cudaStreamDestroy(compStream[iGpu]));
//     cuda_call(cudaStreamDestroy(transferStream[iGpu]));
//
// }
