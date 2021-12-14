#include <cstring>
#include <iostream>
#include "BornExtShotsGpuFunctions_3D.h"
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

/****************************************************************************************/
/******************************* Set GPU propagation parameters *************************/
/****************************************************************************************/
// Display GPU information
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

// Initialize
void initBornExtGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, std::string extension, int nExt1, int nExt2, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nz = nz;
	host_nx = nx;
    host_ny = ny;
	host_yStride = nz * nx;
	host_nts = nts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;
	host_extension = extension;
	host_nExt1 = nExt1;
    host_nExt2 = nExt2;
	host_hExt1 = (nExt1-1)/2;
    host_hExt2 = (nExt2-1)/2;
	host_nVel = nz * nx * ny;
	host_nModelExt = host_nVel * nExt1 * nExt2;
	host_extStride = host_nExt1 * host_nVel;

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
		dev_modelBornExt = new double*[nGpu];

		// Debug model and data
		dev_modelDebug = new double*[nGpu];
		dev_dataDebug = new double*[nGpu];

        // Streams
		compStream = new cudaStream_t[nGpu];
		transferStream = new cudaStream_t[nGpu];
		transferStreamH2D = new cudaStream_t[nGpu];
		dev_pStream = new double*[nGpu];

		// Subsurface offsets
		if (host_extension == "offset"){
			dev_pSourceWavefield = new double*[nGpu];
		}

		// Time-lags
		if (host_extension == "time"){
			dev_pSourceWavefieldTau = new double**[nGpu];
			for (int iGpu=0; iGpu<nGpu; iGpu++){
				dev_pSourceWavefieldTau[iGpu] = new double*[4*host_hExt1+1];
			}
			dev_pTempTau = new double*[nGpu];
		}
	}

	/**************************** COMPUTE LAPLACIAN COEFFICIENTS ************************/
	double host_coeff[COEFF_SIZE] = get_coeffs((double)dz,(double)dx,(double)dy); // Stored on host

	/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
	// Time interpolation filter length / half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		throw std::runtime_error("**** ERROR [BornExtShotsGpuFunctions_3D]: Subsampling parameter for time interpolation is too high ****");
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
		throw std::runtime_error("**** ERROR [BornExtShotsGpuFunctions_3D]: Padding value is too high ****");
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
		throw std::runtime_error("**** ERROR [BornExtShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare file ****");
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

}

// Allocate normal (no Ginsu)
void allocateBornExtShotsGpu_3D(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId){

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

    // Reflectivity model
    cuda_call(cudaMalloc((void**) &dev_modelBornExt[iGpu], host_nModelExt*sizeof(double)));

	// Allocate the slice where we store the wavefield slice before transfering it to the host's pinned memory
	cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], host_nVel*sizeof(double)));

	// Allocate the arrays that will contain the source wavefield slices for time-lags
	if (host_extension == "time"){
			for (int iExt=0; iExt<4*host_hExt1+1; iExt++){
			cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(double)));
		}
	}
	if (host_extension == "offset"){
		cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nVel*sizeof(double)));
	}
}

// Allocate pinned normal
void allocatePinnedBornExtGpu_3D(int nzWavefield, int nxWavefield, int nyWavefield, int ntsWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	// Get GPU number
	cudaSetDevice(iGpuId);

	host_nWavefieldSpace = nzWavefield * nxWavefield * nyWavefield;

	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {
		pin_wavefieldSlice = new double*[nGpu];
	}
	// Allocate pinned memory on host
	cuda_call(cudaHostAlloc((void**) &pin_wavefieldSlice[iGpu], host_nWavefieldSpace*ntsWavefield*sizeof(double), cudaHostAllocDefault));
}

// Allocate pinned normal
void setPinnedBornExtGpuFwime_3D(double* wavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	// Get GPU number
	cudaSetDevice(iGpuId);

	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {
		pin_wavefieldSlice = new double*[nGpu];
	}
	// Set pointer to wavefield
	pin_wavefieldSlice[iGpu] = wavefield;
}

// Init Ginsu
void initBornExtGinsuGpu_3D(double dz, double dx, double dy, int nts, double dts, int sub, int blockSize, double alphaCos, std::string extension, int nExt1, int nExt2, int nGpu, int iGpuId, int iGpuAlloc){

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

	/********************** ALLOCATE ARRAYS OF ARRAYS *************************/
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
		dev_modelBornExt = new double*[nGpu];

		// Debug model and data
		dev_modelDebug = new double*[nGpu];
		dev_dataDebug = new double*[nGpu];

        // Streams
		compStream = new cudaStream_t[nGpu];
		transferStream = new cudaStream_t[nGpu];
		transferStreamH2D = new cudaStream_t[nGpu];
		dev_pStream = new double*[nGpu];

		// Subsurface offsets
		if (host_extension == "offset"){
			dev_pSourceWavefield = new double*[nGpu];
		}

		// Time-lags
		if (host_extension == "time"){
			dev_pSourceWavefieldTau = new double**[nGpu];
			for (int iGpu=0; iGpu<nGpu; iGpu++){
				std::cout << "Allocating wavefield slices Ginsu" << std::endl;
				dev_pSourceWavefieldTau[iGpu] = new double*[4*host_hExt1+1];
			}
			dev_pTempTau = new double*[nGpu];
		}

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
void allocateSetBornExtGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, double alphaCos, double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId){

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
		std::cout << "**** ERROR [BornExtShotsGpuFunctions_3D]: Padding value is too high ****" << std::endl;
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
		std::cout << "**** ERROR [BornExtShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare.h file ****" << std::endl;
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

	cuda_call(cudaMemcpyToSymbol(dev_nExt1, &host_nExt1, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nExt2, &host_nExt2, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_hExt1, &host_hExt1, sizeof(int), 0, cudaMemcpyHostToDevice));
    cuda_call(cudaMemcpyToSymbol(dev_hExt2, &host_hExt2, sizeof(int), 0, cudaMemcpyHostToDevice));

	cuda_call(cudaMemcpyToSymbol(dev_yStride_ginsu, &host_yStride_ginsu[iGpu], sizeof(long long), iGpu*sizeof(long long), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nVel_ginsu, &host_nVel_ginsu[iGpu], sizeof(unsigned long long), iGpu*sizeof(unsigned long long), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nModelExt_ginsu, &host_nModelExt_ginsu[iGpu], sizeof(unsigned long long), iGpu*sizeof(unsigned long long), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_extStride_ginsu, &host_extStride_ginsu[iGpu], sizeof(long long), iGpu*sizeof(long long), cudaMemcpyHostToDevice));

	// Allocate and copy scaled velocity model to device
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate time slices on device for the FD stepping
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_pRight[iGpu], host_nVel_ginsu[iGpu]*sizeof(double)));

	// Reflectivity scaling
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice));

    // Reflectivity model
    cuda_call(cudaMalloc((void**) &dev_modelBornExt[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(double)));

	// Allocate the slice where we store the wavefield slice before transfering it to the host's pinned memory
	cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double)));

	// Allocate the arrays that will contain the source wavefield slices for time-lags
	if (host_extension == "time"){
		for (int iExt=0; iExt<4*host_hExt1+1; iExt++){
			// std::cout << "iExt = " << iExt << std::endl;
			cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel_ginsu[iGpu]*sizeof(double)));
		}
	}
	if (host_extension == "offset"){
		cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nVel_ginsu[iGpu]*sizeof(double)));
	}
}

// Deallocate
void deallocateBornExtShotsGpu_3D(int iGpu, int iGpuId){
	cudaSetDevice(iGpuId);
	cuda_call(cudaFree(dev_vel2Dtw2[iGpu]));
    cuda_call(cudaFree(dev_reflectivityScale[iGpu]));
	cuda_call(cudaFree(dev_p0[iGpu]));
	cuda_call(cudaFree(dev_p1[iGpu]));
    cuda_call(cudaFree(dev_pLeft[iGpu]));
    cuda_call(cudaFree(dev_pRight[iGpu]));
    cuda_call(cudaFree(dev_modelBornExt[iGpu]));
	cuda_call(cudaFree(dev_pStream[iGpu]));
	// Deallocate the arrays that will contain the source wavefield slices for time-lags
	if (host_extension == "time"){
		for (int iExt=0; iExt<4*host_hExt1+1; iExt++){
			cuda_call(cudaFree(dev_pSourceWavefieldTau[iGpu][iExt]));
		}
	}
	if (host_extension == "offset"){
		cuda_call(cudaFree(dev_pSourceWavefield[iGpu]));
	}
}

// Deallocate pinned memory
void deallocatePinnedBornExtShotsGpu_3D(int iGpu, int iGpuId){
	// Set GPU number
	cudaSetDevice(iGpuId);
	cuda_call(cudaFreeHost(pin_wavefieldSlice[iGpu]));
}

/******************************************************************************/
/************************* Born extended forward ******************************/
/******************************************************************************/

/******************************* No free surface ******************************/
// Time
void BornTauShotsFwdGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);
	// cudaStreamCreate(&transferStreamH2D[iGpu]);

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

	// Transfer extended model from host to device
	// cuda_call(cudaMemcpyAsync(dev_modelBornExt[iGpu], model, host_nModelExt*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2D[iGpu]));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

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

	// cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	// cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Timer
	// std::clock_t start;
	// double duration;
	// start = std::clock();

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

			// recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel, dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel, dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************************************************************/
	/********************** Scattered wavefield computation *******************/
	/**************************************************************************/

	// double *dummySliceLeft, *dummySliceRight;
	// dummySliceLeft = new double[host_nVel];
	// dummySliceRight = new double[host_nVel];

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(double)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(double))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// // Copy slice from RAM -> pinned
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+iExt*host_nVel, host_nVel*sizeof(double));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice[iGpu]+iExt*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice));
		}
	}

	// Copy model to device
	// cuda_call(cudaStreamSynchronize(transferStreamH2D[iGpu]));
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nModelExt*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Timer
	// std::clock_t start;
	// double duration;
	// start = std::clock();

	// Apply scaling to reflectivity
	if (slowSquare == 0){
		// Apply both scalings to reflectivity coming from for the wave-equation linearization and the finite-difference propagation (we can do them simultaneously because the extension is in time-lags)
		// scale = 2.0 * 1/v^3 * v^2 * dtw^2
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel;
			scaleReflectivityTau_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride);
		}
	} else {
		// std::cout << "Slowness squared" << std::endl;
		// Apply scaling for slowness squared parametrization (only scale by -v^2 * dtw^2)
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel;
			scaleReflectivityTauSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_vel2Dtw2[iGpu], extStride);
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

	// Transfer slice 2*host_hExt1+1 from host to device
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(2*host_hExt1+1)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		// imagingTauFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
		imagingTauFwdGpu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
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

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				// imagingTauFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
				imagingTauFwdGpu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				// imagingTauFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
				imagingTauFwdGpu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// Last part of the propagation
		else {
			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				// imagingTauFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
				imagingTauFwdGpu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// QC
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefieldTau[iGpu][0], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu]));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

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
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		}
	}

	// duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "Duration scattered fwd: " << duration << std::endl;

	// Wait for all jobs on GPU to be done
	cuda_call(cudaDeviceSynchronize());

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));
	// cuda_call(cudaStreamDestroy(transferStreamH2D[iGpu]));

}

// Time Ginsu
void BornTauShotsFwdGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

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

	// Transfer extended model from host to device
	// cuda_call(cudaMemcpyAsync(dev_modelBornExt[iGpu], model, host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2D[iGpu]));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

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

		// std::cout << "its = " << its << std::endl;

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
			// dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
			dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pLeft and dev_pRight
			// interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);
			// recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel_ginsu[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************************************************************/
	/********************** Scattered wavefield computation *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(double)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel_ginsu[iGpu]*sizeof(double))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// // Copy slice from RAM -> pinned
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+iExt*host_nVel, host_nVel*sizeof(double));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice[iGpu]+iExt*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice));
		}
	}

	// Copy model to device
	// cuda_call(cudaStreamSynchronize(transferStreamH2D[iGpu]));
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	if (slowSquare == 0){
		// Apply both scalings to reflectivity coming from for the wave-equation linearization and the finite-difference propagation (we can do them simultaneously because the extension is in time-lags):
		// scale = 2.0 * 1/v^3 * v^2 * dtw^2
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel_ginsu[iGpu];
			scaleReflectivityTauGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
		}
	} else {
		// Apply scaling for slowness squared parametrization (only scale by -v^2 * dtw^2)
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel_ginsu[iGpu];
			scaleReflectivityTauGinsuSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
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

	// Transfer slice 2*host_hExt1+1 from host to device
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(2*host_hExt1+1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
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

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2*host_hExt1+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}

		}
		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2*host_hExt1+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}
		// Last part of the propagation
		else {
			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

		// QC
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefieldTau[iGpu][0], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

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
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		}
	}

	// Wait for all jobs on GPU to be done
	cuda_call(cudaDeviceSynchronize());

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));
	// cuda_call(cudaStreamDestroy(transferStreamH2D[iGpu]));

}

// Offset
void BornHxHyShotsFwdGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);
	// cudaStreamCreate(&transferStreamH2D[iGpu]);

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

	// Transfer extended model from host to device
	// cuda_call(cudaMemcpyAsync(dev_modelBornExt[iGpu], model, host_nModelExt*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2D[iGpu]));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

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

	// Allocate and initialize data
  	// cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	// cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

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

			// Extract data
			// recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample

		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel, dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel, dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************************************************************/
	/********************** Scattered wavefield computation *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

	// Copy model to device
	// cuda_call(cudaStreamSynchronize(transferStreamH2D[iGpu]));
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nModelExt*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// double *dummySlice, *dummyModel, *dummyData;
	// dummySlice = new double[host_nVel];
	// dummyModel = new double[host_nModelExt];
	// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];
	// cudaMemcpy(dummyModel, dev_modelBornExt[iGpu], host_nModelExt*sizeof(double), cudaMemcpyDeviceToHost);
	// std::cout << "Dummy model min #0 = " << *std::min_element(dummyModel,dummyModel+host_nModelExt) << std::endl;
	// std::cout << "Dummy model max #0 = " << *std::max_element(dummyModel,dummyModel+host_nModelExt) << std::endl;

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	if (slowSquare == 0 ){
		// Apply scalings to reflectivity coming from for the wave-equation linearization: (1) 2.0*1/v^3
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride;
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel;
				scaleReflectivityLinHxHy_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2);
			}
		}
	} else {
		// Apply scalings to reflectivity in slowness squared -> multiply by (-1)
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride;
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel;
				scaleReflectivityLinHxHySlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], extStride1, extStride2);
			}
		}
	}

	/************************ Streaming stuff starts **************************/

	// Copy wavefield time-slice its = 0: RAM -> pinned -> dev_pStream -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nVel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}
	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nVel, host_nVel*sizeof(double)); // -> this should be done with transfer stream
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #0 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #0 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #1 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #1 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #2 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #2 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			// dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
			dampCosineEdge_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// cudaMemcpy(dummyModel, dev_p0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
			// std::cout << "Dummy p0 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
			// std::cout << "Dummy p0 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

			// Extract data
			recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// cudaMemcpy(dummyData, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy data min = " << *std::min_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
		// std::cout << "Dummy data max = " << *std::max_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
		// std::cout << "---------------------" << std::endl;

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu]));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	}

	// // Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Offset Ginsu
void BornHxHyShotsFwdGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);
	// cudaStreamCreate(&transferStreamH2D[iGpu]);

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

	// Transfer extended model from host to device
	// cuda_call(cudaMemcpyAsync(dev_modelBornExt[iGpu], model, host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2D[iGpu]));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

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

	// Allocate and initialize data
  	// cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	// cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

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
			dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample

		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel_ginsu[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************************************************************/
	/********************** Scattered wavefield computation *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Copy model to device
	// cuda_call(cudaStreamSynchronize(transferStreamH2D[iGpu]));
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	if (slowSquare == 0){
		// Apply scalings to reflectivity coming from for the wave-equation linearization: (1) 2.0*1/v^3
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
				scaleReflectivityLinHxHyGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2, iGpu);
			}
		}
	} else {
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
				scaleReflectivityLinHxHyGinsuSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], extStride1, extStride2, iGpu);
			}
		}
	}

	/************************ Streaming stuff starts **************************/

	// Copy wavefield time-slice its = 0: RAM -> pinned -> dev_pStream -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nVel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
		}
	}
	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFdGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nVel, host_nVel*sizeof(double)); // -> this should be done with transfer stream
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
			}
		}

		// imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #0 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #0 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #1 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #1 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFdGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu], iGpu);

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #2 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #2 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// cudaMemcpy(dummyModel, dev_p0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
			// std::cout << "Dummy p0 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
			// std::cout << "Dummy p0 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

			// Extract data
			recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// cudaMemcpy(dummyData, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy data min = " << *std::min_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
		// std::cout << "Dummy data max = " << *std::max_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
		// std::cout << "---------------------" << std::endl;

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	}

	// // Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));
	// cuda_call(cudaStreamDestroy(transferStreamH2D[iGpu]));

}

/******************************* Free surface *********************************/
// Time
void BornTauFreeSurfaceShotsFwdGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);
	// cudaStreamCreate(&transferStreamH2D[iGpu]);

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

	// Transfer extended model from host to device
	// cuda_call(cudaMemcpyAsync(dev_modelBornExt[iGpu], model, host_nModelExt*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2D[iGpu]));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

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

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu]);

			// Step forward
			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel, dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel, dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************************************************************/
	/********************** Scattered wavefield computation *******************/
	/**************************************************************************/

	// double *dummySliceLeft, *dummySliceRight;
	// dummySliceLeft = new double[host_nVel];
	// dummySliceRight = new double[host_nVel];

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(double)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(double))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// // Copy slice from RAM -> pinned
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+iExt*host_nVel, host_nVel*sizeof(double));

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice[iGpu]+iExt*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice));
		}
	}

	// Copy model to device
	// cuda_call(cudaStreamSynchronize(transferStreamH2D[iGpu]));
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nModelExt*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	if (slowSquare == 0){
		// Apply both scalings to reflectivity coming from for the wave-equation linearization and the finite-difference propagation (we can do them simultaneously because the extension is in time-lags):
		// scale = 2.0 * 1/v^3 * v^2 * dtw^2
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel;
			scaleReflectivityTau_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride);
		}
	} else {
		// Apply scaling for slowness squared parametrization (only scale by -v^2 * dtw^2)
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel;
			scaleReflectivityTauSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_vel2Dtw2[iGpu], extStride);
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

	// Transfer slice 2*host_hExt1+1 from host to device
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(2*host_hExt1+1)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
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

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2*host_hExt1+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// Last part of the propagation
		else {
			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// QC
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefieldTau[iGpu][0], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu]));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

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
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		}
	}

	// Wait for all jobs on GPU to be done
	cuda_call(cudaDeviceSynchronize());

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));
    // cuda_call(cudaStreamDestroy(transferStreamH2D[iGpu]));

}

// Time Ginsu
void BornTauFreeSurfaceShotsFwdGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);
	// cudaStreamCreate(&transferStreamH2D[iGpu]);

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

	// Transfer extended model from host to device
	// cuda_call(cudaMemcpyAsync(dev_modelBornExt[iGpu], model, host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2D[iGpu]));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

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

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu], iGpu);

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel_ginsu[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************************************************************/
	/********************** Scattered wavefield computation *******************/
	/**************************************************************************/

	// double *dummySliceLeft, *dummySliceRight;
	// dummySliceLeft = new double[host_nVel];
	// dummySliceRight = new double[host_nVel];

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(double)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel_ginsu[iGpu]*sizeof(double))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice[iGpu]+iExt*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice));
		}
	}

	// Copy model to device
	// cuda_call(cudaStreamSynchronize(transferStreamH2D[iGpu]));
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	if (slowSquare == 0){
		// Apply both scalings to reflectivity coming from for the wave-equation linearization and the finite-difference propagation (we can do them simultaneously because the extension is in time-lags):
		// scale = 2.0 * 1/v^3 * v^2 * dtw^2
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel_ginsu[iGpu];
			scaleReflectivityTauGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
		}
	} else {
		// Apply scaling for slowness squared parametrization (only scale by -v^2 * dtw^2)
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel_ginsu[iGpu];
			scaleReflectivityTauGinsuSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
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

	// Transfer slice 2*host_hExt1+1 from host to device
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(2*host_hExt1+1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
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

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2*host_hExt1+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2*host_hExt1+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

		// Last part of the propagation
		else {
			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

		// QC
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefieldTau[iGpu][0], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu], iGpu);

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

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
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		}
	}

	// Wait for all jobs on GPU to be done
	cuda_call(cudaDeviceSynchronize());

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));
    // cuda_call(cudaStreamDestroy(transferStreamH2D[iGpu]));

}

// Offset
void BornHxHyFreeSurfaceShotsFwdGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);
	// cudaStreamCreate(&transferStreamH2D[iGpu]);

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

	// Transfer extended model from host to device
	// cuda_call(cudaMemcpyAsync(dev_modelBornExt[iGpu], model, host_nModelExt*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2D[iGpu]));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel, dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel, dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************************************************************/
	/********************** Scattered wavefield computation *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

	// Copy model to device
	// cuda_call(cudaStreamSynchronize(transferStreamH2D[iGpu]));
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nModelExt*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// double *dummySlice, *dummyModel, *dummyData;
	// dummySlice = new double[host_nVel];
	// dummyModel = new double[host_nModelExt];
	// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];
	// cudaMemcpy(dummyModel, dev_modelBornExt[iGpu], host_nModelExt*sizeof(double), cudaMemcpyDeviceToHost);
	// std::cout << "Dummy model min #0 = " << *std::min_element(dummyModel,dummyModel+host_nModelExt) << std::endl;
	// std::cout << "Dummy model max #0 = " << *std::max_element(dummyModel,dummyModel+host_nModelExt) << std::endl;

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	if (slowSquare == 0){
		// Apply scalings to reflectivity coming from for the wave-equation linearization: (1) 2.0*1/v^3
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride;
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel;
				scaleReflectivityLinHxHy_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2);
			}
		}
	} else {
		// Apply scalings to reflectivity coming from for the wave-equation linearization in slowness squared: -1.0
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride;
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel;
				scaleReflectivityLinHxHySlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], extStride1, extStride2);
			}
		}
	}

	/************************ Streaming stuff starts **************************/

	// Copy wavefield time-slice its = 0: RAM -> pinned -> dev_pStream -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nVel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}
	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nVel, host_nVel*sizeof(double)); // -> this should be done with transfer stream
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #0 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #0 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #1 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #1 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #2 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #2 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

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

		// cudaMemcpy(dummyData, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy data min = " << *std::min_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
		// std::cout << "Dummy data max = " << *std::max_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
		// std::cout << "---------------------" << std::endl;

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu]));

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
    // cuda_call(cudaStreamDestroy(transferStreamH2D[iGpu]));

}

// Offset
void BornHxHyFreeSurfaceShotsFwdGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

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

	// Transfer extended model from host to device
	// cuda_call(cudaMemcpyAsync(dev_modelBornExt[iGpu], model, host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2D[iGpu]));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

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

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu], iGpu);

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel_ginsu[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************************************************************/
	/********************** Scattered wavefield computation *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Copy model to device
	// cuda_call(cudaStreamSynchronize(transferStreamH2D[iGpu]));
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// double *dummySlice, *dummyModel, *dummyData;
	// dummySlice = new double[host_nVel_ginsu[iGpu]];
	// dummyModel = new double[host_nModelExt_ginsu[iGpu]];
	// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];

	// cudaMemcpy(dummyModel, dev_modelBornExt[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost);
	// std::cout << "Dummy model min #0 = " << *std::min_element(dummyModel,dummyModel+host_nModelExt_ginsu[iGpu]) << std::endl;
	// std::cout << "Dummy model max #0 = " << *std::max_element(dummyModel,dummyModel+host_nModelExt_ginsu[iGpu]) << std::endl;

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	if (slowSquare == 0){
		// Apply scalings to reflectivity coming from for the wave-equation linearization: (1) 2.0*1/v^3
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
				scaleReflectivityLinHxHyGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2, iGpu);
			}
		}
	} else {
		// Apply scalings to reflectivity coming from for the wave-equation linearization parametrized in slowness squared
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
				scaleReflectivityLinHxHyGinsuSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], extStride1, extStride2, iGpu);
			}
		}
	}

	/************************ Streaming stuff starts **************************/

	// Copy wavefield time-slice its = 0: RAM -> pinned -> dev_pStream -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nVel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
		}
	}
	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFdGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nVel, host_nVel*sizeof(double)); // -> this should be done with transfer stream
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
			}
		}

		// imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #0 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #0 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #1 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #1 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFdGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu], iGpu);

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy secondary source min #0 = " << *std::min_element(dummySlice,dummySlice+host_nVel_ginsu[iGpu]) << std::endl;
		// std::cout << "Dummy secondary max #0 = " << *std::max_element(dummySlice,dummySlice+host_nVel_ginsu[iGpu]) << std::endl;

		// cudaMemcpy(dummySlice, dev_pRight[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy slice #2 min = " << *std::min_element(dummySlice,dummySlice+host_nVel) << std::endl;
		// std::cout << "Dummy slice #2 max = " << *std::max_element(dummySlice,dummySlice+host_nVel) << std::endl;

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu], iGpu);

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
			dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Extract data
			recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// cudaMemcpy(dummyData, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost);
		// std::cout << "Dummy data min = " << *std::min_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
		// std::cout << "Dummy data max = " << *std::max_element(dummyData,dummyData+nReceiversReg*host_nts) << std::endl;
		// std::cout << "---------------------" << std::endl;

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

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

/******************************************************************************/
/************************** Born extended adjoint *****************************/
/******************************************************************************/
// Time
void BornTauShotsAdjGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){


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
	  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
	  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
		cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

		// Initialize time-slices for transfer to host's pinned memory
	  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

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
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
			// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

			// if (its>0) {
			// 	// Standard library
			// 	std::memcpy(srcWavefieldDts+(its-1)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
			//
			// }
			// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
			// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			// Asynchronous transfer of pStream => pin [its] [transfer]
			// Launch the transfer while we compute the next coarse time sample
			cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel, dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

			// Switch pointers
			dev_pTemp[iGpu] = dev_pLeft[iGpu];
			dev_pLeft[iGpu] = dev_pRight[iGpu];
			dev_pRight[iGpu] = dev_pTemp[iGpu];
			dev_pTemp[iGpu] = NULL;
	  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
		}

		// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
		// The CPU has stored the wavefield values ranging from 0,...,nts-3

		// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

		// Wait until pLeft -> pStream is done
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));

		cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel, dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));


		/**************************************************************************/
		/************************ Adjoint wavefield computation *******************/
		/**************************************************************************/

		// Reset the time slices to zero
		cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
	  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
		cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
		cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
	  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

		// Allocate the slices in the allocation function
		// Start loading the slices before writing them on RAM at the end of the source computation

		// Allocate time-slices from 0,...,4*hExt1 (included)
		for (int iExt=4*host_hExt1; iExt>-1; iExt--){

			// Allocate source wavefield slice
			// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(double)));
			cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(double))); // Useless

			// Load the source time-slices from its = 2*hExt1,...,4*hExt1 (included)
			if (iExt > 2*host_hExt1-1){

				// Transfer from pinned -> GPU
				cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice[iGpu]+(host_nts-1+iExt-4*host_hExt1)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice));
			}
		}

		// Initialize model to zero
		cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nModelExt*sizeof(double)));

		// Allocate and copy data from host -> device
	  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
		cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

		/************************ Streaming stuff starts **************************/
		// Bounds for time-lag imaging condition
		int iExtMin, iExtMax;

		// Start propagating adjoint wavefield
		for (int its = host_nts-2; its > -1; its--){

			// First part of the adjoint propagation
			// if (its > host_nts-2-2*host_hExt1){
			//
			// 	// Copy slice from RAM -> pinned for its-2hExt1
			// 	std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its-2*host_hExt1)*host_nVel, host_nVel*sizeof(double));
			//
			// 	// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
			// 	cuda_call(cudaStreamSynchronize(compStream[iGpu]));
			//
			// 	// Copy slice its from pin -> pStream
			// 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
			//
			// }

			// First and middle part of the adjoint propagation
			// else if (its <= host_nts-2-2*host_hExt1 && its >= 2*host_hExt1){
			if (its > 2*host_hExt1-1){

				// Copy slice from RAM -> pinned for its-2hExt1
				// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its-2*host_hExt1)*host_nVel, host_nVel*sizeof(double));

				// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
				cuda_call(cudaStreamSynchronize(compStream[iGpu]));

				// Copy slice its from pin -> pStream
				cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its-2*host_hExt1)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

			}

			// Step
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

			// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
			// scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

			// Lower bound for imaging condition at its+1
			iExtMin = (its+2-host_nts)/2;
			iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

			// Upper bound for imaging condition at its+1
			iExtMax = (its+1)/2;
			iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

			// First part of adjoint propagation
			if (its > host_nts-2-2*host_hExt1){

				// Apply imaging condition for its+1
				for (int iExt=iExtMin; iExt<iExtMax; iExt++){

					// Compute index for time-slice
					int iSlice = 6*host_hExt1 + 2 + its - 2*iExt - host_nts;

					// Apply imaging condition for its+1
					imagingTauAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
				}

				// Wait until transfer stream has finished copying slice its from pinned -> pStream
				cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

				// Copy source wavefield slice its-2hExt1 to dev_pSourceWavefieldTau[]
				cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][2*host_hExt1-host_nts+its+1], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
			}

			// Middle part of adjoint propagation
			else if (its <= host_nts-2-2*host_hExt1 && its >= 2*host_hExt1) {

				// Apply imaging condition for its+1
				for (int iExt=iExtMin; iExt<iExtMax; iExt++){

					// Compute index for time-slice
					int iSlice = 4*host_hExt1 - 2*iExt;

					// Apply imaging condition for its+1
					imagingTauAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
				}

				// Wait until transfer stream has finished copying slice its from pinned -> pStream
				cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

				// Switch wavefield pointers
				dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][4*host_hExt1];
				for (int iExt=4*host_hExt1; iExt>0; iExt--){
					dev_pSourceWavefieldTau[iGpu][iExt] = dev_pSourceWavefieldTau[iGpu][iExt-1];
				}
				dev_pSourceWavefieldTau[iGpu][0] = dev_pTempTau[iGpu];
				dev_pTempTau[iGpu] = NULL;

				// Copy source wavefield slice its-2hExt1 to dev_pSourceWavefieldTau[]
				cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

			}

			// Last part of adjoint propagation
			else {

				// Apply imaging condition for its+1
				for (int iExt=iExtMin; iExt<iExtMax; iExt++){

					// Compute index for time-slice
					int iSlice = (its+1) - 2 * (iExt-host_hExt1);

					// Apply imaging condition for its+1
					imagingTauAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
				}
			}

			// Switch pointers for secondary source
			dev_pTemp[iGpu] = dev_pRight[iGpu];
			dev_pRight[iGpu] = dev_pLeft[iGpu];
			dev_pLeft[iGpu] = dev_pTemp[iGpu];
			dev_pTemp[iGpu] = NULL;
			cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu]));

		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme for its = 0
		// scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Lower bound for imaging condition at its = 0
		int its = 0;
		iExtMin = (its+1-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its = 0
		iExtMax = its/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// Apply imaging condition for its = 0
		for (int iExt=iExtMin; iExt<iExtMax; iExt++){

			// Compute index for time-slice
			int iSlice = its - 2 * (iExt-host_hExt1);

			imagingTauAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
		}

		if (slowSquare == 0){
			// Apply both scalings to reflectivity coming from for the wave-equation linearization and the finite-difference propagation (we can do them simultaneously because the extension is in time-lags):
			// scale = 2.0 * 1/v^3 * v^2 * dtw^2
			for (int iExt=0; iExt<host_nExt1; iExt++){
				long long extStride = iExt * host_nVel;
				scaleReflectivityTau_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride);
			}
		} else {
			// Apply scaling for slowness squared parametrization (only scale by -v^2 * dtw^2)
			for (int iExt=0; iExt<host_nExt1; iExt++){
				long long extStride = iExt * host_nVel;
				scaleReflectivityTauSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_vel2Dtw2[iGpu], extStride);
			}
		}

		cuda_call(cudaDeviceSynchronize());

		// Copy model back to host
		cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nModelExt*sizeof(double), cudaMemcpyDeviceToHost));

		/*************************** Memory deallocation **************************/
		// Deallocate the array for sources/receivers' positions
	    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
	    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts[iGpu]));
	    cuda_call(cudaStreamDestroy(compStream[iGpu]));
	    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Time Ginsu
void BornTauShotsAdjGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

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
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

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
			dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pLeft and dev_pRight
			// interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel_ginsu[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/**************************************************************************/
	/************************ Adjoint wavefield computation *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Allocate the slices in the allocation function
	// Start loading the slices before writing them on RAM at the end of the source computation

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=4*host_hExt1; iExt>-1; iExt--){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(double)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel_ginsu[iGpu]*sizeof(double))); // Useless

		// Load the source time-slices from its = 2*hExt1,...,4*hExt1 (included)
		if (iExt > 2*host_hExt1-1){

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice[iGpu]+(host_nts-1+iExt-4*host_hExt1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice));
		}
	}

	// Initialize model to zero
	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nModelExt_ginsu[iGpu]*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	/************************ Streaming stuff starts **************************/
	// Bounds for time-lag imaging condition
	int iExtMin, iExtMax;

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// First part of the adjoint propagation
		// if (its > host_nts-2-2*host_hExt1){
		//
		// 	// Copy slice from RAM -> pinned for its-2hExt1
		// 	std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its-2*host_hExt1)*host_nVel, host_nVel*sizeof(double));
		//
		// 	// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		// 	cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		//
		// 	// Copy slice its from pin -> pStream
		// 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
		//
		// }

		// First and middle part of the adjoint propagation
		// else if (its <= host_nts-2-2*host_hExt1 && its >= 2*host_hExt1){
		if (its > 2*host_hExt1-1){

			// Copy slice from RAM -> pinned for its-2hExt1
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its-2*host_hExt1)*host_nVel, host_nVel*sizeof(double));

			// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));

			// Copy slice its from pin -> pStream
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its-2*host_hExt1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		}

		// Step
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

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		// scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of adjoint propagation
		if (its > host_nts-2-2*host_hExt1){

			// Apply imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Compute index for time-slice
				int iSlice = 6*host_hExt1 + 2 + its - 2*iExt - host_nts;

				// Apply imaging condition for its+1
				imagingTauAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}

			// Wait until transfer stream has finished copying slice its from pinned -> pStream
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy source wavefield slice its-2hExt1 to dev_pSourceWavefieldTau[]
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][2*host_hExt1-host_nts+its+1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		}

		// Middle part of adjoint propagation
		else if (its <= host_nts-2-2*host_hExt1 && its >= 2*host_hExt1) {

			// Apply imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Compute index for time-slice
				int iSlice = 4*host_hExt1 - 2*iExt;

				// Apply imaging condition for its+1
				imagingTauAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}

			// Wait until transfer stream has finished copying slice its from pinned -> pStream
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][4*host_hExt1];
			for (int iExt=4*host_hExt1; iExt>0; iExt--){
				dev_pSourceWavefieldTau[iGpu][iExt] = dev_pSourceWavefieldTau[iGpu][iExt-1];
			}
			dev_pSourceWavefieldTau[iGpu][0] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Copy source wavefield slice its-2hExt1 to dev_pSourceWavefieldTau[]
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		}

		// Last part of adjoint propagation
		else {

			// Apply imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Compute index for time-slice
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);

				// Apply imaging condition for its+1
				imagingTauAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme for its = 0
	// scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

	// Lower bound for imaging condition at its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

	// Upper bound for imaging condition at its = 0
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Apply imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){

		// Compute index for time-slice
		int iSlice = its - 2 * (iExt-host_hExt1);

		imagingTauAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
	}

	if (slowSquare == 0){
		// Apply both scalings to reflectivity coming from for the wave-equation linearization and the finite-difference propagation (we can do them simultaneously because the extension is in time-lags):
		// scale = 2.0 * 1/v^3 * v^2 * dtw^2
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel_ginsu[iGpu];
			scaleReflectivityTauGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
		}
	} else {
		// Apply scaling for slowness squared parametrization (only scale by -v^2 * dtw^2)
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel_ginsu[iGpu];
			scaleReflectivityTauGinsuSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
		}
	}

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Offset
void BornHxHyShotsAdjGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

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
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

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

		/* Note: At that point pLeft [its] is ready to be transfered back to host */
		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
		// At that point, the value of pStream has been transfered back to host pinned memory

		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel, dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel, dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	/********************** Adjoint wavefield computation *********************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize model to zero
	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nModelExt*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on devic

	// double *dummySlice, *dummyModel, *dummyData;
	// dummySlice = new double[host_nVel];
	// dummyModel = new double[host_nModelExt];
	// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];

	/************************ Streaming stuff starts **************************/
	// Copy source wavefield slice nts-1 from pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy slice its from RAM -> pinned
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+its*host_nVel, host_nVel*sizeof(double));

		// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));

		// Copy slice its from pin -> pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+its*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

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

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Apply imaging condition for its+1
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// Apply imaging condition for its+1
		// for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
		// 	long long extStride = (ihx + host_hExt1) * host_nVel;
		// 	imagingHxAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, extStride);
		// }

		// imagingAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

		// Copy source wavefield slice its to dev_pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu]));

	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

	// Apply imaging condition for its+1
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}

	// for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
	// 	int extStride = (ihx + host_hExt1) * host_nVel;
	// 	imagingHxAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, extStride);
	// }


	// for (int iExt1=0; iExt1<host_nExt1; iExt1++){
	// 	long long extStride = iExt1 * host_nVel;
	// 	scaleReflectivityLin_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride);
	// }

	if (slowSquare == 0){
		// Apply scalings to reflectivity coming from for the wave-equation linearization: (1) 2.0*1/v^3
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride;
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel;
				scaleReflectivityLinHxHy_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2);
			}
		}
	} else {
		// Apply scalings to reflectivity coming from for the wave-equation linearization in slowness squared
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride;
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel;
				scaleReflectivityLinHxHySlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], extStride1, extStride2);
			}
		}

	}

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nModelExt*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

void BornHxHyShotsAdjGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

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
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

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
			dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pLeft and dev_pRight
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel_ginsu[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/********************** Adjoint wavefield computation *********************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Initialize model to zero
	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nModelExt_ginsu[iGpu]*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on devic

	// double *dummySlice, *dummyModel, *dummyData;
	// dummySlice = new double[host_nVel];
	// dummyModel = new double[host_nModelExt];
	// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];

	/************************ Streaming stuff starts **************************/
	// Copy source wavefield slice nts-1 from pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy slice its from RAM -> pinned
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+its*host_nVel, host_nVel*sizeof(double));

		// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));

		// Copy slice its from pin -> pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+its*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

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

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFdGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu], iGpu);

		// Apply imaging condition for its+1
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
			}
		}

		// Apply imaging condition for its+1
		// for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
		// 	long long extStride = (ihx + host_hExt1) * host_nVel;
		// 	imagingHxAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, extStride);
		// }

		// imagingAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

		// Copy source wavefield slice its to dev_pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFdGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Apply imaging condition for its+1
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
		}
	}

	// for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
	// 	int extStride = (ihx + host_hExt1) * host_nVel;
	// 	imagingHxAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, extStride);
	// }


	// for (int iExt1=0; iExt1<host_nExt1; iExt1++){
	// 	long long extStride = iExt1 * host_nVel;
	// 	scaleReflectivityLin_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride);
	// }

	if (slowSquare == 0){
		// Apply scalings to reflectivity coming from for the wave-equation linearization: (1) 2.0*1/v^3
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
				scaleReflectivityLinHxHyGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2, iGpu);
			}
		}
	} else {
		// Apply scalings to reflectivity coming from for the wave-equation linearization in slowness squared
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
				scaleReflectivityLinHxHyGinsuSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], extStride1, extStride2, iGpu);
			}
		}
	}

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

/******************************* Free surface *********************************/
// Time
void BornTauFreeSurfaceShotsAdjGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

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
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

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
			// dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
			dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pLeft and dev_pRight
			// interpFineToCoarseSlice_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);
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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel, dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel, dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));


	/**************************************************************************/
	/************************ Adjoint wavefield computation *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

	// Allocate the slices in the allocation function
	// Start loading the slices before writing them on RAM at the end of the source computation

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=4*host_hExt1; iExt>-1; iExt--){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(double)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel*sizeof(double))); // Useless

		// Load the source time-slices from its = 2*hExt1,...,4*hExt1 (included)
		if (iExt > 2*host_hExt1-1){

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice[iGpu]+(host_nts-1+iExt-4*host_hExt1)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice));
		}
	}

	// Initialize model to zero
	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nModelExt*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	/************************ Streaming stuff starts **************************/
	// Bounds for time-lag imaging condition
	int iExtMin, iExtMax;

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// First part of the adjoint propagation
		// if (its > host_nts-2-2*host_hExt1){
		//
		// 	// Copy slice from RAM -> pinned for its-2hExt1
		// 	std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its-2*host_hExt1)*host_nVel, host_nVel*sizeof(double));
		//
		// 	// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		// 	cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		//
		// 	// Copy slice its from pin -> pStream
		// 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
		//
		// }

		// First and middle part of the adjoint propagation
		// else if (its <= host_nts-2-2*host_hExt1 && its >= 2*host_hExt1){
		if (its > 2*host_hExt1-1){

			// Copy slice from RAM -> pinned for its-2hExt1
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its-2*host_hExt1)*host_nVel, host_nVel*sizeof(double));

			// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));

			// Copy slice its from pin -> pStream
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its-2*host_hExt1)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		}

		// Step
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjFreeSurfaceGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject data
			interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Damp wavefields
			dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			interpFineToCoarseSlice_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		// scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of adjoint propagation
		if (its > host_nts-2-2*host_hExt1){

			// Apply imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Compute index for time-slice
				int iSlice = 6*host_hExt1 + 2 + its - 2*iExt - host_nts;

				// Apply imaging condition for its+1
				imagingTauAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}

			// Wait until transfer stream has finished copying slice its from pinned -> pStream
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy source wavefield slice its-2hExt1 to dev_pSourceWavefieldTau[]
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][2*host_hExt1-host_nts+its+1], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		}

		// Middle part of adjoint propagation
		else if (its <= host_nts-2-2*host_hExt1 && its >= 2*host_hExt1) {

			// Apply imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Compute index for time-slice
				int iSlice = 4*host_hExt1 - 2*iExt;

				// Apply imaging condition for its+1
				imagingTauAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}

			// Wait until transfer stream has finished copying slice its from pinned -> pStream
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][4*host_hExt1];
			for (int iExt=4*host_hExt1; iExt>0; iExt--){
				dev_pSourceWavefieldTau[iGpu][iExt] = dev_pSourceWavefieldTau[iGpu][iExt-1];
			}
			dev_pSourceWavefieldTau[iGpu][0] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Copy source wavefield slice its-2hExt1 to dev_pSourceWavefieldTau[]
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		}

		// Last part of adjoint propagation
		else {

			// Apply imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Compute index for time-slice
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);

				// Apply imaging condition for its+1
				imagingTauAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
			}
		}

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu]));

	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme for its = 0
	// scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

	// Lower bound for imaging condition at its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

	// Upper bound for imaging condition at its = 0
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Apply imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){

		// Compute index for time-slice
		int iSlice = its - 2 * (iExt-host_hExt1);

		imagingTauAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt);
	}

	if (slowSquare == 0){
		// Apply both scalings to reflectivity coming from for the wave-equation linearization and the finite-difference propagation (we can do them simultaneously because the extension is in time-lags):
		// scale = 2.0 * 1/v^3 * v^2 * dtw^2
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel;
			scaleReflectivityTau_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride);
		}
	} else {
		// Apply scaling for slowness squared parametrization (only scale by -v^2 * dtw^2)
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel;
			scaleReflectivityTauSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_vel2Dtw2[iGpu], extStride);
		}
	}

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nModelExt*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Time Ginsu
void BornTauFreeSurfaceShotsAdjGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

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
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

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

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu], iGpu);

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel_ginsu[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));


	/**************************************************************************/
	/************************ Adjoint wavefield computation *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Allocate the slices in the allocation function
	// Start loading the slices before writing them on RAM at the end of the source computation

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=4*host_hExt1; iExt>-1; iExt--){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel*sizeof(double)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel_ginsu[iGpu]*sizeof(double))); // Useless

		// Load the source time-slices from its = 2*hExt1,...,4*hExt1 (included)
		if (iExt > 2*host_hExt1-1){

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice[iGpu]+(host_nts-1+iExt-4*host_hExt1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice));
		}
	}

	// Initialize model to zero
	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nModelExt_ginsu[iGpu]*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	/************************ Streaming stuff starts **************************/
	// Bounds for time-lag imaging condition
	int iExtMin, iExtMax;

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// First part of the adjoint propagation
		// if (its > host_nts-2-2*host_hExt1){
		//
		// 	// Copy slice from RAM -> pinned for its-2hExt1
		// 	std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its-2*host_hExt1)*host_nVel, host_nVel*sizeof(double));
		//
		// 	// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		// 	cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		//
		// 	// Copy slice its from pin -> pStream
		// 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
		//
		// }

		// First and middle part of the adjoint propagation
		// else if (its <= host_nts-2-2*host_hExt1 && its >= 2*host_hExt1){
		if (its > 2*host_hExt1-1){

			// Copy slice from RAM -> pinned for its-2hExt1
			// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its-2*host_hExt1)*host_nVel, host_nVel*sizeof(double));

			// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
			cuda_call(cudaStreamSynchronize(compStream[iGpu]));

			// Copy slice its from pin -> pStream
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+(its-2*host_hExt1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		}

		// Step
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjFreeSurfaceGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject data
			interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Damp wavefields
			dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		// scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of adjoint propagation
		if (its > host_nts-2-2*host_hExt1){

			// Apply imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Compute index for time-slice
				int iSlice = 6*host_hExt1 + 2 + its - 2*iExt - host_nts;

				// Apply imaging condition for its+1
				imagingTauAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}

			// Wait until transfer stream has finished copying slice its from pinned -> pStream
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Copy source wavefield slice its-2hExt1 to dev_pSourceWavefieldTau[]
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][2*host_hExt1-host_nts+its+1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		}

		// Middle part of adjoint propagation
		else if (its <= host_nts-2-2*host_hExt1 && its >= 2*host_hExt1) {

			// Apply imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Compute index for time-slice
				int iSlice = 4*host_hExt1 - 2*iExt;

				// Apply imaging condition for its+1
				imagingTauAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}

			// Wait until transfer stream has finished copying slice its from pinned -> pStream
			cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][4*host_hExt1];
			for (int iExt=4*host_hExt1; iExt>0; iExt--){
				dev_pSourceWavefieldTau[iGpu][iExt] = dev_pSourceWavefieldTau[iGpu][iExt-1];
			}
			dev_pSourceWavefieldTau[iGpu][0] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Copy source wavefield slice its-2hExt1 to dev_pSourceWavefieldTau[]
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		}

		// Last part of adjoint propagation
		else {

			// Apply imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Compute index for time-slice
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);

				// Apply imaging condition for its+1
				imagingTauAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme for its = 0
	// scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

	// Lower bound for imaging condition at its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

	// Upper bound for imaging condition at its = 0
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Apply imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){

		// Compute index for time-slice
		int iSlice = its - 2 * (iExt-host_hExt1);

		imagingTauAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
	}

	if (slowSquare == 0){
		// Apply both scalings to reflectivity coming from for the wave-equation linearization and the finite-difference propagation (we can do them simultaneously because the extension is in time-lags):
		// scale = 2.0 * 1/v^3 * v^2 * dtw^2
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel_ginsu[iGpu];
			scaleReflectivityTauGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
		}
	} else {
		// Apply scaling for slowness squared parametrization (only scale by -v^2 * dtw^2)
		for (int iExt=0; iExt<host_nExt1; iExt++){
			long long extStride = iExt * host_nVel_ginsu[iGpu];
			scaleReflectivityTauGinsuSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_vel2Dtw2[iGpu], extStride, iGpu);
		}
	}

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Offset
void BornHxHyFreeSurfaceShotsAdjGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

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
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

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
			dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);


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

		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel, dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel, dev_pLeft[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	/********************** Adjoint wavefield computation *********************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));

	// Initialize model to zero
	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nModelExt*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on devic

	// double *dummySlice, *dummyModel, *dummyData;
	// dummySlice = new double[host_nVel];
	// dummyModel = new double[host_nModelExt];
	// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];

	/************************ Streaming stuff starts **************************/
	// Copy source wavefield slice nts-1 from pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy slice its from RAM -> pinned
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+its*host_nVel, host_nVel*sizeof(double));

		// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));

		// Copy slice its from pin -> pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+its*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjFreeSurfaceGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject data
			interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Damp wavefields
			dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			interpFineToCoarseSlice_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Apply imaging condition for its+1
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// Apply imaging condition for its+1
		// for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
		// 	long long extStride = (ihx + host_hExt1) * host_nVel;
		// 	imagingHxAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, extStride);
		// }

		// imagingAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

		// Copy source wavefield slice its to dev_pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel*sizeof(double), compStream[iGpu]));

	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

	// Apply imaging condition for its+1
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}

	// for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
	// 	int extStride = (ihx + host_hExt1) * host_nVel;
	// 	imagingHxAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, extStride);
	// }


	// for (int iExt1=0; iExt1<host_nExt1; iExt1++){
	// 	long long extStride = iExt1 * host_nVel;
	// 	scaleReflectivityLin_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride);
	// }

	if (slowSquare == 0){
		// Apply scalings to reflectivity coming from for the wave-equation linearization: (1) 2.0*1/v^3
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride;
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel;
				scaleReflectivityLinHxHy_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2);
			}
		}
	} else {
		// Apply scalings to reflectivity coming from for the wave-equation linearization: (1) 2.0*1/v^3
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride;
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel;
				scaleReflectivityLinHxHySlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], extStride1, extStride2);
			}
		}

	}

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nModelExt*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

// Offset
void BornHxHyFreeSurfaceShotsAdjGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId){

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
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

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

			// Apply free surface conditions for Laplacian
			setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface, 0, compStream[iGpu]>>>(dev_p1[iGpu], iGpu);

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
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
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]

		// if (its>0) {
		// 	// Standard library
		// 	std::memcpy(srcWavefieldDts+(its-1)*host_nVel, pin_wavefieldSlice[iGpu], host_nVel*sizeof(double));
		//
		// }
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu]+its*host_nVel_ginsu[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/********************** Adjoint wavefield computation *********************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double)));

	// Initialize model to zero
	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nModelExt_ginsu[iGpu]*sizeof(double)));

	// Allocate and copy data from host -> device
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on devic

	// double *dummySlice, *dummyModel, *dummyData;
	// dummySlice = new double[host_nVel];
	// dummyModel = new double[host_nModelExt];
	// dummyData = new double[nReceiversReg*host_nts*sizeof(double)];

	/************************ Streaming stuff starts **************************/
	// Copy source wavefield slice nts-1 from pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy slice its from RAM -> pinned
		// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+its*host_nVel, host_nVel*sizeof(double));

		// Wait until compStream has done copying wavefield value its+1 from pStream -> dev_pSourceWavefield
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));

		// Copy slice its from pin -> pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu]+its*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjFreeSurfaceGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject data
			interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);

			// Damp wavefields
			dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32, dimBlock32, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFdGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu], iGpu);

		// Apply imaging condition for its+1
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
			}
		}

		// Apply imaging condition for its+1
		// for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
		// 	long long extStride = (ihx + host_hExt1) * host_nVel;
		// 	imagingHxAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, extStride);
		// }

		// imagingAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

		// Copy source wavefield slice its to dev_pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(double), compStream[iGpu]));

	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFdGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Apply imaging condition for its+1
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyAdjGinsuGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
		}
	}

	// for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
	// 	int extStride = (ihx + host_hExt1) * host_nVel;
	// 	imagingHxAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, extStride);
	// }


	// for (int iExt1=0; iExt1<host_nExt1; iExt1++){
	// 	long long extStride = iExt1 * host_nVel;
	// 	scaleReflectivityLin_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride);
	// }

	if (slowSquare == 0){
		// Apply scalings to reflectivity coming from for the wave-equation linearization: (1) 2.0*1/v^3
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
				scaleReflectivityLinHxHyGinsu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], extStride1, extStride2, iGpu);
			}
		}
	} else {
		// Apply scalings to reflectivity coming from for the wave-equation linearization: (1) 2.0*1/v^3
		for (int iExt2=0; iExt2<host_nExt2; iExt2++){
			long long extStride2 = iExt2 * host_extStride_ginsu[iGpu];
			for (int iExt1=0; iExt1<host_nExt1; iExt1++){
				long long extStride1 = iExt1 * host_nVel_ginsu[iGpu];
				scaleReflectivityLinHxHyGinsuSlowSquare_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBornExt[iGpu], extStride1, extStride2, iGpu);
			}
		}
	}

	cuda_call(cudaDeviceSynchronize());

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nModelExt_ginsu[iGpu]*sizeof(double), cudaMemcpyDeviceToHost));

	/*************************** Memory deallocation **************************/
	// Deallocate the array for sources/receivers' positions
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaStreamDestroy(compStream[iGpu]));
    cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}
