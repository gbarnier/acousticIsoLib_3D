#include <cstring>
#include <iostream>
#include "nonlinearShotsGpuFunctions_3D.h"
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
void initNonlinearGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

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
	host_minPad = minPad;

	/************************* ALLOCATE ARRAYS OF ARRAYS **********************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {

		// Time slices for FD stepping
		dev_p0 = new double*[nGpu];
		dev_p1 = new double*[nGpu];
		dev_temp1 = new double*[nGpu];

		// Data and model
		dev_modelRegDtw = new double*[nGpu];
		dev_dataRegDts = new double*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new long long*[nGpu];
		dev_receiversPositionReg = new long long*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new double*[nGpu];

		// Damping slice
		dev_dampingSlice = new double*[nGpu];

		// Debug model and data
		dev_modelDebug = new double*[nGpu];
		dev_dataDebug = new double*[nGpu];

		// Compute and transfer stream
		topStream = new cudaStream_t[nGpu];
		compStream = new cudaStream_t[nGpu];

	}

	/********************* COMPUTE LAPLACIAN COEFFICIENTS *********************/
	// Compute coefficients for 8th order central finite difference Laplacian
	double host_coeff[COEFF_SIZE] = get_coeffs((double)dz,(double)dx,(double)dy); // Stored on host

	/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
	// Time interpolation filter length / half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Subsampling parameter for time interpolation is too high ****" << std::endl;
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

	/************************* COMPUTE COSINE DAMPING COEFFICIENTS **********************/
	if (minPad>=PAD_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Padding value is too high ****" << std::endl;
		throw std::runtime_error("");
	}
	double cosDampingCoeff[minPad];

	// Cosine padding
	for (int iFilter=FAT; iFilter<FAT+minPad; iFilter++){
		double arg = M_PI / (1.0 * minPad) * 1.0 * (minPad-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		cosDampingCoeff[iFilter-FAT] = arg;
		// std::cout << "Damp array gpu [" << iFilter-FAT << "] = " << arg << std::endl;
	}

	// Check that the block size is consistent between parfile and "varDeclare_3D.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare.h file ****" << std::endl;
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
void allocateNonlinearGpu_3D(double *vel2Dtw2, int iGpu, int iGpuId){

	// Get GPU number
	cudaSetDevice(iGpuId);

	// Scaled velocity
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nModel*sizeof(double))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nModel*sizeof(double), cudaMemcpyHostToDevice));

	// Damping slice
	// cuda_call(cudaMalloc((void**) &dev_dampingSlice[iGpu], host_nModel*sizeof(double))); // Allocate scaled velocity model on device
	// cuda_call(cudaMemcpy(dev_dampingSlice[iGpu], vel2Dtw2, host_nModel*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate time slices on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nModel*sizeof(double))); // Allocate time slices on device (for the stepper)
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nModel*sizeof(double)));
}

// Init Ginsu
void initNonlinearGinsuGpu_3D(double dz, double dx, double dy, int nts, double dts, int sub, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nts = nts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;

	/************************ ALLOCATE ARRAYS OF ARRAYS ***********************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {

		// Time slices for FD stepping
		dev_p0 = new double*[nGpu];
		dev_p1 = new double*[nGpu];
		dev_temp1 = new double*[nGpu];

		// Data and model
		dev_modelRegDtw = new double*[nGpu];
		dev_dataRegDts = new double*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new long long*[nGpu];
		dev_receiversPositionReg = new long long*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new double*[nGpu];

		// Damping slice
		dev_dampingSlice = new double*[nGpu];

		// Debug model and data
		dev_modelDebug = new double*[nGpu];
		dev_dataDebug = new double*[nGpu];

		// Compute and transfer stream
		topStream = new cudaStream_t[nGpu];
		compStream = new cudaStream_t[nGpu];

	}

	/*********************** COMPUTE LAPLACIAN COEFFICIENTS *******************/
	// Compute coefficients for 8th order central finite difference Laplacian
	double host_coeff[COEFF_SIZE] = get_coeffs((double)dz,(double)dx,(double)dy); // Stored on host

	/********************** COMPUTE TIME-INTERPOLATION FILTER *****************/
	// Time interpolation filter length / half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (nGpu>N_GPU_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Number of requested GPUs exceeds number allowed for constant memory storage ****" << std::endl;
		throw std::runtime_error("");
	}

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Subsampling parameter for time interpolation is too high ****" << std::endl;
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

	// Check that the block size is consistent between parfile and "varDeclare_3D.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare.h file ****" << std::endl;
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

	// Time parameters
	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

}

// Allocate Ginsu
void allocateSetNonlinearGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, double alphaCos, double *vel2Dtw2, int iGpu, int iGpuId){

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
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Padding value is too high ****" << std::endl;
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
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare.h file ****" << std::endl;
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

	// Copy to global memory
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nModel_ginsu[iGpu]*sizeof(double))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nModel_ginsu[iGpu]*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate time slices on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nModel_ginsu[iGpu]*sizeof(double))); // Allocate time slices on device (for the stepper)
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nModel_ginsu[iGpu]*sizeof(double)));
}

// Deallocate
void deallocateNonlinearGpu_3D(int iGpu, int iGpuId){
	cudaSetDevice(iGpuId); // Set device number on GPU cluster
	cuda_call(cudaFree(dev_vel2Dtw2[iGpu])); // Deallocate scaled velocity
	cuda_call(cudaFree(dev_p0[iGpu]));
	cuda_call(cudaFree(dev_p1[iGpu]));
}

/****************************************************************************************/
/******************************* Nonlinear forward propagation **************************/
/****************************************************************************************/
void propShotsFwdGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate output on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize output on device

	// Time slices
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	// cuda_call(cudaMemset(dev_dampingSlice[iGpu], 0, host_nModel*sizeof(double)));

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Extraction grid size
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Timer
	std::clock_t start;
	double duration;
	start = std::clock();

	// Loop over coarse time samples
	for (long long its = 0; its < host_nts-1; its++){

		// Loop over sub loop
		for (long long it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			long long itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSourceLinear_3D<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu]));
			// kernel_exec(dampCosineEdge_3D_32<<<dimGridTest1, dimBlockTest1>>>(dev_p1[iGpu], dev_p0[iGpu]));
			// kernel_exec(dampCosineEdge_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));
			// kernel_exec(dampCosineEdge_3D_8<<<dimGridTest2, dimBlockTest2>>>(dev_p0[iGpu], dev_p1[iGpu]));
			// kernel_exec(dampCosineEdge_3DBenchmark<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_dampingSlice[iGpu]));

			// Extract and interpolate data
			kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration: " << duration << std::endl;

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsFwdGinsuGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate output on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize output on device

	// Time slices
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Laplacian grid and blocks
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Extraction grid size
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Timer
	std::clock_t start;
	double duration;
	start = std::clock();

	// Loop over coarse time samples
	for (long long its = 0; its < host_nts-1; its++){

		// Loop over sub loop
		for (long long it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			long long itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGinsuGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu));

			// Inject source
			kernel_exec(injectSourceLinear_3D<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu));

			// Extract and interpolate data
			kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration: " << duration << std::endl;

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsFwdGpu_3D_dampTest(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId, double *dampVolume) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate output on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize output on device

	// Time slices
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));
	// cuda_call(cudaMemset(dev_dampingSlice[iGpu], 0, host_nModel*sizeof(double)));

	// double *p0, *p1, *pDiff;
	// p0 = new double[host_nModel];
	// p1 = new double[host_nModel];
	// pDiff = new double[host_nModel];
	// std::memset(p0, 0, host_nModel*sizeof(double));
	// std::memset(p1, 0, host_nModel*sizeof(double));

	// Set p0 and p1 to zero
	// std::fill(p0, p0+host_nModel, 0.0);
	// std::fill(p1, p1+host_nModel, 0.0);
	//
	// std::cout << "p0 min = " << *std::min_element(p0, p0+host_nModel) << std::endl;
	// std::cout << "p0 max = " << *std::max_element(p0, p0+host_nModel) << std::endl;
	// std::cout << "p1 min = " << *std::min_element(p1, p1+host_nModel) << std::endl;
	// std::cout << "p1 max = " << *std::max_element(p1, p1+host_nModel) << std::endl;

	// Fill in p0 and p1 with ones
	// for (long long iy = FAT; iy < host_ny-FAT; iy++){
	// 	for (long long ix = FAT; ix < host_nx-FAT; ix++){
	// 		for (long long iz = FAT; iz < host_nz-FAT; iz++){
	// 			long long iGlobal = iy * host_nx * host_nz + ix * host_nz + iz;
	// 			p0[iGlobal] = 1.0;
	// 			p1[iGlobal] = 1.0;
	// 		}
	// 	}
	// }

	// cuda_call(cudaMemcpy(dev_p0[iGpu], p0, host_nModel*sizeof(double), cudaMemcpyHostToDevice));
	// cuda_call(cudaMemcpy(dev_p1[iGpu], p1, host_nModel*sizeof(double), cudaMemcpyHostToDevice));
	//
	// std::cout << "p0 min after = " << *std::min_element(p0,p0+host_nModel) << std::endl;
	// std::cout << "p0 max after = " << *std::max_element(p0,p0+host_nModel) << std::endl;
	// std::cout << "p1 min after = " << *std::min_element(p1,p1+host_nModel) << std::endl;
	// std::cout << "p1 max after = " << *std::max_element(p1,p1+host_nModel) << std::endl;

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Damping kernels for front / back
	int nBlockDamp = (host_minPad+BLOCK_SIZE_DAMP-1) / BLOCK_SIZE_DAMP;
	dim3 dimGridDampFront(nblocky, nBlockDamp);
	dim3 dimBlockDampFront(BLOCK_SIZE_X, BLOCK_SIZE_DAMP);
	std::cout << "host_minPad = " << host_minPad << std::endl;
	std::cout << "nBlockDamp = " << nBlockDamp << std::endl;
	std::cout << "BLOCK_SIZE_DAMP = " << BLOCK_SIZE_DAMP << std::endl;

	// Damping kernels for left / right
	dim3 dimGridDampLeft(nblockz, nBlockDamp);
	dim3 dimBlockDampLeft(BLOCK_SIZE_Y, BLOCK_SIZE_DAMP);

	// Damping kernels for top / bottom
	dim3 dimGridDampTop(nblocky, nblockz);
	dim3 dimBlockDampTop(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Extraction grid size
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Timer
	std::clock_t start;
	double duration;
	start = std::clock();

	// Damp front / back
	// kernel_exec(dampFrontBack_3D<<<dimGridDampFront, dimBlockDampFront>>>(dev_p0[iGpu], dev_p1[iGpu]));
	//
	// // Damp left / right
	// kernel_exec(dampLeftRight_3D<<<dimGridDampLeft, dimBlockDampLeft>>>(dev_p0[iGpu], dev_p1[iGpu]));
	//
	// // // Damp top / bottom
	// kernel_exec(dampTopBottom_3D<<<dimGridDampTop, dimBlockDampTop>>>(dev_p0[iGpu], dev_p1[iGpu]));


	// Loop over coarse time samples
	for (long long its = 0; its < host_nts-1; its++){

		// Loop over sub loop
		for (long long it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			long long itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSourceLinear_3D<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp front / back
			kernel_exec(dampFrontBack_3D<<<dimGridDampFront, dimBlockDampFront>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Damp left / right
			kernel_exec(dampLeftRight_3D<<<dimGridDampLeft, dimBlockDampLeft>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Damp top / bottom
			// kernel_exec(dampTopBottom_3D<<<dimGridDampTop, dimBlockDampTop>>>(dev_p0[iGpu], dev_p1[iGpu]));
			//
			// // Extract and interpolate data
			// kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Compare damp volumes CPU/GPU
	// cuda_call(cudaMemcpy(p0, dev_p0[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
	// cuda_call(cudaMemcpy(p1, dev_p1[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));
	//
	// for (long long iGlobal = 0; iGlobal < host_nModel; iGlobal++){
	//
	// 	pDiff[iGlobal] = p0[iGlobal] - dampVolume[iGlobal];
	//
	// }
	//
	// std::cout << "pDiff min after = " << *std::min_element(pDiff,pDiff+host_nModel) << std::endl;
	// std::cout << "pDiff max after = " << *std::max_element(pDiff,pDiff+host_nModel) << std::endl;

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	std::cout << "duration: " << duration << std::endl;

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsFwdFreeSurfaceGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate output on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize output on device

	// Time slices
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));

	// Laplacian grid and blocks
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

	// Extraction grid size
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Start propagation
	// Timer
	std::clock_t start;
	double duration;
	start = std::clock();

	// Loop over coarse time samples
	for (long long its = 0; its < host_nts-1; its++){
		// Loop over sub loop
		for (long long it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			long long itw = its * host_sub + it2;

			// Apply free surface conditions for Laplacian
			kernel_exec(setFreeSurfaceConditionFwdGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface>>>(dev_p1[iGpu]));

			// Step forward in time
			kernel_exec(stepFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSourceLinear_3D<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			// kernel_exec(dampCosineEdgeFreeSurface_3D<<<dimGrid, dimBlock>>>(dev_p1[iGpu], dev_p0[iGpu]));
			kernel_exec(dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract and interpolate data
			kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration: " << duration << std::endl;

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsFwdFreeSurfaceGinsuGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate output on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize output on device

	// Time slices
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Laplacian grid and blocks
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

	// Extraction grid size
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Start propagation
	// Loop over coarse time samples
	for (long long its = 0; its < host_nts-1; its++){
		// Loop over sub loop
		for (long long it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			long long itw = its * host_sub + it2;

			// Apply free surface conditions for Laplacian
			kernel_exec(setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface>>>(dev_p1[iGpu], iGpu));

			// Step forward in time
			kernel_exec(stepFwdGinsuGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu));

			// Inject source
			kernel_exec(injectSourceLinear_3D<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			// kernel_exec(dampCosineEdgeFreeSurfaceGinsu_3D<<<dimGrid, dimBlock>>>(dev_p1[iGpu], dev_p0[iGpu], iGpu));
			kernel_exec(dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32>>>(dev_p1[iGpu], dev_p0[iGpu], iGpu));

			// Extract and interpolate data
			kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

/****************************************************************************************/
/******************************* Nonlinear adjoint propagation **************************/
/****************************************************************************************/
void propShotsAdjGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate model on device
  	cuda_call(cudaMemset(dev_modelRegDtw[iGpu], 0, nSourcesReg*host_ntw*sizeof(double))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy data on device

	// Initialize time slices on device
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));

	// Grid and block dimensions for stepper
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Grid and block dimensions for data injection
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Timer
	std::clock_t start;
	double duration;
	start = std::clock();

	// Loop over coarse time samples
	for (int its = host_nts-2; its > -1; its--){
		// Loop over sub loop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward in time
			kernel_exec(stepAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			// kernel_exec(dampCosineEdge_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));
			kernel_exec(dampCosineEdge_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract model
			kernel_exec(recordSource_3D<<<1, nSourcesReg>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration: " << duration << std::endl;

	// Copy data back to host
	cuda_call(cudaMemcpy(modelRegDtw, dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate all slices
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsAdjGinsuGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate model on device
  	cuda_call(cudaMemset(dev_modelRegDtw[iGpu], 0, nSourcesReg*host_ntw*sizeof(double))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy data on device

	// Initialize time slices on device
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Grid and block dimensions for stepper
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny_ginsu[iGpu]-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Grid and block dimensions for data injection
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Loop over coarse time samples
	for (int its = host_nts-2; its > -1; its--){
		// Loop over sub loop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward in time
			kernel_exec(stepAdjGinsuGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu));

			// Inject data
			kernel_exec(interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu));

			// Extract model
			kernel_exec(recordSource_3D<<<1, nSourcesReg>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(modelRegDtw, dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate all slices
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsAdjFreeSurfaceGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&topStream[iGpu]);
	cudaStreamCreate(&compStream[iGpu]);

	// Create to synchornize top/body for free surface computation
	cudaEventCreate(&eventTopFreeSurface);
	cudaEventCreate(&eventBodyFreeSurface);
	cudaEventCreate(&compStreamDone);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate model on device
  	cuda_call(cudaMemset(dev_modelRegDtw[iGpu], 0, nSourcesReg*host_ntw*sizeof(double))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy data on device

	// Initialize time slices on device
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));

	// Grid and block dimensions for stepper
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimGridTop(1, nblocky);
	dim3 dimGridBody(nblockx-1, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Grid and block dimensions for data injection
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Timer
	std::clock_t start;
	double duration;
	start = std::clock();

	// std::cout << "Free surface" << std::endl;
	// Loop over coarse time samples
	for (int its = host_nts-2; its > -1; its--){
		// Loop over sub loop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step backward in time
			kernel_exec(stepAdjFreeSurfaceGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			// kernel_exec(dampCosineEdgeFreeSurface_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));
			kernel_exec(dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract model
			kernel_exec(recordSource_3D<<<1, nSourcesReg>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]));

			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// std::cout << "duration: " << duration << std::endl;

	// std::cout << "Free surface + top-body separation" << std::endl;
	// // Loop over coarse time samples
	// for (int its = host_nts-2; its > -1; its--){
	//
	// 	// Loop over sub loop
	// 	for (int it2 = host_sub-1; it2 > -1; it2--){
	//
	// 		// Compute fine time-step index
	// 		int itw = its * host_sub + it2;
	//
	// 		// Launch top free surface compuation
	// 		stepAdjFreeSurfaceGpu_3D<<<dimGridTop, dimBlock, 0, topStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
	// 		cudaEventRecord(eventTopFreeSurface, topStream[iGpu]);
	//
	// 		stepAdjBodyFreeSurfaceGpu_3D<<<dimGridBody, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
	// 		cudaStreamWaitEvent(compStream[iGpu], eventTopFreeSurface, 0);
	//
	// 		// Inject data
	// 		interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]);
	//
	// 		// Damp wavefield
	// 		dampCosineEdgeFreeSurface_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
	//
	// 		// Extract model
	// 		recordSource_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]);
	//
	// 		dev_temp1[iGpu] = dev_p0[iGpu];
	// 		dev_p0[iGpu] = dev_p1[iGpu];
	// 		dev_p1[iGpu] = dev_temp1[iGpu];
	// 		dev_temp1[iGpu] = NULL;
	//
	// 		cudaEventRecord(compStreamDone, compStream[iGpu]);
	// 		cudaStreamWaitEvent(topStream[iGpu], compStreamDone, 0);
	// 	}
	// }

	// Copy data back to host
	cuda_call(cudaMemcpy(modelRegDtw, dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate all slices
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsAdjFreeSurfaceGinsuGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&topStream[iGpu]);
	cudaStreamCreate(&compStream[iGpu]);

	// Create to synchornize top/body for free surface computation
	cudaEventCreate(&eventTopFreeSurface);
	cudaEventCreate(&eventBodyFreeSurface);
	cudaEventCreate(&compStreamDone);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate model on device
  	cuda_call(cudaMemset(dev_modelRegDtw[iGpu], 0, nSourcesReg*host_ntw*sizeof(double))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy data on device

	// Initialize time slices on device
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(double)));

	// Grid and block dimensions for stepper
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny_ginsu[iGpu]-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimGridTop(1, nblocky);
	dim3 dimGridBody(nblockx-1, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Grid and block dimensions for data injection
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Loop over coarse time samples
	for (int its = host_nts-2; its > -1; its--){
		// Loop over sub loop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step backward in time
			kernel_exec(stepAdjFreeSurfaceGinsuGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu));

			// Inject data
			kernel_exec(interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdgeFreeSurfaceGinsu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu));
			// kernel_exec(dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu));

			// Extract model
			kernel_exec(recordSource_3D<<<1, nSourcesReg>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]));

			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(modelRegDtw, dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate all slices
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}
